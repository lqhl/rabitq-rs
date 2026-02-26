//! Search algorithms with dynamic pruning
//!
//! Handles FastScan batch distance computation and dynamic query pruning.

use super::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

#[derive(Debug, Clone)]
struct HeapCandidate {
    id: u64,
    distance: f32,
}

#[derive(Debug, Clone)]
struct HeapEntry {
    candidate: HeapCandidate,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.candidate
            .distance
            .to_bits()
            .eq(&other.candidate.distance.to_bits())
            && self.candidate.id == other.candidate.id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.candidate.distance.total_cmp(&other.candidate.distance)
    }
}

#[derive(Debug, Default, Clone)]
struct PostingSearchStats {
    posting_lists_visited: usize,
    scanned_vectors: usize,
    skipped_by_lower_bound: usize,
    refined_vectors: usize,
    estimated_vectors: usize,
    posting_search_time_us: f64,
}

#[derive(Debug, Default, Clone)]
struct PostingSearchOutput {
    candidates: Vec<HeapCandidate>,
    stats: PostingSearchStats,
}

/// MSTG search diagnostics (P1 instrumentation)
#[derive(Debug, Default, Clone)]
pub struct SearchDiagnostics {
    pub selected_centroids: usize,
    pub bootstrap_centroids: usize,
    pub posting_lists_visited: usize,
    pub scanned_vectors: usize,
    pub skipped_by_lower_bound: usize,
    pub refined_vectors: usize,
    pub estimated_vectors: usize,
    pub posting_access_time_us: f64,
    pub posting_search_time_us: f64,
    pub posting_decode_overhead_us: f64,
}

impl SearchDiagnostics {
    #[inline]
    fn add_posting_stats(&mut self, stats: &PostingSearchStats) {
        self.posting_lists_visited += stats.posting_lists_visited;
        self.scanned_vectors += stats.scanned_vectors;
        self.skipped_by_lower_bound += stats.skipped_by_lower_bound;
        self.refined_vectors += stats.refined_vectors;
        self.estimated_vectors += stats.estimated_vectors;
        self.posting_search_time_us += stats.posting_search_time_us;
    }
}

impl MstgIndex {
    #[inline]
    fn current_distk(heap: &BinaryHeap<HeapEntry>, top_k: usize, fallback: f32) -> f32 {
        if heap.len() < top_k {
            fallback
        } else {
            heap.peek()
                .map(|entry| entry.candidate.distance.min(fallback))
                .unwrap_or(fallback)
        }
    }

    #[inline]
    fn merge_topk(
        heap: &mut BinaryHeap<HeapEntry>,
        candidates: impl IntoIterator<Item = HeapCandidate>,
        top_k: usize,
    ) {
        for c in candidates {
            heap.push(HeapEntry { candidate: c });
            if heap.len() > top_k {
                heap.pop();
            }
        }
    }

    fn search_internal(
        &self,
        query: &[f32],
        params: &SearchParams,
        diagnostics: Option<&mut SearchDiagnostics>,
    ) -> Vec<SearchResult> {
        use crate::fastscan::QueryContext as FastScanQueryContext;

        if params.top_k == 0 {
            return Vec::new();
        }

        // Step 1: Find candidate centroids
        let centroid_candidates = self.centroid_index.search(query, params.ef_search);

        // Step 2: Dynamic pruning
        let selected_centroids = self.dynamic_prune(&centroid_candidates, params.pruning_epsilon);
        if selected_centroids.is_empty() {
            return Vec::new();
        }

        // Step 3: Create query context once
        let ex_bits = self.config.rabitq_bits.saturating_sub(1);
        let mut query_ctx = FastScanQueryContext::new(query.to_vec(), ex_bits);

        // Build LUT once for all posting lists (if dimensions match)
        if !self.directory.is_empty() {
            let first_cid = self.directory.entries[0].cluster_id;
            self.posting_lists
                .with_posting_list(first_cid, &self.directory, |first_plist| {
                    query_ctx.build_lut(first_plist.padded_dim);
                });
        }

        let mut diagnostics = diagnostics;
        if let Some(diag) = diagnostics.as_deref_mut() {
            diag.selected_centroids = selected_centroids.len();
        }

        // Step 4: Two-stage scheduler
        //   - Stage A (bootstrap): serially process a few centroid posting lists to tighten distk.
        //   - Stage B: process the rest in parallel with thread-local heaps.
        let bootstrap_count = selected_centroids.len().min(8);
        let (bootstrap_centroids, remaining_centroids) =
            selected_centroids.split_at(bootstrap_count);

        if let Some(diag) = diagnostics.as_deref_mut() {
            diag.bootstrap_centroids = bootstrap_count;
        }

        let mut global_heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        for &cid in bootstrap_centroids {
            let distk_seed = Self::current_distk(&global_heap, params.top_k, f32::INFINITY);
            let t_access_start = Instant::now();
            let output = self
                .posting_lists
                .with_posting_list(cid, &self.directory, |plist| {
                    if plist.is_empty() {
                        PostingSearchOutput::default()
                    } else {
                        self.search_posting_list_fastscan(
                            &query_ctx,
                            plist,
                            &plist.batch_data,
                            params.top_k,
                            distk_seed,
                        )
                    }
                })
                .unwrap_or_default();
            let access_us = t_access_start.elapsed().as_secs_f64() * 1_000_000.0;

            if let Some(diag) = diagnostics.as_deref_mut() {
                diag.posting_access_time_us += access_us;
                diag.posting_decode_overhead_us +=
                    (access_us - output.stats.posting_search_time_us).max(0.0);
                diag.add_posting_stats(&output.stats);
            }

            Self::merge_topk(&mut global_heap, output.candidates, params.top_k);
        }

        let distk_seed = Self::current_distk(&global_heap, params.top_k, f32::INFINITY);

        let remaining_outputs: Vec<(PostingSearchOutput, f64)> = remaining_centroids
            .par_iter()
            .map(|&cid| {
                let t_access_start = Instant::now();
                let output = self
                    .posting_lists
                    .with_posting_list(cid, &self.directory, |plist| {
                        if plist.is_empty() {
                            PostingSearchOutput::default()
                        } else {
                            self.search_posting_list_fastscan(
                                &query_ctx,
                                plist,
                                &plist.batch_data,
                                params.top_k,
                                distk_seed,
                            )
                        }
                    })
                    .unwrap_or_default();
                let access_us = t_access_start.elapsed().as_secs_f64() * 1_000_000.0;
                (output, access_us)
            })
            .collect();

        for (output, access_us) in remaining_outputs {
            if let Some(diag) = diagnostics.as_deref_mut() {
                diag.posting_access_time_us += access_us;
                diag.posting_decode_overhead_us +=
                    (access_us - output.stats.posting_search_time_us).max(0.0);
                diag.add_posting_stats(&output.stats);
            }
            Self::merge_topk(&mut global_heap, output.candidates, params.top_k);
        }

        global_heap
            .into_sorted_vec()
            .into_iter()
            .map(|entry| SearchResult {
                vector_id: entry.candidate.id as usize,
                distance: entry.candidate.distance,
            })
            .collect()
    }

    /// Search for k nearest neighbors using FastScan batch distance computation
    pub fn search(&self, query: &[f32], params: &SearchParams) -> Vec<SearchResult> {
        self.search_internal(query, params, None)
    }

    /// Search with diagnostics for profiling/optimization.
    pub fn search_with_diagnostics(
        &self,
        query: &[f32],
        params: &SearchParams,
    ) -> (Vec<SearchResult>, SearchDiagnostics) {
        let mut diagnostics = SearchDiagnostics::default();
        let results = self.search_internal(query, params, Some(&mut diagnostics));
        (results, diagnostics)
    }

    /// Search a posting list using FastScan batch distance computation.
    ///
    /// Returns posting-list-local top-k candidates with IVF-aligned lower-bound pruning.
    #[inline]
    fn search_posting_list_fastscan(
        &self,
        query_ctx: &crate::fastscan::QueryContext,
        plist: &PostingList,
        batch_data: &crate::fastscan::BatchData,
        top_k: usize,
        distk_seed: f32,
    ) -> PostingSearchOutput {
        use crate::math::{dot, l2_distance_sqr};
        use crate::simd;

        if top_k == 0 {
            return PostingSearchOutput::default();
        }

        let t_search_start = Instant::now();

        let query = &query_ctx.query;
        let padded_dim = plist.padded_dim;

        // Compute centroid-related terms
        let centroid_dist = l2_distance_sqr(query, &plist.centroid);
        let dot_query_centroid = dot(query, &plist.centroid);
        let g_add = match self.config.metric {
            crate::Metric::L2 => centroid_dist,
            crate::Metric::InnerProduct => -dot_query_centroid,
        };
        let g_error = centroid_dist.sqrt();

        // Select LUT view
        let use_highacc = padded_dim > 2048;
        let lut_view = if use_highacc {
            query_ctx.lut_highacc.as_ref().map(|lut| {
                crate::fastscan_kernel::FastScanLutView::HighAcc {
                    lut_low8: &lut.lut_low8,
                    lut_high8: &lut.lut_high8,
                    delta: lut.delta,
                    sum_vl_lut: lut.sum_vl_lut,
                }
            })
        } else {
            query_ctx
                .lut
                .as_ref()
                .map(|lut| crate::fastscan_kernel::FastScanLutView::Regular {
                    lut_i8: &lut.lut_i8,
                    delta: lut.delta,
                    sum_vl_lut: lut.sum_vl_lut,
                })
        };

        let Some(lut_view) = lut_view else {
            return PostingSearchOutput::default();
        };

        let num_batches = plist.num_complete_batches();
        let num_remainder = plist.num_remainder_vectors();
        let total_batches = if num_remainder > 0 {
            num_batches + 1
        } else {
            num_batches
        };

        let ex_bits = self.config.rabitq_bits.saturating_sub(1);
        let ex_ip_func = crate::fastscan_kernel::select_ex_ip_func(ex_bits);

        let mut local_heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut stats = PostingSearchStats {
            posting_lists_visited: 1,
            ..PostingSearchStats::default()
        };

        for batch_idx in 0..total_batches {
            let batch_start = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let batch_end = (batch_start + simd::FASTSCAN_BATCH_SIZE).min(plist.len());
            let actual_batch_size = batch_end - batch_start;
            stats.scanned_vectors += actual_batch_size;

            let batch_f_add = batch_data.batch_f_add(batch_idx);
            let batch_f_rescale = batch_data.batch_f_rescale(batch_idx);
            let batch_f_error = batch_data.batch_f_error(batch_idx);

            let mut ip_x0_qr_values = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut est_distances = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut lower_bounds = [0.0f32; simd::FASTSCAN_BATCH_SIZE];

            crate::fastscan_kernel::compute_fastscan_batch(
                lut_view,
                batch_data.batch_bin_codes(batch_idx),
                padded_dim,
                batch_f_add,
                batch_f_rescale,
                batch_f_error,
                g_add,
                g_error,
                query_ctx.k1x_sum_q,
                &mut ip_x0_qr_values,
                &mut est_distances,
                &mut lower_bounds,
            );

            for i in 0..actual_batch_size {
                let global_idx = batch_start + i;

                let lower_bound = crate::fastscan_kernel::sanitize_lower_bound(
                    lower_bounds[i],
                    self.config.metric,
                    dot_query_centroid,
                    query_ctx.query_norm,
                );

                let distk = Self::current_distk(&local_heap, top_k, distk_seed);
                if lower_bound >= distk {
                    stats.skipped_by_lower_bound += 1;
                    continue;
                }

                let mut distance = est_distances[i];

                if ex_bits > 0 {
                    stats.refined_vectors += 1;
                    distance = crate::fastscan_kernel::refine_distance_with_ex(
                        query,
                        &plist.ex_codes_packed[global_idx],
                        padded_dim,
                        ex_bits,
                        ip_x0_qr_values[i],
                        query_ctx.binary_scale,
                        query_ctx.kbx_sum_q,
                        g_add,
                        plist.f_add_ex[global_idx],
                        plist.f_rescale_ex[global_idx],
                        ex_ip_func,
                    );
                }

                if !distance.is_finite() {
                    continue;
                }

                if self.config.metric == crate::Metric::L2 {
                    distance = distance.max(0.0);
                }

                stats.estimated_vectors += 1;

                local_heap.push(HeapEntry {
                    candidate: HeapCandidate {
                        id: plist.ids[global_idx],
                        distance,
                    },
                });
                if local_heap.len() > top_k {
                    local_heap.pop();
                }
            }
        }

        stats.posting_search_time_us = t_search_start.elapsed().as_secs_f64() * 1_000_000.0;

        PostingSearchOutput {
            candidates: local_heap
                .into_vec()
                .into_iter()
                .map(|entry| entry.candidate)
                .collect(),
            stats,
        }
    }

    /// Batch search for multiple queries (parallel)
    ///
    /// This is much faster than calling search() in a loop because it
    /// parallelizes across queries using Rayon.
    ///
    /// # Performance
    /// Expected speedup: 4-8x on typical CPUs (depends on core count)
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        params: &SearchParams,
    ) -> Vec<Vec<SearchResult>> {
        queries.par_iter().map(|q| self.search(q, params)).collect()
    }

    /// Apply dynamic pruning to centroid candidates
    pub(crate) fn dynamic_prune(&self, candidates: &[(u32, f32)], epsilon: f32) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let closest_dist = candidates[0].1;
        let threshold = closest_dist * (1.0 + epsilon);

        candidates
            .iter()
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Search result
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub vector_id: usize,
    pub distance: f32,
}
