//! Search algorithms with dynamic pruning
//!
//! Handles FastScan batch distance computation and dynamic query pruning.

use super::*;
use rayon::prelude::*;

impl MstgIndex {
    /// Search for k nearest neighbors using FastScan batch distance computation
    pub fn search(&self, query: &[f32], params: &SearchParams) -> Vec<SearchResult> {
        use crate::fastscan::QueryContext as FastScanQueryContext;

        // Step 1: Find candidate centroids
        let centroid_candidates = self.centroid_index.search(query, params.ef_search);

        // Step 2: Dynamic pruning
        let selected_centroids = self.dynamic_prune(&centroid_candidates, params.pruning_epsilon);

        // Step 3: Create query context once
        let ex_bits = self.config.rabitq_bits.saturating_sub(1);
        let mut query_ctx = FastScanQueryContext::new(query.to_vec(), ex_bits);

        // Build LUT once for all posting lists (if dimensions match)
        if !self.directory.is_empty() {
            let first_cid = self.directory.entries[0].cluster_id;
            self.posting_lists
                .with_posting_list(first_cid, &self.directory, |first_plist| {
                    let padded_dim = first_plist.padded_dim;
                    query_ctx.build_lut(padded_dim);
                });
        }

        // Step 4: Search posting lists with FastScan batch distance computation
        let mut all_candidates: Vec<(u64, f32)> = selected_centroids
            .par_iter()
            .flat_map(|&cid| {
                self.posting_lists
                    .with_posting_list(cid, &self.directory, |plist| {
                        if plist.is_empty() {
                            return Vec::new();
                        }
                        self.search_posting_list_fastscan(&query_ctx, plist, &plist.batch_data)
                    })
                    .unwrap_or_default()
            })
            .collect();

        // Step 5: Partial sort to get top-k (faster than full sort)
        // Use select_nth_unstable to partition so smallest distances are at front
        let k = params.top_k.min(all_candidates.len());
        if k > 0 && k < all_candidates.len() {
            // Partition: first k elements are the smallest distances
            all_candidates.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Sort only the top-k elements
            all_candidates[..k].sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // k >= len: sort all (fallback)
            all_candidates.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        all_candidates.truncate(k);

        all_candidates
            .into_iter()
            .map(|(id, dist)| SearchResult {
                vector_id: id as usize,
                distance: dist,
            })
            .collect()
    }

    /// Search a posting list using FastScan batch distance computation
    #[inline]
    fn search_posting_list_fastscan(
        &self,
        query_ctx: &crate::fastscan::QueryContext,
        plist: &PostingList,
        batch_data: &crate::fastscan::BatchData,
    ) -> Vec<(u64, f32)> {
        use crate::math::{dot, l2_distance_sqr};
        use crate::simd;

        let query = &query_ctx.query;
        let padded_dim = plist.padded_dim;

        // Compute g_add (query-to-centroid distance)
        let g_add = match self.config.metric {
            crate::Metric::L2 => l2_distance_sqr(query, &plist.centroid),
            crate::Metric::InnerProduct => -dot(query, &plist.centroid),
        };

        let num_batches = plist.num_complete_batches();
        let num_remainder = plist.num_remainder_vectors();
        let total_batches = if num_remainder > 0 {
            num_batches + 1
        } else {
            num_batches
        };

        let mut results = Vec::with_capacity(plist.len());
        let use_highacc = padded_dim > 2048;

        for batch_idx in 0..total_batches {
            let batch_start = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let batch_end = (batch_start + simd::FASTSCAN_BATCH_SIZE).min(plist.len());
            let actual_batch_size = batch_end - batch_start;

            // Get batch parameters
            let batch_f_add = batch_data.batch_f_add(batch_idx);
            let batch_f_rescale = batch_data.batch_f_rescale(batch_idx);

            // Allocate output arrays
            let mut ip_x0_qr_values = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut est_distances = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut lower_bounds = [0.0f32; simd::FASTSCAN_BATCH_SIZE];

            if use_highacc {
                // High-accuracy mode using i32 accumulators
                if let Some(ref lut_ha) = query_ctx.lut_highacc {
                    let mut accu_res_i32 = [0i32; simd::FASTSCAN_BATCH_SIZE];
                    simd::accumulate_batch_highacc_avx2(
                        batch_data.batch_bin_codes(batch_idx),
                        &lut_ha.lut_low8,
                        &lut_ha.lut_high8,
                        padded_dim,
                        &mut accu_res_i32,
                    );

                    simd::compute_batch_distances_i32(
                        &accu_res_i32,
                        lut_ha.delta,
                        lut_ha.sum_vl_lut,
                        batch_f_add,
                        batch_f_rescale,
                        &[0.0f32; simd::FASTSCAN_BATCH_SIZE],
                        g_add,
                        0.0,
                        query_ctx.k1x_sum_q,
                        &mut ip_x0_qr_values,
                        &mut est_distances,
                        &mut lower_bounds,
                    );
                }
            } else if let Some(ref lut) = query_ctx.lut {
                // Standard mode using i16 accumulators
                let mut accu_res = [0u16; simd::FASTSCAN_BATCH_SIZE];
                simd::accumulate_batch_avx2(
                    batch_data.batch_bin_codes(batch_idx),
                    &lut.lut_i8,
                    padded_dim,
                    &mut accu_res,
                );

                simd::compute_batch_distances_u16(
                    &accu_res,
                    lut.delta,
                    lut.sum_vl_lut,
                    batch_f_add,
                    batch_f_rescale,
                    &[0.0f32; simd::FASTSCAN_BATCH_SIZE],
                    g_add,
                    0.0,
                    query_ctx.k1x_sum_q,
                    &mut ip_x0_qr_values,
                    &mut est_distances,
                    &mut lower_bounds,
                );
            }

            // Collect results for this batch
            for (i, &distance) in est_distances.iter().enumerate().take(actual_batch_size) {
                let global_idx = batch_start + i;
                let vector_id = plist.ids[global_idx];

                if distance.is_finite() {
                    // For L2 metric, clamp negative distances to 0 (due to quantization approximation errors)
                    // For InnerProduct, negative values are valid (higher similarity)
                    let clamped_distance = if self.config.metric == crate::Metric::L2 {
                        distance.max(0.0)
                    } else {
                        distance
                    };
                    results.push((vector_id, clamped_distance));
                }
            }
        }

        results
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
