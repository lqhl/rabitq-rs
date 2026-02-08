//! Main MSTG index structure

use super::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Main MSTG (Multi-Scale Tree Graph) index
pub struct MstgIndex {
    pub config: MstgConfig,
    pub centroid_index: CentroidIndex,
    pub posting_lists: Vec<PostingList>,
    pub directory: PostingListDirectory,
}

impl MstgIndex {
    /// Build an MSTG index from data
    pub fn build(data: &[Vec<f32>], config: MstgConfig) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot build index from empty data".to_string());
        }

        println!("Building MSTG index for {} vectors", data.len());

        // Step 1: Hierarchical balanced clustering
        println!("Step 1: Hierarchical balanced clustering...");
        let clustering = HierarchicalClustering::new(
            config.max_posting_size,
            config.branching_factor,
            config.balance_weight,
        );
        let clusters = clustering.cluster(data);
        println!("  Created {} clusters", clusters.len());

        // Step 2: Closure assignment
        println!("Step 2: Closure assignment with RNG rule...");
        let centroids: Vec<Vec<f32>> = clusters.iter().map(|c| c.centroid().to_vec()).collect();

        let assigner = ClosureAssigner::new(config.closure_epsilon, config.max_replicas);

        let cluster_assignments: Vec<Vec<usize>> = data
            .par_iter()
            .map(|v| assigner.assign(v, &centroids))
            .collect();

        // Build reverse mapping: cluster_id -> vector indices
        let mut cluster_to_vec_ids: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
        for (vec_id, assignments) in cluster_assignments.iter().enumerate() {
            for &cluster_id in assignments {
                if let Some(bucket) = cluster_to_vec_ids.get_mut(cluster_id) {
                    bucket.push(vec_id);
                }
            }
        }

        // Count total assignments for statistics
        let total_assignments: usize = cluster_assignments.iter().map(|a| a.len()).sum();
        let replication_factor = total_assignments as f32 / data.len() as f32;
        println!("  Average replication factor: {:.2}", replication_factor);

        // Step 3: Create posting lists with RaBitQ quantization
        println!("Step 3: Creating and quantizing posting lists...");
        let posting_lists: Vec<PostingList> = clusters
            .par_iter()
            .enumerate()
            .map(|(cluster_id, cluster)| {
                let mut plist = PostingList::new(cluster_id as u32, cluster.centroid().to_vec());

                // Collect vectors assigned to this cluster
                if let Some(vec_ids) = cluster_to_vec_ids.get(cluster_id) {
                    if !vec_ids.is_empty() {
                        plist
                            .quantize_vectors_by_ids(
                                data,
                                vec_ids,
                                config.rabitq_bits,
                                config.metric,
                                config.faster_config,
                            )
                            .unwrap_or_else(|e| {
                                eprintln!(
                                    "Warning: Quantization failed for cluster {}: {}",
                                    cluster_id, e
                                );
                            });
                    }
                }

                plist
            })
            .collect();

        println!(
            "  Created {} posting lists with {} total vectors",
            posting_lists.len(),
            posting_lists.iter().map(|p| p.len()).sum::<usize>()
        );

        // Build FastScan batch layout for each posting list
        println!("Step 3.5: Building FastScan batch layouts...");
        let posting_lists: Vec<PostingList> = posting_lists
            .into_par_iter()
            .map(|mut plist| {
                plist.build_batch_layout();
                plist
            })
            .collect();

        // Step 4: Build centroid index
        println!("Step 4: Building centroid index...");
        let centroid_reps: Vec<Vec<f32>> =
            posting_lists.iter().map(|p| p.centroid.clone()).collect();
        let centroid_ids: Vec<u32> = (0..posting_lists.len() as u32).collect();

        let centroid_index = CentroidIndex::build(
            centroid_reps,
            centroid_ids,
            config.centroid_precision,
            config.hnsw_m,
            config.hnsw_ef_construction,
            config.metric,
        );

        println!(
            "  Built centroid index with {} centroids ({} precision)",
            centroid_index.len(),
            match config.centroid_precision {
                ScalarPrecision::FP32 => "FP32",
                ScalarPrecision::BF16 => "BF16",
                _ => "Other",
            }
        );

        // Step 5: Create metadata directory
        let directory = PostingListDirectory::new();

        println!("MSTG index build complete");
        println!(
            "  Memory usage estimate: ~{:.2} MB",
            Self::estimate_memory_mb(&centroid_index, &posting_lists)
        );

        Ok(Self {
            config,
            centroid_index,
            posting_lists,
            directory,
        })
    }

    /// Estimate total memory usage in MB
    fn estimate_memory_mb(centroid_index: &CentroidIndex, posting_lists: &[PostingList]) -> f32 {
        let centroid_mem = centroid_index.memory_usage();
        let posting_mem: usize = posting_lists.iter().map(|p| p.memory_size()).sum();
        ((centroid_mem + posting_mem) as f32) / (1024.0 * 1024.0)
    }

    /// Search for k nearest neighbors using FastScan batch distance computation
    pub fn search(&self, query: &[f32], params: &SearchParams) -> Vec<SearchResult> {
        use crate::fastscan::QueryContext as FastScanQueryContext;

        if params.top_k == 0 {
            return Vec::new();
        }

        // Step 1: Find candidate centroids
        let centroid_candidates = self.centroid_index.search(query, params.ef_search);

        // Step 2: Dynamic pruning
        let selected_centroids = self.dynamic_prune(&centroid_candidates, params.pruning_epsilon);

        // Step 3: Create query context once
        let ex_bits = self.config.rabitq_bits.saturating_sub(1);
        let mut query_ctx = FastScanQueryContext::new(query.to_vec(), ex_bits);

        // Build LUT once for all posting lists (if dimensions match)
        if !self.posting_lists.is_empty() {
            let padded_dim = self.posting_lists[0].padded_dim;
            query_ctx.build_lut(padded_dim);
        }

        // Step 4: Search posting lists with FastScan batch distance computation
        let mut topk = TopK::new(params.top_k);
        for &cid in &selected_centroids {
            let plist = &self.posting_lists[cid as usize];

            // Skip empty posting lists
            if plist.vectors.is_empty() {
                continue;
            }

            // Use FastScan batch distance computation with pruning
            self.search_posting_list_fastscan(&query_ctx, plist, &plist.batch_data, &mut topk);
        }

        topk.into_sorted_results()
    }

    /// Search a posting list using FastScan batch distance computation
    #[inline]
    fn search_posting_list_fastscan(
        &self,
        query_ctx: &crate::fastscan::QueryContext,
        plist: &PostingList,
        batch_data: &crate::fastscan::BatchData,
        topk: &mut TopK,
    ) {
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

        let use_highacc = padded_dim > 2048;

        for batch_idx in 0..total_batches {
            let batch_start = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let batch_end = (batch_start + simd::FASTSCAN_BATCH_SIZE).min(plist.vectors.len());
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

            let threshold = topk.threshold();
            if let Some(th) = threshold {
                let min_lb = lower_bounds
                    .iter()
                    .take(actual_batch_size)
                    .fold(f32::INFINITY, |acc, &v| acc.min(v));
                if min_lb > th {
                    continue;
                }
            }

            // Collect results for this batch
            for (i, &distance) in est_distances.iter().enumerate().take(actual_batch_size) {
                let global_idx = batch_start + i;
                let vector_id = plist.vectors[global_idx].vector_id;

                if distance.is_finite() {
                    if let Some(th) = topk.threshold() {
                        if lower_bounds[i] > th {
                            continue;
                        }
                    }
                    // For L2 metric, clamp negative distances to 0 (due to quantization approximation errors)
                    // For InnerProduct, negative values are valid (higher similarity)
                    let clamped_distance = if self.config.metric == crate::Metric::L2 {
                        distance.max(0.0)
                    } else {
                        distance
                    };
                    topk.push(vector_id, clamped_distance);
                }
            }
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
    fn dynamic_prune(&self, candidates: &[(u32, f32)], epsilon: f32) -> Vec<u32> {
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

#[derive(Debug)]
struct HeapItem {
    distance: f32,
    vector_id: u64,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

struct TopK {
    k: usize,
    heap: BinaryHeap<HeapItem>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::new(),
        }
    }

    fn threshold(&self) -> Option<f32> {
        if self.heap.len() < self.k {
            None
        } else {
            self.heap.peek().map(|item| item.distance)
        }
    }

    fn push(&mut self, vector_id: u64, distance: f32) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(HeapItem {
                distance,
                vector_id,
            });
            return;
        }

        if let Some(top) = self.heap.peek() {
            if distance < top.distance {
                self.heap.pop();
                self.heap.push(HeapItem {
                    distance,
                    vector_id,
                });
            }
        }
    }

    fn into_sorted_results(self) -> Vec<SearchResult> {
        let mut sorted = self.heap.into_sorted_vec();
        sorted
            .drain(..)
            .map(|item| SearchResult {
                vector_id: item.vector_id as usize,
                distance: item.distance,
            })
            .collect()
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub vector_id: usize,
    pub distance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(42);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen()).collect())
            .collect()
    }

    #[test]
    fn test_mstg_index_build_small() {
        let data = generate_test_data(100, 16);
        let config = MstgConfig {
            max_posting_size: 20,
            branching_factor: 4,
            ..Default::default()
        };

        let index = MstgIndex::build(&data, config).unwrap();

        assert!(!index.posting_lists.is_empty());
        assert!(!index.centroid_index.is_empty());
    }

    #[test]
    fn test_mstg_search() {
        let data = generate_test_data(100, 16);
        let config = MstgConfig {
            max_posting_size: 25,
            branching_factor: 4,
            ..Default::default()
        };

        let index = MstgIndex::build(&data, config).unwrap();

        let query = &data[0];
        let params = SearchParams::balanced(10);
        let results = index.search(query, &params);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // First result should be the query itself (or very close)
        assert!(results[0].distance < 0.1);

        // Ensure results are sorted by distance (ascending)
        for pair in results.windows(2) {
            assert!(pair[0].distance <= pair[1].distance);
        }
    }
}
