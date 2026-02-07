//! Centroid index for fast nearest centroid search using HNSW
//!
//! Uses hnsw_rs for efficient approximate nearest neighbor search

use crate::mstg::config::ScalarPrecision;
use crate::mstg::scalar_quant::{BF16Vector, FP32Vector, QuantizedVector as ScalarQuantizedVector};
use crate::Metric;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Centroid index for fast nearest centroid queries using HNSW
pub struct CentroidIndex {
    #[allow(dead_code)]
    precision: ScalarPrecision,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    metric: Metric,
    pub(crate) centroid_ids: Vec<u32>,
    centroids: CentroidData,
    /// Store the centroids in a Box for stable addresses
    centroid_vecs: Arc<[Vec<f32>]>,
    /// Vectors used for HNSW (normalized if metric is InnerProduct)
    hnsw_vecs: Arc<[Vec<f32>]>,
    /// Cached HNSW index (built lazily on first search)
    /// Safety: The HNSW borrows from centroid_vecs, which is never moved after construction
    pub(crate) hnsw_cache: RwLock<Option<HnswCache>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CentroidData {
    FP32(Vec<FP32Vector>),
    BF16(Vec<BF16Vector>),
}

pub(crate) enum HnswCache {
    L2(Hnsw<'static, f32, DistL2>),
    Dot(Hnsw<'static, f32, DistDot>),
}

impl CentroidIndex {
    /// Build a centroid index from full-precision centroids using HNSW
    pub fn build(
        centroids: Vec<Vec<f32>>,
        centroid_ids: Vec<u32>,
        precision: ScalarPrecision,
        hnsw_m: usize,
        hnsw_ef_construction: usize,
        metric: Metric,
    ) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());

        // Quantize centroids for memory efficiency
        let centroids_data = match precision {
            ScalarPrecision::FP32 => {
                let quant: Vec<FP32Vector> =
                    centroids.iter().map(|c| FP32Vector::quantize(c)).collect();
                CentroidData::FP32(quant)
            }
            ScalarPrecision::BF16 => {
                let quant: Vec<BF16Vector> =
                    centroids.iter().map(|c| BF16Vector::quantize(c)).collect();
                CentroidData::BF16(quant)
            }
            _ => panic!("Unsupported precision: {:?}", precision),
        };

        // Store centroids in an Arc for stable addresses and cheap sharing
        let centroid_vecs: Arc<[Vec<f32>]> = Arc::from(centroids.into_boxed_slice());
        let hnsw_vecs: Arc<[Vec<f32>]> = if metric == Metric::InnerProduct {
            let normalized: Vec<Vec<f32>> = centroid_vecs.iter().map(|v| normalize(v)).collect();
            Arc::from(normalized.into_boxed_slice())
        } else {
            Arc::clone(&centroid_vecs)
        };

        Self {
            precision,
            hnsw_m,
            hnsw_ef_construction,
            metric,
            centroid_ids,
            centroids: centroids_data,
            centroid_vecs,
            hnsw_vecs,
            hnsw_cache: RwLock::new(None),
        }
    }

    /// Build and cache the HNSW index
    pub(crate) fn ensure_hnsw_built(&self) {
        // Check if already built (fast path with read lock)
        {
            let cache = self.hnsw_cache.read();
            if cache.is_some() {
                return;
            }
        }

        // Build HNSW (slow path with write lock)
        let mut cache = self.hnsw_cache.write();
        if cache.is_some() {
            return; // Another thread built it while we were waiting
        }

        let nb_elements = self.hnsw_vecs.len();

        // Parameters for HNSW construction
        let max_nb_connection = self.hnsw_m.max(2);
        let ef_construction = self.hnsw_ef_construction.max(2);
        // Use fixed max_layer to avoid hnsw_rs serialization issues
        // The library expects max_layer to match NB_MAX_LAYER during serialization
        let max_layer = 16;

        let mut hnsw = match self.metric {
            Metric::L2 => HnswCache::L2(Hnsw::<f32, DistL2>::new(
                max_nb_connection,
                nb_elements,
                max_layer,
                ef_construction,
                DistL2 {},
            )),
            Metric::InnerProduct => HnswCache::Dot(Hnsw::<f32, DistDot>::new(
                max_nb_connection,
                nb_elements,
                max_layer,
                ef_construction,
                DistDot {},
            )),
        };

        // SAFETY: We're extending the lifetime from 'a to 'static here.
        // This is safe because:
        // 1. hnsw_vecs is owned by this struct and stored in an Arc (stable address)
        // 2. hnsw_vecs will never be moved or dropped while hnsw_cache exists
        // 3. hnsw_cache is only accessible through &self, ensuring proper borrowing
        // 4. The HNSW index will be dropped before hnsw_vecs when the struct is dropped
        let data_with_id: Vec<(&Vec<f32>, usize)> = unsafe {
            std::mem::transmute::<Vec<(&Vec<f32>, usize)>, Vec<(&Vec<f32>, usize)>>(
                self.hnsw_vecs
                    .iter()
                    .zip(0..nb_elements)
                    .collect::<Vec<_>>(),
            )
        };

        match &mut hnsw {
            HnswCache::L2(inner) => {
                inner.parallel_insert(&data_with_id);
                inner.set_searching_mode(true);
            }
            HnswCache::Dot(inner) => {
                inner.parallel_insert(&data_with_id);
                inner.set_searching_mode(true);
            }
        }

        *cache = Some(hnsw);
    }

    /// Search for k nearest centroids to the query using HNSW
    ///
    /// `k` parameter is the ef_search value from SearchParams
    ///
    /// Returns (centroid_id, distance) pairs sorted by distance
    pub fn search(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        let n = self.hnsw_vecs.len();

        // For very small number of centroids, use brute force
        // HNSW doesn't work well with < 2 centroids
        if n < 2 {
            return self.brute_force_search(query, ef_search);
        }

        // Ensure HNSW is built
        self.ensure_hnsw_built();

        // Search using cached HNSW
        let cache = self.hnsw_cache.read();
        let hnsw = cache.as_ref().unwrap();

        // Use ef_search directly as the HNSW ef parameter
        // Return ef_search results (will be pruned later by dynamic pruning)
        let neighbors = match (self.metric, hnsw) {
            (Metric::L2, HnswCache::L2(inner)) => {
                inner.search(query, ef_search.min(n), ef_search.min(n))
            }
            (Metric::InnerProduct, HnswCache::Dot(inner)) => {
                let normalized_query = normalize(query);
                inner.search(&normalized_query, ef_search.min(n), ef_search.min(n))
            }
            _ => Vec::new(),
        };

        // Convert results to (centroid_id, distance) pairs
        neighbors
            .iter()
            .map(|neighbor| {
                let idx = neighbor.d_id;
                let centroid_id = self.centroid_ids[idx];
                let distance = neighbor.distance;
                (centroid_id, distance)
            })
            .collect()
    }

    /// Brute force search for small number of centroids
    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = match self.metric {
            Metric::L2 => self
                .centroid_vecs
                .iter()
                .enumerate()
                .map(|(idx, centroid)| {
                    let dist = Self::l2_distance_sqr(query, centroid);
                    (self.centroid_ids[idx], dist)
                })
                .collect(),
            Metric::InnerProduct => {
                let normalized_query = normalize(query);
                self.hnsw_vecs
                    .iter()
                    .enumerate()
                    .map(|(idx, centroid)| {
                        let dist = 1.0 - dot(&normalized_query, centroid);
                        (self.centroid_ids[idx], dist)
                    })
                    .collect()
            }
        };

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Compute squared L2 distance between two vectors
    fn l2_distance_sqr(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
    }

    /// Get the number of centroids
    pub fn len(&self) -> usize {
        self.centroid_ids.len()
    }

    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.centroid_ids.is_empty()
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let vec_size = match &self.centroids {
            CentroidData::FP32(vecs) => vecs.iter().map(|v| v.memory_size()).sum::<usize>(),
            CentroidData::BF16(vecs) => vecs.iter().map(|v| v.memory_size()).sum::<usize>(),
        };

        let centroid_vecs_size: usize = self
            .centroid_vecs
            .iter()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum();
        let hnsw_vecs_size: usize = if Arc::ptr_eq(&self.centroid_vecs, &self.hnsw_vecs) {
            0
        } else {
            self.hnsw_vecs
                .iter()
                .map(|v| v.len() * std::mem::size_of::<f32>())
                .sum()
        };

        vec_size
            + self.centroid_ids.len() * std::mem::size_of::<u32>()
            + centroid_vecs_size
            + hnsw_vecs_size
    }
}

fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return vec![0.0; vector.len()];
    }
    vector.iter().map(|v| v / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid_index_build() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let ids = vec![0, 1, 2, 3];

        let index =
            CentroidIndex::build(centroids, ids, ScalarPrecision::BF16, 32, 200, Metric::L2);

        assert_eq!(index.len(), 4);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_centroid_search() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
        ];
        let ids = vec![0, 1, 2, 3];

        let index =
            CentroidIndex::build(centroids, ids, ScalarPrecision::FP32, 32, 200, Metric::L2);

        // Query close to centroid 0
        let query = vec![0.1, 0.1];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest should be centroid 0
    }

    #[test]
    fn test_bf16_memory_savings() {
        let centroids: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 960]).collect();
        let ids: Vec<u32> = (0..100).collect();

        let index_fp32 = CentroidIndex::build(
            centroids.clone(),
            ids.clone(),
            ScalarPrecision::FP32,
            32,
            200,
            Metric::L2,
        );
        let index_bf16 =
            CentroidIndex::build(centroids, ids, ScalarPrecision::BF16, 32, 200, Metric::L2);

        let mem_fp32 = index_fp32.memory_usage();
        let mem_bf16 = index_bf16.memory_usage();

        println!("FP32 memory: {} bytes", mem_fp32);
        println!("BF16 memory: {} bytes", mem_bf16);

        // BF16 should use less memory (not 50% due to centroid_vecs overhead)
        assert!(mem_bf16 < mem_fp32);
    }

    #[test]
    fn test_hnsw_caching() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
        ];
        let ids = vec![0, 1, 2, 3];

        let index =
            CentroidIndex::build(centroids, ids, ScalarPrecision::FP32, 32, 200, Metric::L2);

        // First search should build the HNSW
        let query = vec![0.1, 0.1];
        let results1 = index.search(&query, 2);

        // Second search should use cached HNSW
        let results2 = index.search(&query, 2);

        // Results should be identical
        assert_eq!(results1, results2);
        assert_eq!(results1[0].0, 0);
    }

    #[test]
    fn test_centroid_search_inner_product() {
        let centroids = vec![vec![2.0, 0.0], vec![0.0, 3.0], vec![1.0, 1.0]];
        let ids = vec![0, 1, 2];

        let index = CentroidIndex::build(
            centroids,
            ids,
            ScalarPrecision::FP32,
            32,
            200,
            Metric::InnerProduct,
        );

        let query = vec![10.0, 0.0];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }
}
