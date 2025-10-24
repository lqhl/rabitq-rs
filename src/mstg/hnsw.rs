//! Centroid index for fast nearest centroid search using HNSW
//!
//! Uses hnsw_rs for efficient approximate nearest neighbor search

use crate::mstg::config::ScalarPrecision;
use crate::mstg::scalar_quant::{BF16Vector, FP32Vector, QuantizedVector as ScalarQuantizedVector};
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Centroid index for fast nearest centroid queries using HNSW
pub struct CentroidIndex {
    #[allow(dead_code)]
    precision: ScalarPrecision,
    centroid_ids: Vec<u32>,
    centroids: CentroidData,
    /// Store the centroids in a Box for stable addresses
    centroid_vecs: Box<[Vec<f32>]>,
    /// Cached HNSW index (built lazily on first search)
    /// Safety: The HNSW borrows from centroid_vecs, which is never moved after construction
    hnsw_cache: RwLock<Option<Hnsw<'static, f32, DistL2>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum CentroidData {
    FP32(Vec<FP32Vector>),
    BF16(Vec<BF16Vector>),
}

impl CentroidIndex {
    /// Build a centroid index from full-precision centroids using HNSW
    pub fn build(
        centroids: Vec<Vec<f32>>,
        centroid_ids: Vec<u32>,
        precision: ScalarPrecision,
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

        // Store centroids in a Box for stable addresses
        let centroid_vecs: Box<[Vec<f32>]> = centroids.into_boxed_slice();

        Self {
            precision,
            centroid_ids,
            centroids: centroids_data,
            centroid_vecs,
            hnsw_cache: RwLock::new(None),
        }
    }

    /// Build and cache the HNSW index
    fn ensure_hnsw_built(&self) {
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

        let nb_elements = self.centroid_vecs.len();

        // Parameters for HNSW construction
        let max_nb_connection = 32;
        let ef_construction = 200;
        let max_layer = 16.min((nb_elements as f32).ln().floor() as usize);

        let mut hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            nb_elements,
            max_layer,
            ef_construction,
            DistL2 {},
        );

        // SAFETY: We're extending the lifetime from 'a to 'static here.
        // This is safe because:
        // 1. centroid_vecs is owned by this struct and stored in a Box (stable address)
        // 2. centroid_vecs will never be moved or dropped while hnsw_cache exists
        // 3. hnsw_cache is only accessible through &self, ensuring proper borrowing
        // 4. The HNSW index will be dropped before centroid_vecs when the struct is dropped
        let data_with_id: Vec<(&Vec<f32>, usize)> = unsafe {
            std::mem::transmute::<Vec<(&Vec<f32>, usize)>, Vec<(&Vec<f32>, usize)>>(
                self.centroid_vecs
                    .iter()
                    .zip(0..nb_elements)
                    .collect::<Vec<_>>(),
            )
        };

        hnsw.parallel_insert(&data_with_id);
        hnsw.set_searching_mode(true);

        *cache = Some(hnsw);
    }

    /// Search for k nearest centroids to the query using HNSW
    ///
    /// `k` parameter is the ef_search value from SearchParams
    ///
    /// Returns (centroid_id, distance) pairs sorted by distance
    pub fn search(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        // Ensure HNSW is built
        self.ensure_hnsw_built();

        // Search using cached HNSW
        let cache = self.hnsw_cache.read();
        let hnsw = cache.as_ref().unwrap();

        // Use ef_search directly as the HNSW ef parameter
        // Return ef_search results (will be pruned later by dynamic pruning)
        let neighbors = hnsw.search(query, ef_search, ef_search);

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

    /// Get the number of centroids
    pub fn len(&self) -> usize {
        self.centroid_ids.len()
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

        vec_size + self.centroid_ids.len() * std::mem::size_of::<u32>() + centroid_vecs_size
    }
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

        let index = CentroidIndex::build(centroids, ids, ScalarPrecision::BF16);

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

        let index = CentroidIndex::build(centroids, ids, ScalarPrecision::FP32);

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

        let index_fp32 =
            CentroidIndex::build(centroids.clone(), ids.clone(), ScalarPrecision::FP32);
        let index_bf16 = CentroidIndex::build(centroids, ids, ScalarPrecision::BF16);

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

        let index = CentroidIndex::build(centroids, ids, ScalarPrecision::FP32);

        // First search should build the HNSW
        let query = vec![0.1, 0.1];
        let results1 = index.search(&query, 2);

        // Second search should use cached HNSW
        let results2 = index.search(&query, 2);

        // Results should be identical
        assert_eq!(results1, results2);
        assert_eq!(results1[0].0, 0);
    }
}
