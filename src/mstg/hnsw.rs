//! Centroid index for fast nearest centroid search using HNSW
//!
//! Uses instant-distance for efficient approximate nearest neighbor search

use crate::mstg::config::ScalarPrecision;
use crate::mstg::scalar_quant::{BF16Vector, FP32Vector, QuantizedVector as ScalarQuantizedVector};
use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};

/// Wrapper for f32 vectors to implement Point trait
#[derive(Clone, Debug)]
struct CentroidPoint(Vec<f32>);

impl Point for CentroidPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Compute L2 distance (not squared - instant-distance expects regular distance)
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// Centroid index for fast nearest centroid queries using HNSW
pub struct CentroidIndex {
    precision: ScalarPrecision,
    centroid_ids: Vec<u32>,
    centroids: CentroidData,
    hnsw: HnswMap<CentroidPoint, usize>,
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

        // Build HNSW index with full-precision centroids for accurate search
        let points: Vec<CentroidPoint> = centroids
            .into_iter()
            .map(|vec| CentroidPoint(vec))
            .collect();

        let indices: Vec<usize> = (0..points.len()).collect();

        // Build with high ef_search for good recall
        // Note: instant-distance sets ef_search at build time, cannot vary per query
        let hnsw = Builder::default()
            .ef_search(1000)  // High beam width for quality centroid selection
            .build(points, indices);

        Self {
            precision,
            centroid_ids,
            centroids: centroids_data,
            hnsw,
        }
    }

    /// Search for k nearest centroids to the query using HNSW
    ///
    /// Returns (centroid_id, distance) pairs sorted by distance
    ///
    /// Note: ef_search is set at build time (currently 1000) and cannot be varied per query
    /// due to instant-distance library design. The k parameter just limits results returned.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let query_point = CentroidPoint(query.to_vec());
        let mut search = Search::default();

        // Search using HNSW (uses ef_search=1000 set during build)
        let neighbors = self.hnsw.search(&query_point, &mut search);

        // Convert results to (centroid_id, distance) pairs
        neighbors
            .take(k)
            .map(|item| {
                let idx = *item.value; // Internal index
                let centroid_id = self.centroid_ids[idx];
                let distance = item.distance * item.distance; // Square to match L2 distance
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

        vec_size + self.centroid_ids.len() * std::mem::size_of::<u32>()
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

        // BF16 should use ~50% less memory
        assert!((mem_bf16 as f64) < (mem_fp32 as f64 * 0.6));
    }
}
