//! Centroid index for fast nearest centroid search using HNSW
//!
//! Uses hnsw_rs for efficient approximate nearest neighbor search
//! Includes true scalar quantization implementations to save memory

use crate::mstg::config::ScalarPrecision;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Distance function for BF16 quantized vectors
#[derive(Clone)]
pub struct DistBF16;

impl Distance<u16> for DistBF16 {
    fn eval(&self, va: &[u16], vb: &[u16]) -> f32 {
        let mut sum = 0.0f32;
        for (a, b) in va.iter().zip(vb) {
            let diff = crate::mstg::scalar_quant::bf16_to_fp32(*a)
                - crate::mstg::scalar_quant::bf16_to_fp32(*b);
            sum += diff * diff;
        }
        sum
    }
}

/// Distance function for FP16 quantized vectors
#[derive(Clone)]
pub struct DistFP16;

impl Distance<half::f16> for DistFP16 {
    fn eval(&self, va: &[half::f16], vb: &[half::f16]) -> f32 {
        let mut sum = 0.0f32;
        for (a, b) in va.iter().zip(vb) {
            let diff = a.to_f32() - b.to_f32();
            sum += diff * diff;
        }
        sum
    }
}

/// Distance function for INT8 quantized vectors
#[derive(Clone, Serialize, Deserialize)]
pub struct DistINT8 {
    pub scale: f32,
}

impl Distance<i8> for DistINT8 {
    fn eval(&self, va: &[i8], vb: &[i8]) -> f32 {
        let mut sum = 0.0f32;
        for (a, b) in va.iter().zip(vb) {
            let diff = (*a as i32 - *b as i32) as f32;
            sum += diff * diff;
        }
        sum * self.scale * self.scale
    }
}

/// Centroid index for fast nearest centroid queries using HNSW
pub struct CentroidIndex {
    #[allow(dead_code)]
    precision: ScalarPrecision,
    pub(crate) centroid_ids: Vec<u32>,
    /// Store the centroid data securely so it won't move, acting as backing memory for HNSW
    pub(crate) centroids: CentroidData,
    /// Cached HNSW index structure
    pub(crate) hnsw_cache: RwLock<Option<HnswIndex>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum CentroidData {
    FP32(Box<[Vec<f32>]>),
    BF16(Box<[Vec<u16>]>),
    FP16(Box<[Vec<half::f16>]>),
    INT8 {
        data: Box<[Vec<i8>]>,
        scale: f32,
        offset: f32,
    },
}

pub(crate) enum HnswIndex {
    FP32(Hnsw<'static, f32, DistL2>),
    BF16(Hnsw<'static, u16, DistBF16>),
    FP16(Hnsw<'static, half::f16, DistFP16>),
    INT8(Hnsw<'static, i8, DistINT8>),
}

impl CentroidData {
    pub fn memory_size(&self) -> usize {
        match self {
            Self::FP32(data) => data.iter().map(|v| v.len() * 4).sum(),
            Self::BF16(data) => data.iter().map(|v| v.len() * 2).sum(),
            Self::FP16(data) => data.iter().map(|v| v.len() * 2).sum(),
            Self::INT8 { data, .. } => data.iter().map(|v| v.len()).sum(),
        }
    }

    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::FP32(data) => data.first().map(|v| v.len()),
            Self::BF16(data) => data.first().map(|v| v.len()),
            Self::FP16(data) => data.first().map(|v| v.len()),
            Self::INT8 { data, .. } => data.first().map(|v| v.len()),
        }
    }
}

impl CentroidIndex {
    /// Build a centroid index from full-precision centroids using HNSW
    pub fn build(
        centroids: Vec<Vec<f32>>,
        centroid_ids: Vec<u32>,
        precision: ScalarPrecision,
    ) -> Self {
        assert_eq!(centroids.len(), centroid_ids.len());

        let centroids_data = match precision {
            ScalarPrecision::FP32 => CentroidData::FP32(centroids.into_boxed_slice()),
            ScalarPrecision::BF16 => {
                let quant: Vec<Vec<u16>> = centroids
                    .iter()
                    .map(|c| {
                        c.iter()
                            .map(|&x| crate::mstg::scalar_quant::fp32_to_bf16(x))
                            .collect()
                    })
                    .collect();
                CentroidData::BF16(quant.into_boxed_slice())
            }
            ScalarPrecision::FP16 => {
                let quant: Vec<Vec<half::f16>> = centroids
                    .iter()
                    .map(|c| c.iter().map(|&x| half::f16::from_f32(x)).collect())
                    .collect();
                CentroidData::FP16(quant.into_boxed_slice())
            }
            ScalarPrecision::INT8 => {
                let mut max_val = 0.0f32;
                let mut min_val = 0.0f32;

                for c in &centroids {
                    for &v in c {
                        if v > max_val {
                            max_val = v;
                        }
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }

                let range = max_val - min_val;
                let scale = if range == 0.0 { 1.0 } else { range / 255.0 };
                let offset = min_val + (range / 2.0);

                let quant: Vec<Vec<i8>> = centroids
                    .iter()
                    .map(|c| {
                        c.iter()
                            .map(|&x| {
                                let q = (x - offset) / scale;
                                q.clamp(-128.0, 127.0).round() as i8
                            })
                            .collect()
                    })
                    .collect();

                CentroidData::INT8 {
                    data: quant.into_boxed_slice(),
                    scale,
                    offset,
                }
            }
        };

        Self {
            precision,
            centroid_ids,
            centroids: centroids_data,
            hnsw_cache: RwLock::new(None),
        }
    }

    /// Build and cache the HNSW index properly pinned into life times securely
    pub(crate) fn ensure_hnsw_built(&self) {
        {
            let cache = self.hnsw_cache.read();
            if cache.is_some() {
                return;
            }
        }

        let mut cache = self.hnsw_cache.write();
        if cache.is_some() {
            return;
        }

        let max_nb_connection = 32;
        let ef_construction = 200;
        let max_layer = 16;

        let hnsw_index = match &self.centroids {
            CentroidData::FP32(data) => {
                let mut hnsw = Hnsw::<f32, DistL2>::new(
                    max_nb_connection,
                    data.len(),
                    max_layer,
                    ef_construction,
                    DistL2 {},
                );
                let data_with_id: Vec<(&Vec<f32>, usize)> = unsafe {
                    std::mem::transmute(
                        data.iter()
                            .enumerate()
                            .map(|(i, v)| (v, i))
                            .collect::<Vec<_>>(),
                    )
                };
                hnsw.parallel_insert(&data_with_id);
                hnsw.set_searching_mode(true);
                HnswIndex::FP32(hnsw)
            }
            CentroidData::BF16(data) => {
                let mut hnsw = Hnsw::<u16, DistBF16>::new(
                    max_nb_connection,
                    data.len(),
                    max_layer,
                    ef_construction,
                    DistBF16 {},
                );
                let data_with_id: Vec<(&Vec<u16>, usize)> = unsafe {
                    std::mem::transmute(
                        data.iter()
                            .enumerate()
                            .map(|(i, v)| (v, i))
                            .collect::<Vec<_>>(),
                    )
                };
                hnsw.parallel_insert(&data_with_id);
                hnsw.set_searching_mode(true);
                HnswIndex::BF16(hnsw)
            }
            CentroidData::FP16(data) => {
                let mut hnsw = Hnsw::<half::f16, DistFP16>::new(
                    max_nb_connection,
                    data.len(),
                    max_layer,
                    ef_construction,
                    DistFP16 {},
                );
                let data_with_id: Vec<(&Vec<half::f16>, usize)> = unsafe {
                    std::mem::transmute(
                        data.iter()
                            .enumerate()
                            .map(|(i, v)| (v, i))
                            .collect::<Vec<_>>(),
                    )
                };
                hnsw.parallel_insert(&data_with_id);
                hnsw.set_searching_mode(true);
                HnswIndex::FP16(hnsw)
            }
            CentroidData::INT8 { data, scale, .. } => {
                let mut hnsw = Hnsw::<i8, DistINT8>::new(
                    max_nb_connection,
                    data.len(),
                    max_layer,
                    ef_construction,
                    DistINT8 { scale: *scale },
                );
                let data_with_id: Vec<(&Vec<i8>, usize)> = unsafe {
                    std::mem::transmute(
                        data.iter()
                            .enumerate()
                            .map(|(i, v)| (v, i))
                            .collect::<Vec<_>>(),
                    )
                };
                hnsw.parallel_insert(&data_with_id);
                hnsw.set_searching_mode(true);
                HnswIndex::INT8(hnsw)
            }
        };

        *cache = Some(hnsw_index);
    }

    pub fn search(&self, query: &[f32], ef_search: usize) -> Vec<(u32, f32)> {
        let n = self.centroid_ids.len();

        if n < 2 {
            return self.brute_force_search(query, ef_search);
        }

        self.ensure_hnsw_built();

        let cache = self.hnsw_cache.read();
        let hnsw = cache.as_ref().unwrap();

        let limit = ef_search.min(n);

        let neighbors = match hnsw {
            HnswIndex::FP32(h) => h.search(query, limit, limit),
            HnswIndex::BF16(h) => {
                let q_bf16: Vec<u16> = query
                    .iter()
                    .map(|&x| crate::mstg::scalar_quant::fp32_to_bf16(x))
                    .collect();
                h.search(&q_bf16, limit, limit)
            }
            HnswIndex::FP16(h) => {
                let q_fp16: Vec<half::f16> =
                    query.iter().map(|&x| half::f16::from_f32(x)).collect();
                h.search(&q_fp16, limit, limit)
            }
            HnswIndex::INT8(h) => {
                let (scale, offset) =
                    if let CentroidData::INT8 { scale, offset, .. } = &self.centroids {
                        (*scale, *offset)
                    } else {
                        (1.0, 0.0)
                    };
                let q_int8: Vec<i8> = query
                    .iter()
                    .map(|&x| {
                        let q = (x - offset) / scale;
                        q.clamp(-128.0, 127.0).round() as i8
                    })
                    .collect();
                h.search(&q_int8, limit, limit)
            }
        };

        neighbors
            .iter()
            .map(|neighbor| (self.centroid_ids[neighbor.d_id], neighbor.distance))
            .collect()
    }

    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = match &self.centroids {
            CentroidData::FP32(data) => data
                .iter()
                .enumerate()
                .map(|(idx, centroid)| (self.centroid_ids[idx], Self::l2_distance(query, centroid)))
                .collect(),
            CentroidData::BF16(data) => {
                let q_bf16: Vec<u16> = query
                    .iter()
                    .map(|&x| crate::mstg::scalar_quant::fp32_to_bf16(x))
                    .collect();
                let dist = DistBF16;
                data.iter()
                    .enumerate()
                    .map(|(idx, centroid)| (self.centroid_ids[idx], dist.eval(&q_bf16, centroid)))
                    .collect()
            }
            CentroidData::FP16(data) => {
                let q_fp16: Vec<half::f16> =
                    query.iter().map(|&x| half::f16::from_f32(x)).collect();
                let dist = DistFP16;
                data.iter()
                    .enumerate()
                    .map(|(idx, centroid)| (self.centroid_ids[idx], dist.eval(&q_fp16, centroid)))
                    .collect()
            }
            CentroidData::INT8 {
                data,
                scale,
                offset,
            } => {
                let q_int8: Vec<i8> = query
                    .iter()
                    .map(|&x| {
                        let q = (x - offset) / scale;
                        q.clamp(-128.0, 127.0).round() as i8
                    })
                    .collect();
                let dist = DistINT8 { scale: *scale };
                data.iter()
                    .enumerate()
                    .map(|(idx, centroid)| (self.centroid_ids[idx], dist.eval(&q_int8, centroid)))
                    .collect()
            }
        };

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
    }

    pub fn len(&self) -> usize {
        self.centroid_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centroid_ids.is_empty()
    }

    pub fn memory_usage(&self) -> usize {
        let vec_size = self.centroids.memory_size();
        vec_size + self.centroid_ids.len() * std::mem::size_of::<u32>()
    }

    /// Get the dimension of centroid vectors
    pub fn dimension(&self) -> Option<usize> {
        self.centroids.dimension()
    }
}
