use crate::Metric;
use serde::{Deserialize, Serialize};

/// Scalar quantization precision for centroid vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarPrecision {
    /// Full precision (4 bytes/dim)
    FP32,
    /// Brain floating point (2 bytes/dim, hardware accelerated)
    BF16,
    /// Half precision (2 bytes/dim)
    FP16,
    /// 8-bit integer quantization (1 byte/dim, requires training)
    INT8,
}

impl ScalarPrecision {
    pub fn bytes_per_dim(&self) -> usize {
        match self {
            ScalarPrecision::FP32 => 4,
            ScalarPrecision::BF16 => 2,
            ScalarPrecision::FP16 => 2,
            ScalarPrecision::INT8 => 1,
        }
    }

    pub fn memory_multiplier(&self) -> f32 {
        match self {
            ScalarPrecision::FP32 => 1.0,
            ScalarPrecision::BF16 => 0.5,
            ScalarPrecision::FP16 => 0.5,
            ScalarPrecision::INT8 => 0.25,
        }
    }
}

/// MSTG index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MstgConfig {
    // Clustering parameters
    pub max_posting_size: usize,
    pub branching_factor: usize,
    pub balance_weight: f32,

    // Closure assignment
    pub closure_epsilon: f32,
    pub max_replicas: usize,

    // RaBitQ parameters
    pub rabitq_bits: usize,
    pub faster_config: bool,
    pub metric: Metric,

    // HNSW parameters
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub centroid_precision: ScalarPrecision,

    // Search parameters
    pub default_ef_search: usize,
    pub pruning_epsilon: f32,
}

impl Default for MstgConfig {
    fn default() -> Self {
        Self {
            // Clustering
            max_posting_size: 5000,
            branching_factor: 10,
            balance_weight: 1.0,

            // Closure assignment
            closure_epsilon: 0.15,
            max_replicas: 8,

            // RaBitQ
            rabitq_bits: 7,
            faster_config: false,
            metric: Metric::L2,

            // HNSW
            hnsw_m: 32,
            hnsw_ef_construction: 200,
            centroid_precision: ScalarPrecision::BF16, // Default to bf16

            // Search
            default_ef_search: 150,
            pruning_epsilon: 0.6,
        }
    }
}

/// Search parameters for MSTG queries
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub ef_search: usize,
    pub pruning_epsilon: f32,
    pub top_k: usize,
}

impl SearchParams {
    pub fn new(ef_search: usize, pruning_epsilon: f32, top_k: usize) -> Self {
        Self {
            ef_search,
            pruning_epsilon,
            top_k,
        }
    }

    /// Create search params optimized for high recall (95%+)
    pub fn high_recall(top_k: usize) -> Self {
        Self {
            ef_search: 300,
            pruning_epsilon: 0.8,
            top_k,
        }
    }

    /// Create search params optimized for balanced recall (~90%)
    pub fn balanced(top_k: usize) -> Self {
        Self {
            ef_search: 150,
            pruning_epsilon: 0.6,
            top_k,
        }
    }

    /// Create search params optimized for low latency (~80% recall)
    pub fn low_latency(top_k: usize) -> Self {
        Self {
            ef_search: 50,
            pruning_epsilon: 0.4,
            top_k,
        }
    }
}

impl Default for SearchParams {
    fn default() -> Self {
        Self::balanced(100)
    }
}
