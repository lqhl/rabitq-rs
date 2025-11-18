//! Optimized distance estimation for MSTG posting lists
//!
//! This module provides a unified and optimized implementation for estimating
//! distances between queries and quantized vectors, avoiding redundant computations.

use crate::math::{dot, l2_distance_sqr};
use crate::{Metric, QuantizedVector};

/// Precomputed query context to avoid redundant calculations
///
/// This structure stores query-related constants that are computed once
/// per query and reused across all vectors in posting lists.
#[derive(Debug, Clone)]
pub struct QueryContext<'a> {
    /// Reference to the query vector (zero-copy)
    pub query: &'a [f32],
    /// Precomputed sum of query elements
    pub sum_query: f32,
    /// Number of extended bits
    pub ex_bits: u8,
    /// Binary scale factor: 2^ex_bits
    pub binary_scale: f32,
    /// Constant for extended code: -(2^ex_bits - 0.5)
    pub cb: f32,
    /// Constant for binary code: -0.5
    pub c1: f32,
}

impl<'a> QueryContext<'a> {
    /// Create a new query context with precomputed values
    ///
    /// # Performance
    /// This function computes O(dim) sum once, which saves O(N×dim) operations
    /// when computing distances to N vectors in a posting list.
    #[inline]
    pub fn new(query: &'a [f32], ex_bits: u8) -> Self {
        let sum_query: f32 = query.iter().sum();
        let binary_scale = (1 << ex_bits) as f32;
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let c1 = -0.5f32;

        Self {
            query,
            sum_query,
            ex_bits,
            binary_scale,
            cb,
            c1,
        }
    }
}

/// Estimate distance from query to a quantized vector
///
/// This is the optimized distance estimation function that uses precomputed
/// query constants to avoid redundant calculations.
///
/// # Arguments
/// * `ctx` - Precomputed query context
/// * `centroid` - Cluster centroid vector
/// * `quantized` - Quantized vector representation
/// * `metric` - Distance metric (L2 or InnerProduct)
///
/// # Returns
/// Approximate distance (or negative similarity for InnerProduct)
#[inline]
pub fn estimate_distance(
    ctx: &QueryContext,
    centroid: &[f32],
    quantized: &QuantizedVector,
    metric: Metric,
) -> f32 {
    // Step 1: Compute g_add (query-to-centroid distance component)
    let g_add = match metric {
        Metric::L2 => l2_distance_sqr(ctx.query, centroid),
        Metric::InnerProduct => -dot(ctx.query, centroid),
    };

    // Step 2: Compute binary code contribution
    let binary_code = quantized.unpack_binary_code();
    let binary_dot: f32 = binary_code
        .iter()
        .zip(ctx.query.iter())
        .map(|(&bit, &q)| (bit as f32) * q)
        .sum();

    let binary_term = binary_dot + ctx.c1 * ctx.sum_query;
    let distance_1bit = quantized.f_add + g_add + quantized.f_rescale * binary_term;

    // Step 3: Add extended code contribution if available
    let final_distance = if ctx.ex_bits > 0 {
        let ex_code = quantized.unpack_ex_code();
        let ex_dot: f32 = ex_code
            .iter()
            .zip(ctx.query.iter())
            .map(|(&code, &q)| (code as f32) * q)
            .sum();

        let total_term = ctx.binary_scale * binary_dot + ex_dot + ctx.cb * ctx.sum_query;
        let distance_ex = quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term;

        // Ensure finite result
        if distance_ex.is_finite() {
            distance_ex
        } else {
            distance_1bit
        }
    } else {
        distance_1bit
    };

    // For L2 metric, clamp negative distances to 0 (due to quantization approximation errors)
    // For InnerProduct, negative values are valid (higher similarity)
    if metric == Metric::L2 {
        final_distance.max(0.0)
    } else {
        final_distance
    }
}

/// Batch estimate distances for all vectors in a posting list
///
/// This is a convenience function that creates a QueryContext and computes
/// distances to all vectors in the provided list.
pub fn estimate_distances_batch(
    query: &[f32],
    centroid: &[f32],
    quantized_vecs: &[QuantizedVector],
    ex_bits: u8,
    metric: Metric,
) -> Vec<f32> {
    let ctx = QueryContext::new(query, ex_bits);

    quantized_vecs
        .iter()
        .map(|qvec| estimate_distance(&ctx, centroid, qvec, metric))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::quantize_with_centroid;
    use crate::RabitqConfig;
    use rand::prelude::*;

    fn generate_test_vectors(n: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<QuantizedVector>) {
        let mut rng = StdRng::seed_from_u64(42);
        let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let centroid: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

        let config = RabitqConfig::new(7);
        let vectors: Vec<QuantizedVector> = (0..n)
            .map(|_| {
                let data: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                quantize_with_centroid(&data, &centroid, &config, Metric::L2)
            })
            .collect();

        (query, centroid, vectors)
    }

    #[test]
    fn test_query_context_creation() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let ctx = QueryContext::new(&query, 6);

        assert_eq!(ctx.query, &query);
        assert_eq!(ctx.sum_query, 10.0);
        assert_eq!(ctx.ex_bits, 6);
        assert_eq!(ctx.binary_scale, 64.0);
        assert_eq!(ctx.cb, -63.5);
        assert_eq!(ctx.c1, -0.5);
    }

    #[test]
    fn test_estimate_distance_l2() {
        let (query, centroid, vectors) = generate_test_vectors(10, 128);
        let ctx = QueryContext::new(&query, 6);

        for qvec in &vectors {
            let dist = estimate_distance(&ctx, &centroid, qvec, Metric::L2);
            assert!(dist.is_finite(), "Distance should be finite");
            assert!(dist >= 0.0, "L2 distance should be non-negative");
        }
    }

    #[test]
    fn test_estimate_distance_inner_product() {
        let (query, centroid, vectors) = generate_test_vectors(10, 128);
        let ctx = QueryContext::new(&query, 6);

        for qvec in &vectors {
            let dist = estimate_distance(&ctx, &centroid, qvec, Metric::InnerProduct);
            assert!(dist.is_finite(), "Distance should be finite");
        }
    }

    #[test]
    fn test_batch_estimation() {
        let (query, centroid, vectors) = generate_test_vectors(100, 960);
        let distances = estimate_distances_batch(&query, &centroid, &vectors, 6, Metric::L2);

        assert_eq!(distances.len(), 100);
        for dist in &distances {
            assert!(dist.is_finite());
        }
    }

    #[test]
    fn test_consistency_with_individual_calls() {
        let (query, centroid, vectors) = generate_test_vectors(50, 256);

        // Batch computation
        let batch_distances = estimate_distances_batch(&query, &centroid, &vectors, 6, Metric::L2);

        // Individual computation with reused context
        let ctx = QueryContext::new(&query, 6);
        let individual_distances: Vec<f32> = vectors
            .iter()
            .map(|qvec| estimate_distance(&ctx, &centroid, qvec, Metric::L2))
            .collect();

        // Should be identical
        for (batch, individual) in batch_distances.iter().zip(individual_distances.iter()) {
            assert!(
                (batch - individual).abs() < 1e-6,
                "Batch and individual should match"
            );
        }
    }
}
