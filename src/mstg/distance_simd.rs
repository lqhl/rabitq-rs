//! SIMD-accelerated distance estimation for MSTG posting lists
//!
//! This module provides optimized distance calculations using AVX2/AVX-512 SIMD instructions
//! to accelerate the hot path in MSTG search.

use super::distance::{estimate_distance, QueryContext};
use crate::{Metric, QuantizedVector};

/// SIMD-accelerated distance estimation (auto-detects CPU features)
///
/// This is the main entry point that automatically selects the best implementation
/// based on available CPU features.
///
/// # Performance
/// Expected speedup: 3-5x compared to scalar implementation on AVX2 CPUs
#[inline]
pub fn estimate_distance_fast(
    ctx: &QueryContext,
    centroid: &[f32],
    quantized: &QuantizedVector,
    metric: Metric,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Try AVX-512 first (best performance) - auto-detected at runtime
        // Only available with nightly Rust and avx512 feature
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { estimate_distance_avx512(ctx, centroid, quantized, metric) };
            }
        }

        // Fall back to AVX2 (widely available)
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { estimate_distance_avx2(ctx, centroid, quantized, metric) };
        }
    }

    // Fallback to scalar version for non-x86 or old CPUs
    estimate_distance(ctx, centroid, quantized, metric)
}

// ============================================================================
// AVX2 Implementation (256-bit SIMD)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn estimate_distance_avx2(
    ctx: &QueryContext,
    centroid: &[f32],
    quantized: &QuantizedVector,
    metric: Metric,
) -> f32 {

    // Step 1: Compute g_add (query-to-centroid distance component)
    let g_add = match metric {
        Metric::L2 => l2_distance_sqr_avx2(ctx.query, centroid),
        Metric::InnerProduct => -dot_avx2(ctx.query, centroid),
    };

    // Step 2: Binary code dot product with SIMD
    let binary_code = quantized.unpack_binary_code();
    let binary_dot = binary_u8_dot_f32_avx2(ctx.query, &binary_code);

    let binary_term = binary_dot + ctx.c1 * ctx.sum_query;
    let distance_1bit = quantized.f_add + g_add + quantized.f_rescale * binary_term;

    // Step 3: Extended code contribution (if present)
    if ctx.ex_bits > 0 {
        let ex_code = quantized.unpack_ex_code();
        // Convert u16 ex_code to u8 for SIMD (most values fit in u8 range for 7-bit encoding)
        let ex_code_u8: Vec<u8> = ex_code.iter().map(|&x| x.min(255) as u8).collect();
        let ex_dot = ex_u8_dot_f32_avx2(ctx.query, &ex_code_u8);

        let total_term = ctx.binary_scale * binary_dot + ex_dot + ctx.cb * ctx.sum_query;
        let distance_ex = quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term;
        distance_ex
    } else {
        distance_1bit
    }
}

/// Compute dot product between f32 query and u8 binary code (0 or 1) using AVX2
///
/// # Safety
/// Requires AVX2 support. Caller must check CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn binary_u8_dot_f32_avx2(query: &[f32], binary_code: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let len = query.len().min(binary_code.len());
    let mut sum = _mm256_setzero_ps();

    // Process 8 elements at a time
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 f32 values from query
        let q = _mm256_loadu_ps(query.as_ptr().add(offset));

        // Load 8 u8 values and convert to f32
        // Note: binary_code contains only 0 or 1
        let b_u8 = _mm_loadl_epi64(binary_code.as_ptr().add(offset) as *const __m128i);
        let b_i32 = _mm256_cvtepu8_epi32(b_u8);
        let b = _mm256_cvtepi32_ps(b_i32);

        // Multiply and accumulate using FMA
        sum = _mm256_fmadd_ps(q, b, sum);
    }

    // Horizontal sum using hadd
    let sum = _mm256_hadd_ps(sum, sum);
    let sum = _mm256_hadd_ps(sum, sum);

    // Extract result (sum of all 8 lanes)
    let lo = _mm256_extractf128_ps(sum, 0);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ss(lo, hi);
    let mut result = _mm_cvtss_f32(sum128);

    // Handle remaining elements (scalar)
    for i in (chunks * 8)..len {
        result += query[i] * (binary_code[i] as f32);
    }

    result
}

/// Compute dot product between f32 query and u8 ex_code (0-127) using AVX2
///
/// # Safety
/// Requires AVX2 support. Caller must check CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ex_u8_dot_f32_avx2(query: &[f32], ex_code: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let len = query.len().min(ex_code.len());
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;

        let q = _mm256_loadu_ps(query.as_ptr().add(offset));

        // Load and convert u8 to f32
        let ex_u8 = _mm_loadl_epi64(ex_code.as_ptr().add(offset) as *const __m128i);
        let ex_i32 = _mm256_cvtepu8_epi32(ex_u8);
        let ex = _mm256_cvtepi32_ps(ex_i32);

        sum = _mm256_fmadd_ps(q, ex, sum);
    }

    // Horizontal sum
    let sum = _mm256_hadd_ps(sum, sum);
    let sum = _mm256_hadd_ps(sum, sum);
    let lo = _mm256_extractf128_ps(sum, 0);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ss(lo, hi);
    let mut result = _mm_cvtss_f32(sum128);

    for i in (chunks * 8)..len {
        result += query[i] * (ex_code[i] as f32);
    }

    result
}

/// L2 distance squared using AVX2 with FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_distance_sqr_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let sum = _mm256_hadd_ps(sum, sum);
    let sum = _mm256_hadd_ps(sum, sum);
    let lo = _mm256_extractf128_ps(sum, 0);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ss(lo, hi);
    let mut result = _mm_cvtss_f32(sum128);

    // Remainder
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// Dot product using AVX2 with FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
    }

    // Horizontal sum
    let sum = _mm256_hadd_ps(sum, sum);
    let sum = _mm256_hadd_ps(sum, sum);
    let lo = _mm256_extractf128_ps(sum, 0);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ss(lo, hi);
    let mut result = _mm_cvtss_f32(sum128);

    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

// ============================================================================
// AVX-512 Implementation (512-bit SIMD) - Auto-detected at runtime
// Note: Requires nightly Rust and avx512 feature flag
// ============================================================================

#[cfg(feature = "avx512")]
/// Compute dot product between f32 query and u8 binary code using AVX-512
///
/// # Safety
/// Requires AVX-512F support. Caller must check CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn binary_u8_dot_f32_avx512(query: &[f32], binary_code: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let len = query.len().min(binary_code.len());
    let mut sum = _mm512_setzero_ps();

    // Process 16 elements at a time (512-bit SIMD)
    let chunks = len / 16;
    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 f32 values from query
        let q = _mm512_loadu_ps(query.as_ptr().add(offset));

        // Load 16 u8 values and convert to f32
        // Load to 128-bit register first
        let b_u8 = _mm_loadu_si128(binary_code.as_ptr().add(offset) as *const __m128i);
        // Convert u8 -> i32 -> f32
        let b_i32 = _mm512_cvtepu8_epi32(b_u8);
        let b = _mm512_cvtepi32_ps(b_i32);

        // Multiply and accumulate using FMA
        sum = _mm512_fmadd_ps(q, b, sum);
    }

    // Horizontal sum using AVX-512 reduce instruction (much simpler than AVX2!)
    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remaining elements (scalar)
    for i in (chunks * 16)..len {
        result += query[i] * (binary_code[i] as f32);
    }

    result
}

#[cfg(feature = "avx512")]
/// Compute dot product between f32 query and u8 ex_code using AVX-512
///
/// # Safety
/// Requires AVX-512F support. Caller must check CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn ex_u8_dot_f32_avx512(query: &[f32], ex_code: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let len = query.len().min(ex_code.len());
    let mut sum = _mm512_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let offset = i * 16;

        let q = _mm512_loadu_ps(query.as_ptr().add(offset));

        // Load and convert u8 to f32
        let ex_u8 = _mm_loadu_si128(ex_code.as_ptr().add(offset) as *const __m128i);
        let ex_i32 = _mm512_cvtepu8_epi32(ex_u8);
        let ex = _mm512_cvtepi32_ps(ex_i32);

        sum = _mm512_fmadd_ps(q, ex, sum);
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    for i in (chunks * 16)..len {
        result += query[i] * (ex_code[i] as f32);
    }

    result
}

#[cfg(feature = "avx512")]
/// L2 distance squared using AVX-512 with FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_sqr_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm512_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm512_sub_ps(a_vec, b_vec);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    // Remainder
    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

#[cfg(feature = "avx512")]
/// Dot product using AVX-512 with FMA
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm512_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(offset));
        sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    for i in (chunks * 16)..len {
        result += a[i] * b[i];
    }

    result
}

#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn estimate_distance_avx512(
    ctx: &QueryContext,
    centroid: &[f32],
    quantized: &QuantizedVector,
    metric: Metric,
) -> f32 {
    // Step 1: Compute g_add (query-to-centroid distance component)
    let g_add = match metric {
        Metric::L2 => l2_distance_sqr_avx512(ctx.query, centroid),
        Metric::InnerProduct => -dot_avx512(ctx.query, centroid),
    };

    // Step 2: Binary code dot product with SIMD
    let binary_code = quantized.unpack_binary_code();
    let binary_dot = binary_u8_dot_f32_avx512(ctx.query, &binary_code);

    let binary_term = binary_dot + ctx.c1 * ctx.sum_query;
    let distance_1bit = quantized.f_add + g_add + quantized.f_rescale * binary_term;

    // Step 3: Extended code contribution (if present)
    if ctx.ex_bits > 0 {
        let ex_code = quantized.unpack_ex_code();
        // Convert u16 ex_code to u8 for SIMD
        let ex_code_u8: Vec<u8> = ex_code.iter().map(|&x| x.min(255) as u8).collect();
        let ex_dot = ex_u8_dot_f32_avx512(ctx.query, &ex_code_u8);

        let total_term = ctx.binary_scale * binary_dot + ex_dot + ctx.cb * ctx.sum_query;
        let distance_ex = quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term;
        distance_ex
    } else {
        distance_1bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::{quantize_with_centroid, RabitqConfig};
    use crate::Metric;
    use rand::prelude::*;

    #[test]
    fn test_simd_vs_scalar() {
        let mut rng = StdRng::seed_from_u64(42);

        // Generate test data
        let dim = 960;
        let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let centroid: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let vector: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

        // Use RaBitQ
        let config = RabitqConfig::faster(dim, 7, 42);
        let quantized = quantize_with_centroid(&vector, &centroid, &config, Metric::L2);

        // Create query context (ex_bits = total_bits - 1 for the sign bit)
        let ex_bits = config.total_bits.saturating_sub(1) as u8;
        let ctx = QueryContext::new(&query, ex_bits);

        // Compare SIMD and scalar versions
        let result_scalar = estimate_distance(&ctx, &centroid, &quantized, Metric::L2);
        let result_simd = estimate_distance_fast(&ctx, &centroid, &quantized, Metric::L2);

        // Results should be very close (may have small floating point differences)
        let diff = (result_scalar - result_simd).abs();
        assert!(
            diff < 0.01,
            "SIMD and scalar results differ: {} vs {} (diff: {})",
            result_scalar,
            result_simd,
            diff
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_binary_dot_simd() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test on non-AVX2 CPU");
            return;
        }

        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let binary = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

        let result_simd = unsafe { binary_u8_dot_f32_avx2(&query, &binary) };

        // Expected: 1*1 + 3*1 + 5*1 + 7*1 + 9*1 = 25
        let expected = 25.0;

        assert!(
            (result_simd - expected).abs() < 0.001,
            "Binary dot SIMD: got {}, expected {}",
            result_simd,
            expected
        );
    }
}
