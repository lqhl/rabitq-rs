use rabitq_rs::math::{dot, l2_distance_sqr, l2_norm_sqr};
use rabitq_rs::quantizer::{quantize_with_centroid, RabitqConfig};
use rabitq_rs::rotation::{DynamicRotator, RotatorType};
use rabitq_rs::Metric;

fn main() {
    println!("=== Quantization Debug Tool ===");
    println!("Platform: {}\n", std::env::consts::ARCH);

    // Use simple, predictable vectors
    let dim = 4;
    let data = vec![1.0, 0.0, 0.0, 0.0];
    let centroid = vec![0.0, 0.0, 0.0, 0.0]; // Zero centroid for simplicity

    println!("Vector: {:?}", data);
    println!("Centroid: {:?}", centroid);
    println!("Dim: {}\n", dim);

    // Test basic math operations first
    println!("=== Basic Math Operations ===");
    let norm = l2_norm_sqr(&data);
    println!("l2_norm_sqr(data) = {} (expected: 1.0)", norm);

    let dist = l2_distance_sqr(&data, &centroid);
    println!("l2_distance_sqr(data, centroid) = {} (expected: 1.0)", dist);

    let dot_val = dot(&data, &centroid);
    println!("dot(data, centroid) = {} (expected: 0.0)\n", dot_val);

    // Test with rotation
    println!("=== With Rotation (FHT-Kac) ===");
    let rotator = DynamicRotator::new(dim, RotatorType::FhtKacRotator, 42);
    let rotated = rotator.rotate(&data);
    println!("Rotated vector: {:?}", rotated);

    let rotated_norm = l2_norm_sqr(&rotated);
    println!(
        "l2_norm_sqr(rotated) = {} (should be ~1.0 for orthonormal rotation)",
        rotated_norm
    );

    // Calculate residual (data - centroid in rotated space)
    let rotated_centroid = rotator.rotate(&centroid);
    let mut residual = vec![0.0f32; dim];
    for i in 0..dim {
        residual[i] = rotated[i] - rotated_centroid[i];
    }
    println!("Residual: {:?}\n", residual);

    // Test quantization with different bit configurations
    for total_bits in [1, 3, 7] {
        println!("=== Quantization with {} bits ===", total_bits);
        test_quantization(&residual, &rotated_centroid, total_bits, Metric::L2, dim);
        println!();
    }

    // Test with InnerProduct metric
    println!("=== Testing with InnerProduct Metric ===");
    test_quantization(&residual, &rotated_centroid, 7, Metric::InnerProduct, dim);
}

fn test_quantization(
    residual: &[f32],
    centroid: &[f32],
    total_bits: usize,
    metric: Metric,
    dim: usize,
) {
    let config = RabitqConfig::new(total_bits);
    let quantized = quantize_with_centroid(residual, centroid, &config, metric);

    println!("Metric: {:?}", metric);
    println!("Total bits: {}", total_bits);
    println!("Ex bits: {}", quantized.ex_bits);

    // Print quantization factors
    println!("\nQuantization Factors:");
    println!("  delta:         {:.10}", quantized.delta);
    println!("  vl:            {:.10}", quantized.vl);
    println!("  f_add:         {:.10}", quantized.f_add);
    println!("  f_rescale:     {:.10}", quantized.f_rescale);
    println!("  f_error:       {:.10}", quantized.f_error);
    println!("  residual_norm: {:.10}", quantized.residual_norm);
    println!("  f_add_ex:      {:.10}", quantized.f_add_ex);
    println!("  f_rescale_ex:  {:.10}", quantized.f_rescale_ex);

    // Check for suspicious values
    let mut issues = Vec::new();

    if !quantized.f_add.is_finite() {
        issues.push("f_add is not finite");
    }
    if !quantized.f_rescale.is_finite() {
        issues.push("f_rescale is not finite");
    }
    if !quantized.f_add_ex.is_finite() {
        issues.push("f_add_ex is not finite");
    }
    if !quantized.f_rescale_ex.is_finite() {
        issues.push("f_rescale_ex is not finite");
    }

    // For L2, check if factors could lead to negative distances
    if metric == Metric::L2 {
        // A very rough check: if f_add is very negative and f_rescale is positive,
        // we might get negative distances
        if quantized.f_add < -100.0 {
            issues.push("f_add is very negative (< -100)");
        }
        if quantized.f_add_ex < -100.0 {
            issues.push("f_add_ex is very negative (< -100)");
        }
    }

    if issues.is_empty() {
        println!("\n✅ No obvious issues detected");
    } else {
        println!("\n⚠️  Potential issues:");
        for issue in issues {
            println!("  - {}", issue);
        }
    }

    // Print binary code preview
    let binary_unpacked = quantized.unpack_binary_code();
    println!(
        "\nBinary code (first 4 bits): {:?}",
        &binary_unpacked[..dim.min(4)]
    );

    if quantized.ex_bits > 0 {
        let ex_unpacked = quantized.unpack_ex_code();
        println!("Extended code (first 4): {:?}", &ex_unpacked[..dim.min(4)]);
    }

    // Simulate distance calculation
    println!("\n--- Simulated Distance Calculation ---");

    // For a self-query (query == data), we expect:
    // - residual ~= 0
    // - distance ~= 0
    let query = residual; // Same as residual for this test

    // Compute binary dot product
    let binary_unpacked = quantized.unpack_binary_code();
    let mut binary_dot = 0.0f32;
    for i in 0..dim {
        let bit = if binary_unpacked[i] > 0 { 1.0 } else { -1.0 };
        binary_dot += query[i] * bit;
    }
    println!("Binary dot product: {:.6}", binary_dot);

    // Compute extended dot product if applicable
    let mut ex_dot = 0.0f32;
    if quantized.ex_bits > 0 {
        let ex_unpacked = quantized.unpack_ex_code();
        for i in 0..dim {
            ex_dot += query[i] * ex_unpacked[i] as f32;
        }
        println!("Extended dot product: {:.6}", ex_dot);
    }

    // Estimate distance using quantization factors
    let g_add = match metric {
        Metric::L2 => l2_distance_sqr(query, centroid),
        Metric::InnerProduct => -dot(query, centroid),
    };

    let distance = if quantized.ex_bits > 0 {
        // Using extended code path
        let query_norm = l2_norm_sqr(query).sqrt();
        let ipnorm_inv = if query_norm > 1e-6 {
            1.0 / query_norm
        } else {
            0.0
        };
        let total_term = binary_dot * quantized.delta + ex_dot;
        quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term
    } else {
        // Binary-only path
        let cb = -((1 << 0) as f32 - 0.5); // ex_bits = 0
        let ip_cent_xucb = dot(
            centroid,
            &binary_unpacked
                .iter()
                .map(|&b| if b > 0 { 1.0 } else { -1.0 })
                .collect::<Vec<_>>(),
        );
        let total_term = binary_dot * quantized.delta + ip_cent_xucb * quantized.vl;
        quantized.f_add + g_add + quantized.f_rescale * total_term
    };

    println!("g_add: {:.6}", g_add);
    println!("Estimated distance: {:.6}", distance);

    if metric == Metric::L2 && distance < 0.0 {
        println!("\n❌ ERROR: Distance is NEGATIVE for L2 metric!");
    } else if metric == Metric::L2 && distance > 10.0 {
        println!(
            "\n⚠️  WARNING: Distance is unexpectedly large: {:.6}",
            distance
        );
    } else {
        println!("\n✅ Distance looks reasonable");
    }
}
