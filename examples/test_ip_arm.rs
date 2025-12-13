use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};

fn main() {
    println!("=== Inner Product Diagnostic Test (ARM64 platform) ===\n");

    // Create simple test vectors
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    println!(
        "Training index with {} vectors (dim={})...",
        vectors.len(),
        vectors[0].len()
    );
    let index = IvfRabitqIndex::train(
        &vectors,
        2, // 2 clusters
        7, // 7 bits
        Metric::InnerProduct,
        RotatorType::FhtKacRotator,
        42,
        false,
    )
    .unwrap();

    println!("Index trained successfully.\n");

    let params = SearchParams::new(10, 4); // nprobe=10, top_k=4

    let mut has_error = false;
    let mut has_warning = false;

    // Query each vector and check if we find it with high score
    for (idx, query) in vectors.iter().enumerate() {
        println!("Query {}: {:?}", idx, query);
        let results = index.search(query, params).unwrap();

        println!("  Results:");
        for (i, r) in results.iter().enumerate() {
            println!("    [{}] ID={}, score={:.6}", i, r.id, r.score);
        }

        // Check if we found the query vector itself
        let found_self = results.iter().any(|r| r.id == idx);
        if found_self {
            let self_result = results.iter().find(|r| r.id == idx).unwrap();
            let self_position = results.iter().position(|r| r.id == idx).unwrap();

            // For inner product of a normalized vector with itself, score should be ~1.0
            // But due to quantization, it might be different
            if self_result.score < 0.5 {
                println!(
                    "      ⚠️  WARNING: Self-score is low: {:.6} (position {})",
                    self_result.score, self_position
                );
                has_warning = true;
            } else if self_position > 0 {
                println!(
                    "      ⚠️  WARNING: Self not ranked first (position {}, score: {:.6})",
                    self_position, self_result.score
                );
                has_warning = true;
            } else {
                println!(
                    "      ✅ Found self at top with score: {:.6}",
                    self_result.score
                );
            }
        } else {
            println!("      ❌ ERROR: Did not find query vector in results!");
            has_error = true;
        }

        // Check for NaN or Inf scores
        for r in &results {
            if !r.score.is_finite() {
                println!("      ❌ ERROR: Non-finite score detected: {}", r.score);
                has_error = true;
            }
        }

        // For orthogonal vectors, inner product should be ~0
        // Check if we see reasonable scores for non-self vectors
        let non_self_scores: Vec<f32> = results
            .iter()
            .filter(|r| r.id != idx)
            .map(|r| r.score)
            .collect();

        if !non_self_scores.is_empty() {
            let max_non_self = non_self_scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            println!("      Max non-self score: {:.6}", max_non_self);
        }

        println!();
    }

    println!("\n=== Summary ===");
    if has_error {
        println!("❌ FAILED: Critical errors detected!");
        println!("   - Inner Product search is not working correctly on this platform.");
        println!("   - See ARM64_KNOWN_ISSUES.md for details.");
        std::process::exit(1);
    } else if has_warning {
        println!("⚠️  PARTIAL: Tests passed but with warnings.");
        println!("   - Results may not be optimal on this platform.");
        println!("   - Consider checking quantization precision.");
    } else {
        println!("✅ PASSED: Inner Product search appears to work correctly.");
    }
}
