use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};

fn main() {
    println!("=== L2 Distance Diagnostic Test (ARM64 platform) ===\n");

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
        Metric::L2,
        RotatorType::FhtKacRotator,
        42,
        false,
    )
    .unwrap();

    println!("Index trained successfully.\n");

    let params = SearchParams::new(10, 2); // nprobe=10, top_k=2

    let mut has_error = false;

    // Query each vector and check if we find it with near-zero distance
    for (idx, query) in vectors.iter().enumerate() {
        println!("Query {}: {:?}", idx, query);
        let results = index.search(query, params).unwrap();

        println!("  Results:");
        for (i, r) in results.iter().enumerate() {
            println!("    [{}] ID={}, score={:.6}", i, r.id, r.score);

            // Check for issues
            if r.score < 0.0 {
                println!("      ❌ ERROR: Negative L2 distance!");
                has_error = true;
            }
        }

        // Check if we found the query vector itself
        let found_self = results.iter().any(|r| r.id == idx);
        if found_self {
            let self_result = results.iter().find(|r| r.id == idx).unwrap();
            if self_result.score > 1.0 {
                println!(
                    "      ⚠️  WARNING: Self-distance is large: {:.6}",
                    self_result.score
                );
            } else {
                println!(
                    "      ✅ Found self with distance: {:.6}",
                    self_result.score
                );
            }
        } else {
            println!("      ⚠️  WARNING: Did not find query vector in results");
        }
        println!();
    }

    if has_error {
        println!("\n❌ FAILED: L2 distances should never be negative!");
        println!("This indicates a bug in the ARM64 SIMD implementation or distance calculation.");
        std::process::exit(1);
    } else {
        println!("\n✅ PASSED: All L2 distances are non-negative.");
    }
}
