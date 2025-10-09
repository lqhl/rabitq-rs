use rabitq_rs::{BruteForceRabitqIndex, BruteForceSearchParams, Metric, RotatorType};

fn main() {
    // Create some example data - smaller dataset suitable for brute-force
    let dim = 128;
    let num_vectors = 500;

    println!(
        "Generating {} vectors with dimension {}...",
        num_vectors, dim
    );
    let data: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32).sin() * ((j + 1) as f32).sqrt())
                .collect()
        })
        .collect();

    // Train a brute-force RaBitQ index (no clustering)
    println!("Training brute-force index...");
    let index = BruteForceRabitqIndex::train(
        &data,
        7,                          // 7 bits for quantization
        Metric::L2,                 // L2 distance metric
        RotatorType::FhtKacRotator, // Fast Hadamard Transform rotator
        42,                         // random seed
        false,                      // use accurate config (set true for faster training)
    )
    .expect("Failed to train brute-force index");

    println!("Index trained with {} vectors", index.len());

    // Create a query vector (same as vector 10 in the dataset)
    let query = &data[10];

    // Perform brute-force search
    let params = BruteForceSearchParams::new(10); // top_k=10
    let results = index.search(query, params).expect("Search failed");

    println!("\nBrute-force search results (top 10):");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. ID: {}, Score: {:.6}", i + 1, result.id, result.score);
    }

    // The first result should be the query vector itself
    assert_eq!(results[0].id, 10, "First result should be the query vector");
    println!("\n✓ Found query vector at position 1 (as expected)");

    // Test with a random query
    let random_query: Vec<f32> = (0..dim).map(|i| ((i + 123) as f32).cos()).collect();
    let random_results = index
        .search(&random_query, params)
        .expect("Random search failed");

    println!("\nRandom query results (top 10):");
    for (i, result) in random_results.iter().enumerate() {
        println!("  {}. ID: {}, Score: {:.6}", i + 1, result.id, result.score);
    }

    // Demonstrate index persistence
    println!("\nTesting index save/load...");
    let save_path = "/tmp/brute_force_index.bin";
    index.save_to_path(save_path).expect("Failed to save index");
    println!("Index saved to {}", save_path);

    let loaded_index =
        BruteForceRabitqIndex::load_from_path(save_path).expect("Failed to load index");
    println!("Index loaded from {}", save_path);

    // Verify loaded index produces same results
    let loaded_results = loaded_index
        .search(query, params)
        .expect("Search on loaded index failed");

    assert_eq!(results.len(), loaded_results.len(), "Result count mismatch");
    for (orig, loaded) in results.iter().zip(loaded_results.iter()) {
        assert_eq!(orig.id, loaded.id, "ID mismatch after load");
    }

    println!("✓ Loaded index produces identical results");

    // Demonstrate when to use brute-force vs IVF
    println!("\n=== When to use BruteForceRabitqIndex ===");
    println!("✓ Small to medium datasets (<200K vectors)");
    println!("✓ High-dimensional vectors (e.g., 1024D)");
    println!("✓ When IVF clustering overhead is not justified");
    println!("✓ When you need exhaustive search guarantees");
    println!("\n=== When to use IvfRabitqIndex ===");
    println!("✓ Large datasets (>200K vectors)");
    println!("✓ When you can tolerate approximate search");
    println!("✓ When clustering provides meaningful speedups");
}
