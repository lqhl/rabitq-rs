use rabitq_rs::{IvfRabitqIndex, Metric, RoaringBitmap, RotatorType, SearchParams};

fn main() {
    // Create some example data
    let dim = 32;
    let data: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect())
        .collect();

    // Train an index
    println!("Training index...");
    let index = IvfRabitqIndex::train(
        &data,
        16,                         // nlist: 16 clusters
        7,                          // 7 bits for quantization
        Metric::L2,                 // L2 distance metric
        RotatorType::FhtKacRotator, // Fast Hadamard Transform rotator
        42,                         // random seed
        false,                      // use accurate config
    )
    .expect("Failed to train index");

    println!("Index trained with {} vectors", index.len());

    // Create a query vector (same as vector 10 in the dataset)
    let query = &data[10];

    // Regular search (no filter)
    let params = SearchParams::new(5, 16); // top_k=5, nprobe=16
    let results = index.search(query, params).expect("Search failed");

    println!("\nRegular search results:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. ID: {}, Score: {:.4}", i + 1, result.id, result.score);
    }

    // Filtered search - only search among IDs 0-19
    let mut filter = RoaringBitmap::new();
    for id in 0..20 {
        filter.insert(id);
    }

    let filtered_results = index
        .search_filtered(query, params, &filter)
        .expect("Filtered search failed");

    println!("\nFiltered search results (IDs 0-19 only):");
    for (i, result) in filtered_results.iter().enumerate() {
        println!("  {}. ID: {}, Score: {:.4}", i + 1, result.id, result.score);
    }

    // Verify all results are in the filter
    for result in &filtered_results {
        assert!(
            filter.contains(result.id as u32),
            "Result ID {} not in filter!",
            result.id
        );
    }

    println!("\n✓ All filtered results are within the specified ID range");
}
