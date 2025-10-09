use rabitq_rs::{
    BruteForceRabitqIndex, BruteForceSearchParams, IvfRabitqIndex, Metric, RabitqIndex,
    RotatorType, SearchParams,
};

fn main() {
    // Create some example data
    let dim = 64;
    let num_vectors = 200;

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

    // Train and save an IVF index
    println!("\n=== Training IVF Index ===");
    let ivf_index = IvfRabitqIndex::train(
        &data,
        16, // 16 clusters
        7,  // 7 bits
        Metric::L2,
        RotatorType::FhtKacRotator,
        42,
        false,
    )
    .expect("Failed to train IVF index");

    let ivf_path = "/tmp/ivf_index.bin";
    ivf_index
        .save_to_path(ivf_path)
        .expect("Failed to save IVF index");
    println!("IVF index saved to {}", ivf_path);

    // Train and save a BruteForce index
    println!("\n=== Training BruteForce Index ===");
    let bf_index = BruteForceRabitqIndex::train(
        &data,
        7, // 7 bits
        Metric::L2,
        RotatorType::FhtKacRotator,
        42,
        false,
    )
    .expect("Failed to train BruteForce index");

    let bf_path = "/tmp/brute_force_index.bin";
    bf_index
        .save_to_path(bf_path)
        .expect("Failed to save BruteForce index");
    println!("BruteForce index saved to {}", bf_path);

    // Use smart loader to load both indices without knowing their type
    println!("\n=== Smart Loading Indices ===");

    let query = &data[10];

    // Load IVF index using smart loader
    println!("\nLoading index from {}...", ivf_path);
    let loaded_index = RabitqIndex::load_from_path(ivf_path).expect("Failed to load index");

    match loaded_index {
        RabitqIndex::Ivf(ivf) => {
            println!("✓ Detected IVF index with {} clusters", ivf.cluster_count());
            let params = SearchParams::new(5, 16); // top_k=5, nprobe=16
            let results = ivf.search(query, params).expect("IVF search failed");
            println!("  Search results (top 5):");
            for (i, result) in results.iter().enumerate() {
                println!(
                    "    {}. ID: {}, Score: {:.4}",
                    i + 1,
                    result.id,
                    result.score
                );
            }
        }
        RabitqIndex::BruteForce(_) => {
            panic!("Expected IVF index but got BruteForce!");
        }
    }

    // Load BruteForce index using smart loader
    println!("\nLoading index from {}...", bf_path);
    let loaded_index = RabitqIndex::load_from_path(bf_path).expect("Failed to load index");

    match loaded_index {
        RabitqIndex::BruteForce(bf) => {
            println!("✓ Detected BruteForce index with {} vectors", bf.len());
            let params = BruteForceSearchParams::new(5); // top_k=5
            let results = bf.search(query, params).expect("BruteForce search failed");
            println!("  Search results (top 5):");
            for (i, result) in results.iter().enumerate() {
                println!(
                    "    {}. ID: {}, Score: {:.4}",
                    i + 1,
                    result.id,
                    result.score
                );
            }
        }
        RabitqIndex::Ivf(_) => {
            panic!("Expected BruteForce index but got IVF!");
        }
    }

    // Demonstrate helper methods
    println!("\n=== Using Helper Methods ===");

    let ivf_loaded = RabitqIndex::load_from_path(ivf_path).expect("Failed to load");
    println!("Is IVF: {}", ivf_loaded.is_ivf());
    println!("Is BruteForce: {}", ivf_loaded.is_brute_force());
    println!("Vector count: {}", ivf_loaded.len());

    // Use as_ivf() for safe access
    if let Some(ivf) = ivf_loaded.as_ivf() {
        println!(
            "IVF index has {} clusters and {} vectors",
            ivf.cluster_count(),
            ivf.len()
        );
    }

    let bf_loaded = RabitqIndex::load_from_path(bf_path).expect("Failed to load");
    println!("\nIs IVF: {}", bf_loaded.is_ivf());
    println!("Is BruteForce: {}", bf_loaded.is_brute_force());
    println!("Vector count: {}", bf_loaded.len());

    // Use as_brute_force() for safe access
    if let Some(bf) = bf_loaded.as_brute_force() {
        println!("BruteForce index has {} vectors", bf.len());
    }

    // Demonstrate automatic type detection in a generic function
    println!("\n=== Generic Index Processing ===");
    process_any_index(ivf_path);
    process_any_index(bf_path);

    println!("\n✓ Smart loader demonstration complete!");
}

// Generic function that works with any index type
fn process_any_index(path: &str) {
    let index = RabitqIndex::load_from_path(path).expect("Failed to load index");

    println!("\nProcessing index from: {}", path);
    println!(
        "  Type: {}",
        if index.is_ivf() { "IVF" } else { "BruteForce" }
    );
    println!("  Vectors: {}", index.len());
    println!("  Empty: {}", index.is_empty());
}
