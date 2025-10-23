//! Simple MSTG demonstration

use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams};
use rabitq_rs::Metric;
use rand::prelude::*;

fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen()).collect())
        .collect()
}

fn main() {
    println!("MSTG Demo");
    println!("=========\n");

    // Generate test data
    let n_vectors = 1000;
    let dim = 128;
    println!(
        "Generating {} random {}-dimensional vectors...",
        n_vectors, dim
    );
    let data = generate_test_data(n_vectors, dim);

    // Configure MSTG
    let config = MstgConfig {
        max_posting_size: 100,
        branching_factor: 5,
        balance_weight: 1.0,
        closure_epsilon: 0.15,
        max_replicas: 8,
        rabitq_bits: 7,
        faster_config: true,
        metric: Metric::L2,
        hnsw_m: 32,
        hnsw_ef_construction: 200,
        centroid_precision: rabitq_rs::mstg::ScalarPrecision::BF16,
        default_ef_search: 150,
        pruning_epsilon: 0.6,
    };

    println!("\nBuilding MSTG index...");
    println!("Configuration:");
    println!("  max_posting_size: {}", config.max_posting_size);
    println!("  branching_factor: {}", config.branching_factor);
    println!("  rabitq_bits: {}", config.rabitq_bits);
    println!("  centroid_precision: {:?}\n", config.centroid_precision);

    let index = MstgIndex::build(&data, config).expect("Failed to build index");

    println!("\nIndex built successfully!");
    println!("  Number of posting lists: {}", index.posting_lists.len());
    println!("  Number of centroids: {}", index.centroid_index.len());

    // Perform search
    println!("\nPerforming searches...");
    let n_queries = 10;
    let top_k = 10;

    let params = SearchParams::balanced(top_k);

    for (i, query) in data.iter().enumerate().take(n_queries) {
        let results = index.search(query, &params);

        println!("\nQuery {}: Found {} results", i, results.len());
        println!("  Top-3 results:");
        for (j, result) in results.iter().take(3).enumerate() {
            println!(
                "    {}. vector_id={}, distance={:.6}",
                j + 1,
                result.vector_id,
                result.distance
            );
        }

        // Verify that the query itself is in top results (should have very small distance)
        let has_self = results
            .iter()
            .any(|r| r.vector_id == i && r.distance < 0.01);
        if has_self {
            println!("  ✓ Query found itself with near-zero distance");
        }
    }

    println!("\n✓ MSTG demo complete!");
}
