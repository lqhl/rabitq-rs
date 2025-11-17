//! Tiny profiling benchmark for MSTG search hotspots
//!
//! Usage:
//!   cargo run --release --example profile_mstg_search_tiny

use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams};
use rabitq_rs::Metric;
use rand::prelude::*;
use std::time::Instant;

fn generate_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen()).collect())
        .collect()
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║       MSTG Search Hotspot Analysis (Tiny Version)        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Very small: 10K vectors, 960 dims
    let n_base = 10_000;
    let dim = 960;
    let n_queries = 50;

    println!("Configuration:");
    println!("  Vectors:     {}", n_base);
    println!("  Dimensions:  {}", dim);
    println!("  Queries:     {}\n", n_queries);

    // Generate data
    println!("🔄 Generating data...");
    let data = generate_data(n_base, dim, 42);
    let queries = generate_data(n_queries, dim, 123);
    println!("✓ Data generated\n");

    // Build index with GIST-like parameters
    println!("🔨 Building MSTG index...");
    let config = MstgConfig {
        max_posting_size: 256,
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

    let build_start = Instant::now();
    let index = MstgIndex::build(&data, config).expect("Build failed");
    let build_time = build_start.elapsed();

    println!("✓ Index built in {:.2}s", build_time.as_secs_f64());
    println!("  Posting lists: {}", index.posting_lists.len());
    println!(
        "  Avg list size: {:.1}",
        n_base as f64 / index.posting_lists.len() as f64
    );
    println!();

    // Warmup
    println!("🔥 Warming up (5 queries)...");
    let params = SearchParams::new(400, 2.0, 100);
    for q in &queries[..5] {
        let _ = index.search(q, &params);
    }
    println!("✓ Warmup complete\n");

    // Profile search
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Search Performance Profiling                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test_configs = vec![
        ("High Recall (ef=400)", SearchParams::new(400, 2.0, 100)),
        ("Balanced (ef=200)", SearchParams::new(200, 1.0, 100)),
        ("Fast (ef=100)", SearchParams::new(100, 0.5, 100)),
    ];

    println!(
        "{:<25} {:>12} {:>12} {:>12}",
        "Config", "Avg µs", "Min µs", "Max µs"
    );
    println!("{}", "─".repeat(65));

    for (name, params) in test_configs {
        let mut times = Vec::new();

        for q in &queries {
            let start = Instant::now();
            let _ = index.search(q, &params);
            times.push(start.elapsed().as_micros());
        }

        let avg = times.iter().sum::<u128>() / times.len() as u128;
        let min = *times.iter().min().unwrap();
        let max = *times.iter().max().unwrap();

        println!("{:<25} {:>12} {:>12} {:>12}", name, avg, min, max);
    }

    println!("\n✅ Tiny benchmark complete!");
}
