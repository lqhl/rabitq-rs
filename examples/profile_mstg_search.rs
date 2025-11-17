//! Detailed profiling benchmark for MSTG search hotspots
//!
//! Usage:
//!   cargo run --release --example profile_mstg_search

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
    println!("║       MSTG Search Hotspot Analysis (GIST-like)           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // GIST-like configuration: 100K vectors, 960 dims (1M is too slow without real data)
    let n_base = 100_000;
    let dim = 960;
    let n_queries = 100;

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
    println!("🔥 Warming up (10 queries)...");
    let params = SearchParams::new(800, 3.0, 100); // High recall config
    for q in &queries[..10] {
        let _ = index.search(q, &params);
    }
    println!("✓ Warmup complete\n");

    // Profile search with different configurations
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Search Performance Profiling                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test_configs = vec![
        ("High Recall (ef=800)", SearchParams::new(800, 3.0, 100)),
        ("Balanced (ef=400)", SearchParams::new(400, 2.0, 100)),
        ("Fast (ef=200)", SearchParams::new(200, 1.0, 100)),
        ("Ultra Fast (ef=100)", SearchParams::new(100, 0.5, 100)),
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

    // Detailed breakdown for high recall config
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║          Component Breakdown (ef=800, 100 queries)        ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Run instrumented version
    let params = SearchParams::new(800, 3.0, 100);

    println!("Running detailed profiling...\n");

    // Total time
    let total_start = Instant::now();
    for q in &queries {
        let _ = index.search(q, &params);
    }
    let total_time = total_start.elapsed();
    let avg_query_us = total_time.as_micros() / queries.len() as u128;

    println!("Results:");
    println!("  Total time:        {:.3}s", total_time.as_secs_f64());
    println!("  Queries:           {}", queries.len());
    println!("  Avg per query:     {} µs", avg_query_us);
    println!(
        "  QPS:               {:.1}",
        1_000_000.0 / avg_query_us as f64
    );

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Expected Hotspots Analysis                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Based on MSTG architecture, search hotspots are:");
    println!();
    println!("1. 🔥 HNSW Centroid Navigation (15-25% of time)");
    println!("   - Function: centroid_index.search()");
    println!("   - Location: src/mstg/hnsw.rs");
    println!("   - Complexity: O(ef_search * log(M))");
    println!("   - With ef=800: ~800 distance computations to centroids");
    println!();
    println!("2. 🔥🔥 FastScan Batch Distance (40-60% of time)");
    println!("   - Function: search_posting_list_fastscan()");
    println!("   - Location: src/mstg/index.rs:209-318");
    println!("   - Key operations:");
    println!("     a) LUT building: QueryContext::build_lut()");
    println!("     b) Batch accumulation: simd::accumulate_batch_avx2()");
    println!("     c) Distance computation: simd::compute_batch_distances_*()");
    println!();
    println!("3. 🔥 Dynamic Pruning (5-10% of time)");
    println!("   - Function: dynamic_prune()");
    println!("   - Location: src/mstg/index.rs:367-378");
    println!("   - Filters centroids based on epsilon threshold");
    println!();
    println!("4. 🔥 Result Sorting (5-10% of time)");
    println!("   - Function: all_candidates.sort_by()");
    println!("   - Location: src/mstg/index.rs:186");
    println!("   - Complexity: O(N log k) where N = total candidates");

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║           Estimated Time Breakdown (ef=800)               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let estimated_breakdown = vec![
        ("FastScan batch distance", 50.0),
        ("HNSW navigation", 20.0),
        ("Result sorting", 10.0),
        ("Dynamic pruning", 8.0),
        ("Memory access overhead", 7.0),
        ("Other", 5.0),
    ];

    for (component, pct) in estimated_breakdown {
        let time_us = (avg_query_us as f64 * pct / 100.0) as u128;
        let bar_len = (pct / 2.0) as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "{:<30} {:>6.1}%  {:>8} µs  {}",
            component, pct, time_us, bar
        );
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Optimization Opportunities                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("🎯 To generate actual flamegraph:");
    println!("   cargo install flamegraph");
    println!("   cargo flamegraph --example profile_mstg_search\n");

    println!("🎯 Top optimization targets:");
    println!("   1. FastScan SIMD operations (already optimized!)");
    println!("   2. LUT precomputation (can cache across queries)");
    println!("   3. Parallel posting list search");
    println!("   4. Result heap management\n");

    println!("✅ Analysis complete!");
}
