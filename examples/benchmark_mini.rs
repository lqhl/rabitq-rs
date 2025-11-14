//! Mini benchmark for quick performance comparison
//!
//! Usage:
//!   cargo run --release --example benchmark_mini

use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams as MstgSearchParams};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use rand::prelude::*;
use std::time::Instant;

fn main() {
    println!("\n╔═══════════════════════════════════════════╗");
    println!("║  MSTG vs IVF Mini Benchmark (FastScan)   ║");
    println!("╚═══════════════════════════════════════════╝\n");

    // Small but representative dataset
    let n = 20_000;
    let dim = 128;
    let n_queries = 50;

    println!("Dataset: {} vectors × {} dims", n, dim);
    println!("Queries: {}\n", n_queries);

    // Generate data
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen()).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen()).collect())
        .collect();

    // ========== MSTG ==========
    println!("┌─ MSTG ─────────────────────────┐");

    let config = MstgConfig {
        max_posting_size: 800,
        branching_factor: 6,
        rabitq_bits: 7,
        faster_config: true,
        metric: Metric::L2,
        hnsw_m: 24,
        hnsw_ef_construction: 100,
        centroid_precision: rabitq_rs::mstg::ScalarPrecision::BF16,
        ..Default::default()
    };

    let start = Instant::now();
    let mstg_idx = MstgIndex::build(&data, config).expect("MSTG build");
    let mstg_build = start.elapsed();
    println!("│ Build:  {:?}", mstg_build);

    // Warmup
    for q in &queries[..5] {
        let params = MstgSearchParams {
            ef_search: 80,
            pruning_epsilon: 1.2,
            top_k: 10,
        };
        let _ = mstg_idx.search(q, &params);
    }

    // Search benchmark
    let params = MstgSearchParams {
        ef_search: 80,
        pruning_epsilon: 1.2,
        top_k: 10,
    };

    let start = Instant::now();
    for q in &queries {
        let _ = mstg_idx.search(q, &params);
    }
    let mstg_search = start.elapsed();
    let mstg_qps = n_queries as f64 / mstg_search.as_secs_f64();

    println!("│ Search: {:?}", mstg_search);
    println!("│ QPS:    {:.0}", mstg_qps);
    println!("└────────────────────────────────┘\n");

    // ========== IVF ==========
    println!("┌─ IVF ──────────────────────────┐");

    let nlist = ((n as f32).sqrt() as usize).max(50);

    let start = Instant::now();
    let ivf_idx = IvfRabitqIndex::train(
        &data,
        nlist,
        7,
        Metric::L2,
        RotatorType::FhtKacRotator,
        42,
        true,
    )
    .expect("IVF train");
    let ivf_build = start.elapsed();
    println!("│ Build:  {:?}", ivf_build);

    // Warmup
    let nprobe = nlist / 10;
    for q in &queries[..5] {
        let _ = ivf_idx.search(q, SearchParams { nprobe, top_k: 10 });
    }

    // Search benchmark
    let start = Instant::now();
    for q in &queries {
        let _ = ivf_idx.search(q, SearchParams { nprobe, top_k: 10 });
    }
    let ivf_search = start.elapsed();
    let ivf_qps = n_queries as f64 / ivf_search.as_secs_f64();

    println!("│ Search: {:?}", ivf_search);
    println!("│ QPS:    {:.0}", ivf_qps);
    println!("│ nprobe: {}", nprobe);
    println!("└────────────────────────────────┘\n");

    // ========== COMPARISON ==========
    println!("╔═══════════════════════════════════════════╗");
    println!("║              COMPARISON                   ║");
    println!("╚═══════════════════════════════════════════╝");

    let build_speedup = mstg_build.as_secs_f64() / ivf_build.as_secs_f64();
    let search_speedup = mstg_search.as_micros() as f64 / ivf_search.as_micros() as f64;

    println!("\nBuild Time:");
    if build_speedup > 1.0 {
        println!("  IVF is {:.2}x faster", build_speedup);
    } else {
        println!("  MSTG is {:.2}x faster", 1.0 / build_speedup);
    }

    println!("\nSearch Performance:");
    if search_speedup > 1.0 {
        println!("  IVF is {:.2}x faster", search_speedup);
        println!("  IVF: {:.0} QPS", ivf_qps);
        println!("  MSTG: {:.0} QPS", mstg_qps);
    } else {
        println!("  MSTG is {:.2}x faster", 1.0 / search_speedup);
        println!("  MSTG: {:.0} QPS", mstg_qps);
        println!("  IVF: {:.0} QPS", ivf_qps);
    }

    println!("\n✅ Note: MSTG's advantage increases with:");
    println!("   • Larger datasets (>100K vectors)");
    println!("   • Higher dimensions (>512 dims)");
    println!("   • Higher recall requirements (>90%)");
    println!();
}
