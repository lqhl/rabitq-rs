//! Benchmark comparing MSTG vs IVF performance
//!
//! Usage:
//!   cargo run --release --example benchmark_comparison

use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams as MstgSearchParams};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use rand::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

fn generate_test_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Compute recall@k: what fraction of true neighbors were retrieved
fn compute_recall(retrieved: &[usize], ground_truth: &[usize]) -> f32 {
    let gt_set: HashSet<_> = ground_truth.iter().copied().collect();
    let matches = retrieved.iter().filter(|id| gt_set.contains(id)).count();
    matches as f32 / ground_truth.len() as f32
}

/// Compute ground truth using brute force search
fn brute_force_search(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, vec)| {
            let dist: f32 = query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(i, _)| *i).collect()
}

struct BenchmarkResult {
    name: String,
    build_time_ms: f64,
    avg_search_time_us: f64,
    recall_at_10: f32,
    recall_at_100: f32,
    memory_mb: f32,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n{}", "=".repeat(60));
        println!("  {}", self.name);
        println!("{}", "=".repeat(60));
        println!("  Build time:        {:>10.2} ms", self.build_time_ms);
        println!("  Avg search time:   {:>10.2} µs", self.avg_search_time_us);
        println!("  Recall@10:         {:>10.2}%", self.recall_at_10 * 100.0);
        println!("  Recall@100:        {:>10.2}%", self.recall_at_100 * 100.0);
        println!("  Memory usage:      {:>10.2} MB", self.memory_mb);
        println!("{}", "=".repeat(60));
    }
}

fn benchmark_mstg(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth_10: &[Vec<usize>],
    ground_truth_100: &[Vec<usize>],
) -> BenchmarkResult {
    println!("\n[MSTG] Building index...");

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

    let start = Instant::now();
    let index = MstgIndex::build(data, config).expect("Failed to build MSTG index");
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("[MSTG] Running searches...");

    // Warmup
    for query in queries.iter().take(10) {
        let params = MstgSearchParams::balanced(100);
        let _ = index.search(query, &params);
    }

    // Benchmark search time
    let start = Instant::now();
    let mut all_results_10 = Vec::new();
    let mut all_results_100 = Vec::new();

    for query in queries.iter() {
        let params_10 = MstgSearchParams::balanced(10);
        let results_10 = index.search(query, &params_10);
        all_results_10.push(results_10.iter().map(|r| r.vector_id).collect::<Vec<_>>());

        let params_100 = MstgSearchParams::balanced(100);
        let results_100 = index.search(query, &params_100);
        all_results_100.push(results_100.iter().map(|r| r.vector_id).collect::<Vec<_>>());
    }

    let avg_search_time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / queries.len() as f64;

    // Compute recall
    let recall_10: f32 = all_results_10
        .iter()
        .zip(ground_truth_10.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt))
        .sum::<f32>()
        / queries.len() as f32;

    let recall_100: f32 = all_results_100
        .iter()
        .zip(ground_truth_100.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt))
        .sum::<f32>()
        / queries.len() as f32;

    // Estimate memory (rough approximation)
    let memory_mb = index
        .posting_lists
        .iter()
        .map(|p| p.memory_size())
        .sum::<usize>() as f32
        / (1024.0 * 1024.0);

    BenchmarkResult {
        name: "MSTG (Multi-Scale Tree Graph)".to_string(),
        build_time_ms,
        avg_search_time_us,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        memory_mb,
    }
}

fn benchmark_ivf(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth_10: &[Vec<usize>],
    ground_truth_100: &[Vec<usize>],
    nlist: usize,
) -> BenchmarkResult {
    println!("\n[IVF] Building index...");

    let start = Instant::now();
    let index = IvfRabitqIndex::train(
        data,
        nlist,
        7, // total_bits
        Metric::L2,
        RotatorType::FhtKacRotator, // Changed from MatrixRotator for better build performance
        42,   // seed
        true, // use_faster_config
    )
    .expect("Failed to build IVF index");
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("[IVF] Running searches...");

    // Determine nprobe for similar recall to MSTG
    let nprobe = (nlist / 10).max(10); // Use ~10% of clusters

    // Warmup
    for query in queries.iter().take(10) {
        let params = SearchParams { nprobe, top_k: 100 };
        let _ = index.search(query, params);
    }

    // Benchmark search time
    let start = Instant::now();
    let mut all_results_10 = Vec::new();
    let mut all_results_100 = Vec::new();

    for query in queries.iter() {
        let params_10 = SearchParams { nprobe, top_k: 10 };
        let results_10 = index.search(query, params_10).unwrap();
        all_results_10.push(results_10.iter().map(|r| r.id).collect::<Vec<_>>());

        let params_100 = SearchParams { nprobe, top_k: 100 };
        let results_100 = index.search(query, params_100).unwrap();
        all_results_100.push(results_100.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let avg_search_time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / queries.len() as f64;

    // Compute recall
    let recall_10: f32 = all_results_10
        .iter()
        .zip(ground_truth_10.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt))
        .sum::<f32>()
        / queries.len() as f32;

    let recall_100: f32 = all_results_100
        .iter()
        .zip(ground_truth_100.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt))
        .sum::<f32>()
        / queries.len() as f32;

    // Estimate memory (rough approximation)
    let memory_mb = 10.0; // Placeholder - IVF doesn't expose memory usage directly

    BenchmarkResult {
        name: format!("IVF (nlist={}, nprobe={})", nlist, nprobe),
        build_time_ms,
        avg_search_time_us,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        memory_mb,
    }
}

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║         MSTG vs IVF Benchmark Comparison                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Test parameters
    let n_base = 10_000;
    let n_queries = 100;
    let dim = 128;
    let nlist = 100;

    println!("\n📊 Benchmark Configuration:");
    println!("  • Base vectors:    {:>8}", n_base);
    println!("  • Query vectors:   {:>8}", n_queries);
    println!("  • Dimensions:      {:>8}", dim);
    println!("  • IVF clusters:    {:>8}", nlist);
    println!("  • MSTG max_size:   {:>8}", 100);

    // Generate test data
    println!("\n🔄 Generating test data...");
    let data = generate_test_data(n_base, dim, 42);
    let queries = generate_test_data(n_queries, dim, 123);

    // Compute ground truth
    println!("🔍 Computing ground truth (brute force)...");
    let mut ground_truth_10 = Vec::new();
    let mut ground_truth_100 = Vec::new();

    for (i, query) in queries.iter().enumerate() {
        if i % 20 == 0 {
            print!("  Progress: {}/{}\r", i, n_queries);
        }
        ground_truth_10.push(brute_force_search(&data, query, 10));
        ground_truth_100.push(brute_force_search(&data, query, 100));
    }
    println!("  Progress: {}/{}  ✓", n_queries, n_queries);

    // Run benchmarks
    let mstg_result = benchmark_mstg(&data, &queries, &ground_truth_10, &ground_truth_100);
    let ivf_result = benchmark_ivf(&data, &queries, &ground_truth_10, &ground_truth_100, nlist);

    // Print results
    println!("\n\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                       ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    mstg_result.print();
    ivf_result.print();

    // Comparison
    println!("\n\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARISON SUMMARY                      ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    let build_speedup = ivf_result.build_time_ms / mstg_result.build_time_ms;
    let search_speedup = ivf_result.avg_search_time_us / mstg_result.avg_search_time_us;
    let recall_10_diff = (mstg_result.recall_at_10 - ivf_result.recall_at_10) * 100.0;
    let recall_100_diff = (mstg_result.recall_at_100 - ivf_result.recall_at_100) * 100.0;

    println!("\n  Build Time:");
    if build_speedup > 1.0 {
        println!("    MSTG is {:.2}x FASTER than IVF", build_speedup);
    } else {
        println!("    IVF is {:.2}x FASTER than MSTG", 1.0 / build_speedup);
    }

    println!("\n  Search Time:");
    if search_speedup > 1.0 {
        println!("    MSTG is {:.2}x FASTER than IVF", search_speedup);
    } else {
        println!("    IVF is {:.2}x FASTER than MSTG", 1.0 / search_speedup);
    }

    println!("\n  Recall@10:");
    if recall_10_diff > 0.0 {
        println!("    MSTG has {:.2}% HIGHER recall", recall_10_diff);
    } else {
        println!("    IVF has {:.2}% HIGHER recall", -recall_10_diff);
    }

    println!("\n  Recall@100:");
    if recall_100_diff > 0.0 {
        println!("    MSTG has {:.2}% HIGHER recall", recall_100_diff);
    } else {
        println!("    IVF has {:.2}% HIGHER recall", -recall_100_diff);
    }

    println!("\n{}", "=".repeat(60));
    println!("\n✅ Benchmark complete!\n");
}
