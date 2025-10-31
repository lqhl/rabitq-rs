//! MSTG vs IVF Benchmark on GIST Dataset
//!
//! Usage:
//!   # Quick test with 10K vectors
//!   cargo run --release --example benchmark_gist -- --max-base 10000
//!
//!   # Medium test with 100K vectors
//!   cargo run --release --example benchmark_gist -- --max-base 100000
//!
//!   # Full GIST benchmark (1M vectors, takes ~10-20 minutes)
//!   cargo run --release --example benchmark_gist

use rabitq_rs::io::{read_fvecs, read_ivecs};
use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams as MstgSearchParams};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use std::collections::HashSet;
use std::env;
use std::time::Instant;

struct BenchmarkResult {
    name: String,
    build_time_ms: f64,
    avg_search_time_us: f64,
    recall_at_1: f32,
    recall_at_10: f32,
    recall_at_100: f32,
    qps: f32,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n{}", "=".repeat(70));
        println!("  {}", self.name);
        println!("{}", "=".repeat(70));
        println!("  Build time:        {:>12.2} ms", self.build_time_ms);
        println!("  Avg search time:   {:>12.2} µs", self.avg_search_time_us);
        println!("  QPS (queries/sec): {:>12.2}", self.qps);
        println!("  Recall@1:          {:>12.2}%", self.recall_at_1 * 100.0);
        println!("  Recall@10:         {:>12.2}%", self.recall_at_10 * 100.0);
        println!("  Recall@100:        {:>12.2}%", self.recall_at_100 * 100.0);
        println!("{}", "=".repeat(70));
    }
}

/// Compute recall@k: what fraction of true neighbors were retrieved
fn compute_recall(retrieved: &[usize], ground_truth: &[i32], k: usize) -> f32 {
    let gt_set: HashSet<_> = ground_truth.iter().take(k).map(|&id| id as usize).collect();
    let matches = retrieved
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id))
        .count();
    matches as f32 / k.min(ground_truth.len()) as f32
}

fn benchmark_mstg(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
) -> BenchmarkResult {
    println!("\n[MSTG] Building index...");

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

    let start = Instant::now();
    let index = MstgIndex::build(data, config).expect("Failed to build MSTG index");
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "[MSTG] Index built: {} posting lists",
        index.posting_lists.len()
    );
    println!("[MSTG] Running searches...");

    // Warmup
    for query in queries.iter().take(10) {
        let params = MstgSearchParams::balanced(100);
        let _ = index.search(query, &params);
    }

    // Benchmark search
    let start = Instant::now();
    let mut all_results = Vec::new();

    for query in queries.iter() {
        let params = MstgSearchParams::balanced(100);
        let results = index.search(query, &params);
        all_results.push(results.iter().map(|r| r.vector_id).collect::<Vec<_>>());
    }

    let total_time = start.elapsed();
    let avg_search_time_us = total_time.as_secs_f64() * 1_000_000.0 / queries.len() as f64;
    let qps = queries.len() as f32 / total_time.as_secs_f32();

    // Compute recall at different k values
    let recall_1: f32 = all_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt, 1))
        .sum::<f32>()
        / queries.len() as f32;

    let recall_10: f32 = all_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt, 10))
        .sum::<f32>()
        / queries.len() as f32;

    let recall_100: f32 = all_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt, 100))
        .sum::<f32>()
        / queries.len() as f32;

    BenchmarkResult {
        name: "MSTG (Multi-Scale Tree Graph)".to_string(),
        build_time_ms,
        avg_search_time_us,
        recall_at_1: recall_1,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        qps,
    }
}

fn benchmark_ivf(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
    nlist: usize,
    nprobe: usize,
) -> BenchmarkResult {
    println!("\n[IVF] Building index...");

    let start = Instant::now();
    let index = IvfRabitqIndex::train(
        data,
        nlist,
        7, // total_bits
        Metric::L2,
        RotatorType::FhtKacRotator, // Changed from MatrixRotator for better build performance
        42,                         // seed
        true,                       // use_faster_config
    )
    .expect("Failed to build IVF index");
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("[IVF] Index built: {} clusters", nlist);
    println!("[IVF] Running searches with nprobe={}...", nprobe);

    // Warmup
    for query in queries.iter().take(10) {
        let params = SearchParams { nprobe, top_k: 100 };
        let _ = index.search(query, params);
    }

    // Benchmark search
    let start = Instant::now();
    let mut all_results = Vec::new();

    for query in queries.iter() {
        let params = SearchParams { nprobe, top_k: 100 };
        let results = index.search(query, params).unwrap();
        all_results.push(results.iter().map(|r| r.id).collect::<Vec<_>>());
    }

    let total_time = start.elapsed();
    let avg_search_time_us = total_time.as_secs_f64() * 1_000_000.0 / queries.len() as f64;
    let qps = queries.len() as f32 / total_time.as_secs_f32();

    // Compute recall
    let recall_1: f32 = all_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt, 1))
        .sum::<f32>()
        / queries.len() as f32;

    let recall_10: f32 = all_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt, 10))
        .sum::<f32>()
        / queries.len() as f32;

    let recall_100: f32 = all_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(retrieved, gt)| compute_recall(retrieved, gt, 100))
        .sum::<f32>()
        / queries.len() as f32;

    BenchmarkResult {
        name: format!("IVF (nlist={}, nprobe={})", nlist, nprobe),
        build_time_ms,
        avg_search_time_us,
        recall_at_1: recall_1,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
        qps,
    }
}

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║       MSTG vs IVF Benchmark - GIST Dataset (960D)             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Parse command line args
    let args: Vec<String> = env::args().collect();
    let max_base = if args.len() > 2 && args[1] == "--max-base" {
        args[2].parse().unwrap_or(1_000_000)
    } else {
        1_000_000 // Default to full 1M dataset
    };

    println!("\n📂 Loading GIST dataset...");
    println!("  Base vectors limit: {}", max_base);

    // Load data
    let data = read_fvecs("data/gist/gist_base.fvecs", Some(max_base))
        .expect("Failed to load base vectors");
    let queries =
        read_fvecs("data/gist/gist_query.fvecs", None).expect("Failed to load query vectors");
    let ground_truth =
        read_ivecs("data/gist/gist_groundtruth.ivecs", None).expect("Failed to load ground truth");

    let dim = data[0].len();
    println!("\n📊 Dataset Statistics:");
    println!("  • Base vectors:    {:>8}", data.len());
    println!("  • Query vectors:   {:>8}", queries.len());
    println!("  • Dimensions:      {:>8}", dim);
    println!("  • Ground truth k:  {:>8}", ground_truth[0].len());

    // Determine nlist based on dataset size (sqrt heuristic)
    let nlist = (data.len() as f64).sqrt() as usize;
    let nprobe = (nlist / 16).max(8); // Use ~6% of clusters (more practical for 1M scale)

    println!("\n⚙️  Index Configuration:");
    println!("  • IVF nlist:       {:>8}", nlist);
    println!("  • IVF nprobe:      {:>8}", nprobe);
    println!("  • MSTG max_size:   {:>8}", 256);

    // Run benchmarks
    let mstg_result = benchmark_mstg(&data, &queries, &ground_truth);
    let ivf_result = benchmark_ivf(&data, &queries, &ground_truth, nlist, nprobe);

    // Print results
    println!("\n\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                       BENCHMARK RESULTS                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    mstg_result.print();
    ivf_result.print();

    // Comparison
    println!("\n\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                      COMPARISON SUMMARY                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let build_speedup = ivf_result.build_time_ms / mstg_result.build_time_ms;
    let search_speedup = ivf_result.avg_search_time_us / mstg_result.avg_search_time_us;
    let qps_ratio = mstg_result.qps / ivf_result.qps;

    println!("  ⏱️  Build Time:");
    if build_speedup > 1.0 {
        println!("      → MSTG is {:.2}x FASTER", build_speedup);
    } else {
        println!("      → IVF is {:.2}x FASTER", 1.0 / build_speedup);
    }

    println!("\n  🚀 Search Speed:");
    if search_speedup > 1.0 {
        println!("      → MSTG is {:.2}x FASTER", search_speedup);
        println!(
            "      → MSTG: {:.0} QPS vs IVF: {:.0} QPS",
            mstg_result.qps, ivf_result.qps
        );
    } else {
        println!("      → IVF is {:.2}x FASTER", 1.0 / search_speedup);
        println!(
            "      → IVF: {:.0} QPS vs MSTG: {:.0} QPS",
            ivf_result.qps, mstg_result.qps
        );
    }

    println!("\n  🎯 Recall Quality:");
    let recall_1_diff = (mstg_result.recall_at_1 - ivf_result.recall_at_1) * 100.0;
    let recall_10_diff = (mstg_result.recall_at_10 - ivf_result.recall_at_10) * 100.0;
    let recall_100_diff = (mstg_result.recall_at_100 - ivf_result.recall_at_100) * 100.0;

    print!("      → Recall@1:   ");
    if recall_1_diff > 0.0 {
        println!("MSTG +{:.2}%", recall_1_diff);
    } else {
        println!("IVF +{:.2}%", -recall_1_diff);
    }

    print!("      → Recall@10:  ");
    if recall_10_diff > 0.0 {
        println!("MSTG +{:.2}%", recall_10_diff);
    } else {
        println!("IVF +{:.2}%", -recall_10_diff);
    }

    print!("      → Recall@100: ");
    if recall_100_diff > 0.0 {
        println!("MSTG +{:.2}%", recall_100_diff);
    } else {
        println!("IVF +{:.2}%", -recall_100_diff);
    }

    // Recommendation
    println!("\n  💡 Recommendation:");
    if mstg_result.recall_at_10 > ivf_result.recall_at_10 * 1.2 && qps_ratio > 0.5 {
        println!("      → Use MSTG for better accuracy at acceptable speed");
    } else if ivf_result.qps > mstg_result.qps * 2.0
        && (ivf_result.recall_at_10 - mstg_result.recall_at_10).abs() < 0.1
    {
        println!("      → Use IVF for maximum speed with similar accuracy");
    } else {
        println!("      → Trade-off depends on your speed vs accuracy requirements");
    }

    println!("\n{}", "=".repeat(70));
    println!("\n✅ Benchmark complete!\n");
}
