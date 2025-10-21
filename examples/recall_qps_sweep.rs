//! Recall vs QPS Sweep: Compare MSTG and IVF at different recall levels
//!
//! Usage:
//!   # Run on subset
//!   cargo run --release --example recall_qps_sweep -- --max-base 100000
//!
//!   # Run on full 1M dataset
//!   cargo run --release --example recall_qps_sweep

use rabitq_rs::io::{read_fvecs, read_ivecs};
use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams as MstgSearchParams};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use std::collections::HashSet;
use std::env;
use std::time::Instant;

#[derive(Debug, Clone)]
struct SweepPoint {
    method: String,
    config: String,
    recall_at_100: f32,
    avg_latency_ms: f64,
    qps: f32,
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

fn sweep_mstg(
    index: &MstgIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
) -> Vec<SweepPoint> {
    println!("\n🔍 MSTG Parameter Sweep...");

    let ef_search_values = vec![50, 100, 200, 400, 800, 1600, 3200];
    let pruning_epsilons = vec![0.3, 0.5, 0.8, 1.2, 2.0, 3.0];

    let mut results = Vec::new();

    for &ef_search in &ef_search_values {
        for &epsilon in &pruning_epsilons {
            let params = MstgSearchParams::new(ef_search, epsilon, 100);

            // Warmup
            for query in queries.iter().take(5) {
                let _ = index.search(query, &params);
            }

            // Benchmark
            let start = Instant::now();
            let mut all_results = Vec::new();

            for query in queries.iter() {
                let results = index.search(query, &params);
                all_results.push(results.iter().map(|r| r.vector_id).collect::<Vec<_>>());
            }

            let total_time = start.elapsed();
            let avg_latency_ms = total_time.as_secs_f64() * 1000.0 / queries.len() as f64;
            let qps = queries.len() as f32 / total_time.as_secs_f32();

            // Compute recall@100
            let recall_100: f32 = all_results
                .iter()
                .zip(ground_truth.iter())
                .map(|(retrieved, gt)| compute_recall(retrieved, gt, 100))
                .sum::<f32>()
                / queries.len() as f32;

            let point = SweepPoint {
                method: "MSTG".to_string(),
                config: format!("ef={}, eps={:.1}", ef_search, epsilon),
                recall_at_100: recall_100,
                avg_latency_ms,
                qps,
            };

            println!(
                "  ef={:4}, eps={:.1}: recall={:.1}%, latency={:.2}ms, qps={:.0}",
                ef_search, epsilon, recall_100 * 100.0, avg_latency_ms, qps
            );

            results.push(point);
        }
    }

    results
}

fn sweep_ivf(
    index: &IvfRabitqIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
    nlist: usize,
) -> Vec<SweepPoint> {
    println!("\n🔍 IVF Parameter Sweep...");

    let nprobe_values = vec![4, 8, 16, 32, 64, 128, 256, 512];

    let mut results = Vec::new();

    for &nprobe in &nprobe_values {
        if nprobe > nlist {
            continue;
        }

        let params = SearchParams { nprobe, top_k: 100 };

        // Warmup
        for query in queries.iter().take(5) {
            let _ = index.search(query, params);
        }

        // Benchmark
        let start = Instant::now();
        let mut all_results = Vec::new();

        for query in queries.iter() {
            let results = index.search(query, params).unwrap();
            all_results.push(results.iter().map(|r| r.id).collect::<Vec<_>>());
        }

        let total_time = start.elapsed();
        let avg_latency_ms = total_time.as_secs_f64() * 1000.0 / queries.len() as f64;
        let qps = queries.len() as f32 / total_time.as_secs_f32();

        // Compute recall@100
        let recall_100: f32 = all_results
            .iter()
            .zip(ground_truth.iter())
            .map(|(retrieved, gt)| compute_recall(retrieved, gt, 100))
            .sum::<f32>()
            / queries.len() as f32;

        let point = SweepPoint {
            method: "IVF".to_string(),
            config: format!("nprobe={}", nprobe),
            recall_at_100: recall_100,
            avg_latency_ms,
            qps,
        };

        println!(
            "  nprobe={:4}: recall={:.1}%, latency={:.2}ms, qps={:.0}",
            nprobe, recall_100 * 100.0, avg_latency_ms, qps
        );

        results.push(point);
    }

    results
}

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         Recall vs QPS Sweep: MSTG vs IVF                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Parse command line args
    let args: Vec<String> = env::args().collect();
    let max_base = if args.len() > 2 && args[1] == "--max-base" {
        args[2].parse().unwrap_or(1_000_000)
    } else {
        1_000_000
    };

    println!("\n📂 Loading GIST dataset...");
    println!("  Base vectors limit: {}", max_base);

    // Load data
    let data = read_fvecs("data/gist/gist_base.fvecs", Some(max_base))
        .expect("Failed to load base vectors");
    let queries =
        read_fvecs("data/gist/gist_query.fvecs", None).expect("Failed to load query vectors");
    let ground_truth = read_ivecs("data/gist/gist_groundtruth.ivecs", None)
        .expect("Failed to load ground truth");

    let dim = data[0].len();
    println!("\n📊 Dataset Statistics:");
    println!("  • Base vectors:    {:>8}", data.len());
    println!("  • Query vectors:   {:>8}", queries.len());
    println!("  • Dimensions:      {:>8}", dim);

    // Build MSTG index
    println!("\n🏗️  Building MSTG index...");
    let mstg_config = MstgConfig {
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

    let mstg_start = Instant::now();
    let mstg_index = MstgIndex::build(&data, mstg_config).expect("Failed to build MSTG index");
    let mstg_build_time = mstg_start.elapsed();
    println!("  ✓ MSTG built in {:.2}s", mstg_build_time.as_secs_f64());

    // Build IVF index
    println!("\n🏗️  Building IVF index...");
    let nlist = (data.len() as f64).sqrt() as usize;

    let ivf_start = Instant::now();
    let ivf_index = IvfRabitqIndex::train(
        &data,
        nlist,
        7,
        Metric::L2,
        RotatorType::MatrixRotator,
        42,
        true,
    )
    .expect("Failed to build IVF index");
    let ivf_build_time = ivf_start.elapsed();
    println!("  ✓ IVF built in {:.2}s (nlist={})", ivf_build_time.as_secs_f64(), nlist);

    // Run sweeps
    let mstg_points = sweep_mstg(&mstg_index, &queries, &ground_truth);
    let ivf_points = sweep_ivf(&ivf_index, &queries, &ground_truth, nlist);

    // Print summary table
    println!("\n\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                     SWEEP RESULTS SUMMARY                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("MSTG Results:");
    println!("{:30} {:>10} {:>12} {:>10}", "Config", "Recall@100", "Latency(ms)", "QPS");
    println!("{}", "-".repeat(70));
    for point in &mstg_points {
        println!(
            "{:30} {:>9.1}% {:>12.2} {:>10.0}",
            point.config, point.recall_at_100 * 100.0, point.avg_latency_ms, point.qps
        );
    }

    println!("\nIVF Results:");
    println!("{:30} {:>10} {:>12} {:>10}", "Config", "Recall@100", "Latency(ms)", "QPS");
    println!("{}", "-".repeat(70));
    for point in &ivf_points {
        println!(
            "{:30} {:>9.1}% {:>12.2} {:>10.0}",
            point.config, point.recall_at_100 * 100.0, point.avg_latency_ms, point.qps
        );
    }

    // Generate CSV for plotting
    println!("\n📊 Generating CSV data for plotting...");
    let csv_path = "recall_qps_results.csv";
    let mut csv_content = String::from("method,config,recall_at_100,latency_ms,qps\n");

    for point in &mstg_points {
        csv_content.push_str(&format!(
            "{},{},{},{},{}\n",
            point.method, point.config, point.recall_at_100, point.avg_latency_ms, point.qps
        ));
    }

    for point in &ivf_points {
        csv_content.push_str(&format!(
            "{},{},{},{},{}\n",
            point.method, point.config, point.recall_at_100, point.avg_latency_ms, point.qps
        ));
    }

    std::fs::write(csv_path, csv_content).expect("Failed to write CSV");
    println!("  ✓ Results saved to {}", csv_path);

    // Find comparable points
    println!("\n\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              COMPARISON AT SIMILAR RECALL LEVELS               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let target_recalls = vec![0.70, 0.80, 0.90, 0.95];

    for &target in &target_recalls {
        // Find closest MSTG point
        let mstg_closest = mstg_points
            .iter()
            .min_by_key(|p| ((p.recall_at_100 - target).abs() * 1000.0) as i32);

        // Find closest IVF point
        let ivf_closest = ivf_points
            .iter()
            .min_by_key(|p| ((p.recall_at_100 - target).abs() * 1000.0) as i32);

        if let (Some(mstg), Some(ivf)) = (mstg_closest, ivf_closest) {
            println!("Target Recall@100: {:.0}%", target * 100.0);
            println!(
                "  MSTG: {} - recall={:.1}%, qps={:.0}",
                mstg.config,
                mstg.recall_at_100 * 100.0,
                mstg.qps
            );
            println!(
                "  IVF:  {} - recall={:.1}%, qps={:.0}",
                ivf.config,
                ivf.recall_at_100 * 100.0,
                ivf.qps
            );

            let speedup = mstg.qps / ivf.qps;
            if speedup > 1.0 {
                println!("  → MSTG is {:.1}x faster at similar recall\n", speedup);
            } else {
                println!("  → IVF is {:.1}x faster at similar recall\n", 1.0 / speedup);
            }
        }
    }

    println!("\n✅ Sweep complete! Use recall_qps_results.csv for plotting.\n");
}
