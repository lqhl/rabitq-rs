//! MSTG Benchmark on GIST 1M Dataset
//!
//! This benchmark evaluates MSTG performance on the GIST dataset,
//! measuring recall, QPS, and latency across different search parameters.

use rabitq_rs::io::{read_fvecs, read_ivecs};
use rabitq_rs::mstg::{MstgConfig, MstgIndex, ScalarPrecision, SearchParams};
use rabitq_rs::Metric;
use std::time::Instant;

const DATA_DIR: &str = "data/gist";

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║          MSTG Benchmark on GIST 1M Dataset               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Load GIST dataset
    println!("Loading GIST dataset...");
    let base_path = format!("{}/gist_base.fvecs", DATA_DIR);
    let query_path = format!("{}/gist_query.fvecs", DATA_DIR);
    let gt_path = format!("{}/gist_groundtruth.ivecs", DATA_DIR);

    let base_vectors = read_fvecs(&base_path, None).expect("Failed to read base vectors");
    let query_vectors = read_fvecs(&query_path, None).expect("Failed to read query vectors");
    let ground_truth = read_ivecs(&gt_path, None).expect("Failed to read ground truth");

    let n_base = base_vectors.len();
    let n_queries = query_vectors.len();
    let dim = base_vectors[0].len();

    println!("✓ Dataset loaded:");
    println!("  Base vectors: {}", n_base);
    println!("  Query vectors: {}", n_queries);
    println!("  Dimension: {}\n", dim);

    // Build MSTG index
    println!("Building MSTG index...");
    println!("Configuration (matching recall_qps_sweep for fair comparison):");
    let config = MstgConfig {
        max_posting_size: 256, // Same as recall_qps_sweep
        branching_factor: 5,   // Same as recall_qps_sweep
        balance_weight: 1.0,
        closure_epsilon: 0.15,
        max_replicas: 8,
        rabitq_bits: 7,
        faster_config: true, // Use faster quantization
        metric: Metric::L2,
        hnsw_m: 32,
        hnsw_ef_construction: 200,
        centroid_precision: ScalarPrecision::BF16, // 50% memory savings
        default_ef_search: 150,
        pruning_epsilon: 0.6,
    };

    println!("  max_posting_size: {}", config.max_posting_size);
    println!("  branching_factor: {}", config.branching_factor);
    println!("  closure_epsilon: {}", config.closure_epsilon);
    println!("  rabitq_bits: {}", config.rabitq_bits);
    println!("  centroid_precision: {:?}", config.centroid_precision);
    println!("  hnsw_m: {}", config.hnsw_m);
    println!();

    let build_start = Instant::now();
    let index =
        MstgIndex::build(&base_vectors, config.clone()).expect("Failed to build MSTG index");
    let build_time = build_start.elapsed();

    println!("\n✓ Index built successfully!");
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    println!("  Number of posting lists: {}", index.posting_lists.len());
    println!("  Number of centroids: {}", index.centroid_index.len());

    // Calculate memory usage
    let centroid_mem = index.centroid_index.memory_usage() as f64 / 1024.0 / 1024.0;
    let posting_mem: f64 = index
        .posting_lists
        .iter()
        .map(|p| p.memory_size() as f64)
        .sum::<f64>()
        / 1024.0
        / 1024.0;
    let total_mem = centroid_mem + posting_mem;

    println!("\nMemory Usage:");
    println!("  Centroid index: {:.2} MB", centroid_mem);
    println!("  Posting lists: {:.2} MB", posting_mem);
    println!("  Total: {:.2} MB", total_mem);
    println!(
        "  Compression ratio: {:.1}x",
        (n_base * dim * 4) as f64 / 1024.0 / 1024.0 / total_mem
    );

    // Benchmark: Sweep different search parameters
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Search Performance Evaluation               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<20} {:<12} {:<12} {:<12} {:<12}",
        "Config", "Recall@1", "Recall@10", "Recall@100", "QPS"
    );
    println!("{}", "─".repeat(68));

    let test_configs = vec![
        ("High Recall", SearchParams::high_recall(100)),
        ("Balanced", SearchParams::balanced(100)),
        ("Low Latency", SearchParams::low_latency(100)),
    ];

    for (name, params) in test_configs {
        let recall_results = evaluate_recall(&index, &query_vectors, &ground_truth, &params);
        println!(
            "{:<20} {:<12.2} {:<12.2} {:<12.2} {:<12.0}",
            name,
            recall_results.recall_at_1 * 100.0,
            recall_results.recall_at_10 * 100.0,
            recall_results.recall_at_100 * 100.0,
            recall_results.qps
        );
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                Parameter Sensitivity Sweep                ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Sweeping ef_search values...\n");
    println!(
        "{:<15} {:<12} {:<12} {:<12} {:<12}",
        "ef_search", "Recall@1", "Recall@10", "Avg Latency", "QPS"
    );
    println!("{}", "─".repeat(63));

    let ef_values = vec![50, 100, 150, 200, 300, 500];
    for &ef in &ef_values {
        let params = SearchParams::new(ef, 0.6, 100);
        let recall_results = evaluate_recall(&index, &query_vectors, &ground_truth, &params);
        println!(
            "{:<15} {:<12.2} {:<12.2} {:<12.2} {:<12.0}",
            ef,
            recall_results.recall_at_1 * 100.0,
            recall_results.recall_at_10 * 100.0,
            recall_results.avg_latency_ms,
            recall_results.qps
        );
    }

    println!("\n✓ Benchmark complete!");
}

struct RecallResults {
    recall_at_1: f32,
    recall_at_10: f32,
    recall_at_100: f32,
    qps: f32,
    avg_latency_ms: f64,
}

fn evaluate_recall(
    index: &MstgIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
    params: &SearchParams,
) -> RecallResults {
    let n_queries = queries.len().min(ground_truth.len());

    let mut total_recall_1 = 0.0;
    let mut total_recall_10 = 0.0;
    let mut total_recall_100 = 0.0;

    let start = Instant::now();

    for i in 0..n_queries {
        let results = index.search(&queries[i], params);
        let gt = &ground_truth[i];

        // Calculate recall@1
        if !results.is_empty() && !gt.is_empty() && results[0].vector_id == gt[0] as usize {
            total_recall_1 += 1.0;
        }

        // Calculate recall@10
        let gt_10: std::collections::HashSet<_> = gt.iter().take(10).map(|&x| x as usize).collect();
        let found_10 = results
            .iter()
            .take(10)
            .filter(|r| gt_10.contains(&r.vector_id))
            .count();
        total_recall_10 += found_10 as f32 / 10.0;

        // Calculate recall@100
        let gt_100: std::collections::HashSet<_> =
            gt.iter().take(100).map(|&x| x as usize).collect();
        let found_100 = results
            .iter()
            .take(100)
            .filter(|r| gt_100.contains(&r.vector_id))
            .count();
        total_recall_100 += found_100 as f32 / 100.0;
    }

    let elapsed = start.elapsed();
    let qps = n_queries as f32 / elapsed.as_secs_f32();
    let avg_latency_ms = elapsed.as_secs_f64() * 1000.0 / n_queries as f64;

    RecallResults {
        recall_at_1: total_recall_1 / n_queries as f32,
        recall_at_10: total_recall_10 / n_queries as f32,
        recall_at_100: total_recall_100 / n_queries as f32,
        qps,
        avg_latency_ms,
    }
}
