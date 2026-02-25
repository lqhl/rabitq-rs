//! IVF vs MSTG benchmark
//!
//! Supports two modes:
//!   1. Real dataset from VectorDBBench Cohere 1M-768d (Parquet files)
//!   2. Fallback to random data when dataset is not available
//!
//! Dataset setup:
//!   mkdir -p dataset/cohere_1m_768d
//!   curl -L -o dataset/cohere_1m_768d/train.parquet \
//!     "https://assets.zilliz.com/benchmark/cohere_medium_1m/train.parquet"
//!   curl -L -o dataset/cohere_1m_768d/test.parquet \
//!     "https://assets.zilliz.com/benchmark/cohere_medium_1m/test.parquet"
//!   curl -L -o dataset/cohere_1m_768d/neighbors.parquet \
//!     "https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors.parquet"
//!
//! Usage:
//!   cargo run --release --example bench_ivf_vs_mstg
//!   cargo run --release --example bench_ivf_vs_mstg -- --random --n 100000 --dim 128
//!   cargo run --release --example bench_ivf_vs_mstg -- --epsilon 0.05

use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams as MstgSearchParams};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams as IvfSearchParams};
use rand::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    use_random: bool,
    n_base: usize,
    n_queries: usize,
    dim: usize,
    k: usize,
    dataset_dir: String,
    limit: Option<usize>,
    epsilon: Option<f32>, // --epsilon: fixed closure epsilon
    epsilon_sweep: bool,  // --epsilon-sweep: test multiple epsilon values
    skip_ivf: bool,       // --skip-ivf: skip IVF benchmark
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        use_random: false,
        n_base: 100_000,
        n_queries: 100,
        dim: 128,
        k: 100,
        dataset_dir: "dataset/cohere_100k_768d".to_string(),
        limit: None,
        epsilon: None,
        epsilon_sweep: false,
        skip_ivf: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--random" => {
                cfg.use_random = true;
                i += 1;
            }
            "--n" => {
                cfg.n_base = args[i + 1].parse().expect("bad --n");
                i += 2;
            }
            "--dim" => {
                cfg.dim = args[i + 1].parse().expect("bad --dim");
                i += 2;
            }
            "--queries" => {
                cfg.n_queries = args[i + 1].parse().expect("bad --queries");
                i += 2;
            }
            "--k" => {
                cfg.k = args[i + 1].parse().expect("bad --k");
                i += 2;
            }
            "--dataset" => {
                cfg.dataset_dir = args[i + 1].clone();
                i += 2;
            }
            "--limit" => {
                cfg.limit = Some(args[i + 1].parse().expect("bad --limit"));
                i += 2;
            }
            "--epsilon" => {
                cfg.epsilon = Some(args[i + 1].parse().expect("bad --epsilon"));
                i += 2;
            }
            "--epsilon-sweep" => {
                cfg.epsilon_sweep = true;
                i += 1;
            }
            "--skip-ivf" => {
                cfg.skip_ivf = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    cfg
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Loaded benchmark dataset
struct BenchData {
    data: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    ground_truth: Vec<Vec<usize>>,
    dim: usize,
    metric: Metric,
}

fn load_parquet_dataset(dir: &str, limit: Option<usize>) -> BenchData {
    use arrow::array::{Array, Float32Array, Int64Array, LargeListArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let load_vectors = |path: &str| -> Vec<Vec<f32>> {
        println!("  loading {} ...", path);
        let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to create parquet reader");
        let reader = builder.build().expect("Failed to build reader");

        let mut all_vecs: Vec<Vec<f32>> = Vec::new();
        for batch in reader {
            let batch = batch.expect("Failed to read batch");
            let emb_col = batch.column_by_name("emb").expect("No 'emb' column found");
            let list_arr = emb_col
                .as_any()
                .downcast_ref::<LargeListArray>()
                .expect("emb column is not LargeListArray");

            for row in 0..list_arr.len() {
                if let Some(max) = limit {
                    if all_vecs.len() >= max {
                        return all_vecs;
                    }
                }
                let values = list_arr.value(row);
                let float_arr = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("emb values are not Float32");
                all_vecs.push(float_arr.values().to_vec());
            }
        }
        all_vecs
    };

    let load_neighbors = |path: &str| -> Vec<Vec<usize>> {
        println!("  loading {} ...", path);
        let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to create parquet reader");
        let reader = builder.build().expect("Failed to build reader");

        let mut all_neighbors: Vec<Vec<usize>> = Vec::new();
        for batch in reader {
            let batch = batch.expect("Failed to read batch");
            let col = batch
                .column_by_name("neighbors_id")
                .expect("No 'neighbors_id' column found");
            let list_arr = col
                .as_any()
                .downcast_ref::<LargeListArray>()
                .expect("neighbors_id column is not LargeListArray");

            for row in 0..list_arr.len() {
                let values = list_arr.value(row);
                let int_arr = values
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("neighbors_id values are not Int64");
                all_neighbors.push(int_arr.values().iter().map(|&v| v as usize).collect());
            }
        }
        all_neighbors
    };

    let train_path = format!("{}/train.parquet", dir);
    let test_path = format!("{}/test.parquet", dir);
    let neighbors_path = format!("{}/neighbors.parquet", dir);

    let data = load_vectors(&train_path);
    let queries = load_vectors(&test_path);
    let ground_truth = load_neighbors(&neighbors_path);
    let dim = data[0].len();

    println!(
        "  loaded: {} train vectors, {} queries, {} ground truth, dim={}",
        data.len(),
        queries.len(),
        ground_truth.len(),
        dim
    );

    // Cohere dataset uses cosine similarity, but we work with L2 here
    // (RaBitQ supports L2 natively; for cosine, vectors should be normalized)
    BenchData {
        data,
        queries,
        ground_truth,
        dim,
        metric: Metric::L2,
    }
}

fn generate_random_dataset(n: usize, n_queries: usize, dim: usize, k: usize) -> BenchData {
    println!("  generating {} random vectors (dim={}) ...", n, dim);
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    let mut rng2 = StdRng::seed_from_u64(123);
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng2.gen::<f32>()).collect())
        .collect();

    println!("  computing brute-force ground truth ...");
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_knn(&data, q, k))
        .collect();

    BenchData {
        data,
        queries,
        ground_truth,
        dim,
        metric: Metric::L2,
    }
}

fn brute_force_knn(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (i, d)
        })
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

fn recall(retrieved: &[usize], ground_truth: &[usize], k: usize) -> f32 {
    let gt: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| gt.contains(id))
        .count();
    hits as f32 / k.min(ground_truth.len()) as f32
}

// ---------------------------------------------------------------------------
// Helper: run a recall/latency sweep on an MSTG index
// ---------------------------------------------------------------------------

struct MstgResult {
    label: String,
    recall: f32,
    latency_us: f64,
}

fn run_mstg_sweep(index: &MstgIndex, bench: &BenchData, k: usize, tag: &str) -> Vec<MstgResult> {
    // Warmup
    let warmup_params = MstgSearchParams::balanced(k);
    for q in bench.queries.iter().take(10) {
        let _ = index.search(q, &warmup_params);
    }

    let sweep: Vec<(usize, f32, &str)> = vec![
        (50, 0.3, "low-latency"),
        (100, 0.5, "balanced-low"),
        (150, 0.6, "balanced"),
        (200, 0.8, "balanced-high"),
        (300, 1.0, "high-recall"),
        (500, 1.5, "max-recall"),
    ];

    println!();
    println!(
        "  {:>12}  {:>10}  {:>10}  {:>10}  {:>12}",
        "profile", "recall@K", "latency", "QPS", "ef / eps"
    );
    println!(
        "  {:->12}  {:->10}  {:->10}  {:->10}  {:->12}",
        "", "", "", "", ""
    );

    let mut results: Vec<MstgResult> = Vec::new();

    for &(ef, eps, label) in &sweep {
        let params = MstgSearchParams::new(ef, eps, k);

        let start = Instant::now();
        let search_results: Vec<Vec<usize>> = bench
            .queries
            .iter()
            .map(|q| {
                index
                    .search(q, &params)
                    .iter()
                    .map(|r| r.vector_id)
                    .collect()
            })
            .collect();
        let elapsed = start.elapsed();
        let n_q = bench.queries.len();
        let latency_us = elapsed.as_secs_f64() * 1_000_000.0 / n_q as f64;
        let qps = n_q as f64 / elapsed.as_secs_f64();

        let avg_recall: f32 = search_results
            .iter()
            .zip(bench.ground_truth.iter())
            .map(|(r, gt)| recall(r, gt, k))
            .sum::<f32>()
            / n_q as f32;

        println!(
            "  {:>12}  {:>9.1}%  {:>8.0} us  {:>10.0}  ef={:<3} e={:.1}",
            label,
            avg_recall * 100.0,
            latency_us,
            qps,
            ef,
            eps,
        );

        results.push(MstgResult {
            label: format!("{}-{}", tag, label),
            recall: avg_recall,
            latency_us,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cfg = parse_args();

    println!();
    println!("========================================================");
    println!("  IVF vs MSTG-mem vs MSTG-disk Benchmark");
    println!("========================================================");

    // Load data
    println!();
    println!("Loading data ...");
    let t0 = Instant::now();

    let bench = if cfg.use_random {
        generate_random_dataset(cfg.n_base, cfg.n_queries, cfg.dim, cfg.k)
    } else {
        let dir = &cfg.dataset_dir;
        let train_path = format!("{}/train.parquet", dir);
        if Path::new(&train_path).exists() {
            load_parquet_dataset(dir, cfg.limit)
        } else {
            println!(
                "  dataset not found at {}, falling back to random data",
                dir
            );
            generate_random_dataset(cfg.n_base, cfg.n_queries, cfg.dim, cfg.k)
        }
    };

    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!();
    println!("  vectors  : {}", bench.data.len());
    println!("  queries  : {}", bench.queries.len());
    println!("  dim      : {}", bench.dim);
    println!("  k        : {}", cfg.k);
    println!("  metric   : {:?}", bench.metric);
    println!("  load time: {:.0} ms", load_ms);

    // =====================================================================
    //  IVF
    // =====================================================================
    struct IvfResult {
        nprobe: usize,
        recall: f32,
        latency_us: f64,
    }
    let mut ivf_results: Vec<IvfResult> = Vec::new();
    let mut ivf_build_ms = 0.0;

    if !cfg.skip_ivf {
        let nlist = (bench.data.len() as f32).sqrt() as usize;

        println!();
        println!("--- IVF (nlist={}) ---", nlist);

        let t0 = Instant::now();
        let ivf_index = IvfRabitqIndex::train(
            &bench.data,
            nlist,
            7,
            bench.metric,
            RotatorType::FhtKacRotator,
            42,
            true,
        )
        .expect("IVF train failed");
        ivf_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ivf_mem_mb = ivf_index.estimate_memory_mb();
        println!("  build    : {:.0} ms", ivf_build_ms);
        println!("  memory   : {:.1} MB", ivf_mem_mb);

        // Warmup
        for q in bench.queries.iter().take(10) {
            let _ = ivf_index.search(
                q,
                IvfSearchParams {
                    nprobe: nlist / 10,
                    top_k: cfg.k,
                },
            );
        }

        // nprobe sweep
        let nprobe_candidates: Vec<usize> = {
            let mut v: Vec<usize> = vec![1, 2, 4, 8];
            let mut p = nlist / 40;
            while p >= 1 && p <= nlist {
                v.push(p);
                p = (p as f64 * 1.5) as usize;
            }
            v.push(nlist);
            v.sort();
            v.dedup();
            v.into_iter().filter(|&p| p >= 1 && p <= nlist).collect()
        };

        println!();
        println!(
            "  {:>8}  {:>10}  {:>10}  {:>10}",
            "nprobe", "recall@K", "latency", "QPS"
        );
        println!("  {:->8}  {:->10}  {:->10}  {:->10}", "", "", "", "");

        for &nprobe in &nprobe_candidates {
            let params = IvfSearchParams {
                nprobe,
                top_k: cfg.k,
            };

            let start = Instant::now();
            let results: Vec<Vec<usize>> = bench
                .queries
                .iter()
                .map(|q| {
                    ivf_index
                        .search(q, params)
                        .unwrap()
                        .iter()
                        .map(|r| r.id)
                        .collect()
                })
                .collect();
            let elapsed = start.elapsed();
            let n_q = bench.queries.len();
            let latency_us = elapsed.as_secs_f64() * 1_000_000.0 / n_q as f64;
            let qps = n_q as f64 / elapsed.as_secs_f64();

            let avg_recall: f32 = results
                .iter()
                .zip(bench.ground_truth.iter())
                .map(|(r, gt)| recall(r, gt, cfg.k))
                .sum::<f32>()
                / n_q as f32;

            println!(
                "  {:>8}  {:>9.1}%  {:>8.0} us  {:>10.0}",
                nprobe,
                avg_recall * 100.0,
                latency_us,
                qps,
            );

            ivf_results.push(IvfResult {
                nprobe,
                recall: avg_recall,
                latency_us,
            });
        }
    }

    // =====================================================================
    //  Epsilon sweep (if requested)
    // =====================================================================
    if cfg.epsilon_sweep {
        let epsilons = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.3];

        println!();
        println!("========================================================");
        println!("  Epsilon Sweep (replication factor analysis)");
        println!("========================================================");
        println!();
        println!(
            "  {:>8}  {:>14}  {:>10}  {:>10}",
            "epsilon", "repl. factor", "build (ms)", "memory (MB)"
        );
        println!("  {:->8}  {:->14}  {:->10}  {:->10}", "", "", "", "");

        for &eps in &epsilons {
            let mstg_config = MstgConfig {
                max_posting_size: 1000,
                branching_factor: 8,
                balance_weight: 1.0,
                closure_epsilon: eps,
                max_replicas: 5,
                rabitq_bits: 7,
                faster_config: true,
                metric: bench.metric,
                hnsw_m: 32,
                hnsw_ef_construction: 200,
                centroid_precision: rabitq_rs::mstg::ScalarPrecision::BF16,
                default_ef_search: 200,
                pruning_epsilon: 1.5,
            };

            let t0 = Instant::now();
            let mstg_index = MstgIndex::build(&bench.data, mstg_config).expect("MSTG build failed");
            let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Count total vectors across all posting lists
            let total_vecs: usize = mstg_index
                .directory
                .entries
                .iter()
                .map(|e| e.num_vectors as usize)
                .sum();
            let repl_factor = total_vecs as f64 / bench.data.len() as f64;
            let mem_mb = MstgIndex::estimate_memory_mb(
                &mstg_index.centroid_index,
                &mstg_index.posting_lists,
            );

            println!(
                "  {:>8.3}  {:>14.2}  {:>10.0}  {:>10.1}",
                eps, repl_factor, build_ms, mem_mb,
            );
        }
    }

    // =====================================================================
    //  MSTG (memory mode)
    // =====================================================================
    let closure_eps = cfg.epsilon.unwrap_or(0.3);

    println!();
    println!("--- MSTG-mem (epsilon={}) ---", closure_eps);

    let mstg_config = MstgConfig {
        max_posting_size: 1000,
        branching_factor: 8,
        balance_weight: 1.0,
        closure_epsilon: closure_eps,
        max_replicas: 5,
        rabitq_bits: 7,
        faster_config: true,
        metric: bench.metric,
        hnsw_m: 32,
        hnsw_ef_construction: 200,
        centroid_precision: rabitq_rs::mstg::ScalarPrecision::BF16,
        default_ef_search: 200,
        pruning_epsilon: 1.5,
    };

    let t0 = Instant::now();
    let mstg_index = MstgIndex::build(&bench.data, mstg_config).expect("MSTG build failed");
    let mstg_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  build    : {:.0} ms", mstg_build_ms);

    // Report replication factor
    let total_vecs: usize = mstg_index
        .directory
        .entries
        .iter()
        .map(|e| e.num_vectors as usize)
        .sum();
    let repl_factor = total_vecs as f64 / bench.data.len() as f64;
    let mem_mb =
        MstgIndex::estimate_memory_mb(&mstg_index.centroid_index, &mstg_index.posting_lists);
    println!("  repl     : {:.2}x", repl_factor);
    println!("  memory   : {:.1} MB", mem_mb);

    let mstg_mem_results = run_mstg_sweep(&mstg_index, &bench, cfg.k, "mem");

    // =====================================================================
    //  MSTG (disk mode)
    // =====================================================================
    println!();
    println!("--- MSTG-disk (mmap, same index) ---");

    // Save index to temp path
    let save_path = format!(
        "/tmp/bench_mstg_disk_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    let t0 = Instant::now();
    mstg_index.save_to_path(&save_path).unwrap();
    let save_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  save     : {:.0} ms", save_ms);

    // Load as mmap
    let t0 = Instant::now();
    let mstg_disk = MstgIndex::load_from_path_mmap(&save_path).unwrap();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  mmap load: {:.0} ms", load_ms);

    let disk_mem_mb =
        MstgIndex::estimate_memory_mb(&mstg_disk.centroid_index, &mstg_disk.posting_lists);
    println!(
        "  memory   : {:.1} MB (centroid + directory only)",
        disk_mem_mb
    );

    let mstg_disk_results = run_mstg_sweep(&mstg_disk, &bench, cfg.k, "disk");

    // Clean up temp files
    let _ = std::fs::remove_file(format!("{}.mstg", save_path));
    let _ = std::fs::remove_file(format!("{}.hnsw.graph", save_path));
    let _ = std::fs::remove_file(format!("{}.hnsw.data", save_path));

    // =====================================================================
    //  Head-to-head comparison
    // =====================================================================
    println!();
    println!("========================================================");
    println!("  Head-to-head (matched recall)");
    println!("========================================================");
    println!();

    // Compare MSTG-mem vs IVF
    if !cfg.skip_ivf {
        println!("  MSTG-mem vs IVF:");
        for mstg_r in &mstg_mem_results {
            if let Some(ivf_r) = ivf_results.iter().min_by(|a, b| {
                (a.recall - mstg_r.recall)
                    .abs()
                    .partial_cmp(&(b.recall - mstg_r.recall).abs())
                    .unwrap()
            }) {
                let recall_diff = (ivf_r.recall - mstg_r.recall).abs();
                if recall_diff > 0.10 {
                    continue;
                }
                let speedup = ivf_r.latency_us / mstg_r.latency_us;
                let faster = if speedup > 1.0 { "MSTG-mem" } else { "IVF" };
                let ratio = if speedup > 1.0 {
                    speedup
                } else {
                    1.0 / speedup
                };

                println!(
                    "    recall ~{:.0}%: MSTG-mem({}) {:.0}us vs IVF(nprobe={}) {:.0}us => {} {:.2}x faster",
                    mstg_r.recall * 100.0,
                    mstg_r.label,
                    mstg_r.latency_us,
                    ivf_r.nprobe,
                    ivf_r.latency_us,
                    faster,
                    ratio,
                );
            }
        }
        println!();
    }

    // Compare MSTG-disk vs MSTG-mem
    println!("  MSTG-disk vs MSTG-mem:");
    for (disk_r, mem_r) in mstg_disk_results.iter().zip(mstg_mem_results.iter()) {
        let speedup = disk_r.latency_us / mem_r.latency_us;
        println!(
            "    recall ~{:.0}%: disk {:.0}us vs mem {:.0}us => disk {:.1}x slower",
            mem_r.recall * 100.0,
            disk_r.latency_us,
            mem_r.latency_us,
            speedup,
        );
    }

    // =====================================================================
    //  Summary
    // =====================================================================
    println!();
    println!("========================================================");
    println!("  Summary");
    println!("========================================================");
    println!();

    println!(
        "  MSTG config: epsilon={}, replication={:.2}x",
        closure_eps, repl_factor
    );
    println!(
        "  MSTG-mem : build={:.0}ms, memory={:.1}MB",
        mstg_build_ms, mem_mb
    );
    println!(
        "  MSTG-disk: save={:.0}ms, mmap_load={:.0}ms, memory={:.1}MB",
        save_ms, load_ms, disk_mem_mb
    );

    if !cfg.skip_ivf {
        let build_ratio = mstg_build_ms / ivf_build_ms;
        if build_ratio > 1.0 {
            println!(
                "  Build: IVF {:.0}ms vs MSTG {:.0}ms (IVF {:.1}x faster)",
                ivf_build_ms, mstg_build_ms, build_ratio,
            );
        } else {
            println!(
                "  Build: IVF {:.0}ms vs MSTG {:.0}ms (MSTG {:.1}x faster)",
                ivf_build_ms,
                mstg_build_ms,
                1.0 / build_ratio,
            );
        }

        if let Some(best_mstg) = mstg_mem_results
            .iter()
            .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap())
        {
            if let Some(best_ivf) = ivf_results
                .iter()
                .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap())
            {
                println!(
                    "  Peak recall: IVF {:.1}% (nprobe={}) vs MSTG-mem {:.1}% ({})",
                    best_ivf.recall * 100.0,
                    best_ivf.nprobe,
                    best_mstg.recall * 100.0,
                    best_mstg.label,
                );
            }
        }
    }

    if let Some(best_disk) = mstg_disk_results
        .iter()
        .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap())
    {
        if let Some(best_mem) = mstg_mem_results
            .iter()
            .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap())
        {
            println!(
                "  Peak recall: MSTG-mem {:.1}% ({}) vs MSTG-disk {:.1}% ({})",
                best_mem.recall * 100.0,
                best_mem.label,
                best_disk.recall * 100.0,
                best_disk.label,
            );
        }
    }

    println!();
    println!("Done.");
}
