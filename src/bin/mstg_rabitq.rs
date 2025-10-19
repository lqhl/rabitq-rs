use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};

use rabitq_rs::io::{read_fvecs, read_groundtruth};
use rabitq_rs::mstg::{MstgConfig, MstgRabitqIndex, MstgSearchParams};
use rabitq_rs::{Metric, RotatorType};

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    if env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return;
    }

    let args: Vec<String> = env::args().skip(1).collect();
    let config = match Config::parse(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("Error: {message}\n");
            print_usage();
            process::exit(1);
        }
    };

    if let Err(err) = run(config) {
        eprintln!("Error: {err}");
        let mut source = err.source();
        while let Some(inner) = source {
            eprintln!("  caused by: {inner}");
            source = inner.source();
        }
        process::exit(1);
    }
}

fn run(config: Config) -> CliResult<()> {
    println!("Loading base vectors from {}...", config.base.display());
    let base = read_fvecs(&config.base, config.max_base)?;

    if base.is_empty() {
        return Err("No base vectors loaded".into());
    }

    let dim = base.first().unwrap().len();
    println!("Loaded {} vectors (dim: {})", base.len(), dim);

    // Build MSTG index
    println!("\n=== Building MSTG Index ===");
    let mstg_config = MstgConfig {
        k: config.k,
        target_cluster_size: config.target_cluster_size,
        total_bits: config.bits,
        use_faster_config: config.faster_config,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 400,
    };

    let start = Instant::now();
    let index = MstgRabitqIndex::train(
        &base,
        mstg_config,
        config.metric,
        RotatorType::FhtKacRotator,
        config.seed,
    )?;
    let build_time = start.elapsed();

    println!("\n=== Index Built ===");
    println!("Time: {:.2?}", build_time);
    println!("Vectors: {}", index.len());
    println!("Clusters: {}", index.cluster_count());

    // Load queries and ground truth
    println!("\n=== Loading Queries ===");
    println!("Loading query vectors from {}...", config.queries.display());
    let queries = read_fvecs(&config.queries, config.max_queries)?;
    println!("Loaded {} queries", queries.len());

    println!(
        "Loading ground truth from {}...",
        config.groundtruth.display()
    );
    let groundtruth = read_groundtruth(&config.groundtruth, config.max_queries)?;
    println!("Loaded ground truth ({} entries)", groundtruth.len());

    if queries.is_empty() {
        return Err("No query vectors loaded".into());
    }
    if groundtruth.len() != queries.len() {
        return Err("Ground truth count doesn't match query count".into());
    }

    // Run benchmark
    println!("\n=== Running Query Evaluation ===");
    if config.benchmark_mode {
        benchmark_with_sweep(&index, &queries, &groundtruth, config.top_k)?;
    } else {
        evaluate_single_config(&index, &queries, &groundtruth, &config)?;
    }

    Ok(())
}

fn evaluate_single_config(
    index: &MstgRabitqIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<usize>],
    config: &Config,
) -> CliResult<()> {
    let params = MstgSearchParams::new(config.top_k, config.nprobe);

    // Warmup
    if config.warmup_queries > 0 && !queries.is_empty() {
        let warmup_count = config.warmup_queries.min(queries.len());
        println!("\nWarmup: {} queries...", warmup_count);
        for query in queries.iter().take(warmup_count) {
            let _ = index.search(query, params);
        }
    }

    println!(
        "\nEvaluating {} queries (top-k={}, nprobe={})...",
        queries.len(),
        config.top_k,
        config.nprobe
    );

    let total_hits = AtomicUsize::new(0);
    let total_possible = AtomicUsize::new(0);

    let start = Instant::now();
    let latencies: Vec<Duration> = queries
        .par_iter()
        .enumerate()
        .map(|(idx, query)| {
            let query_start = Instant::now();
            let results = index.search(query, params).expect("search failed");
            let latency = query_start.elapsed();

            let gt = &groundtruth[idx];
            let limit = config.top_k.min(gt.len());
            total_possible.fetch_add(limit, Ordering::Relaxed);

            let gt_set: HashSet<usize> = gt[..limit].iter().copied().collect();
            let hits = results
                .iter()
                .take(limit)
                .filter(|r| gt_set.contains(&r.id))
                .count();
            total_hits.fetch_add(hits, Ordering::Relaxed);

            latency
        })
        .collect();

    let total_time = start.elapsed();
    let hits = total_hits.load(Ordering::Relaxed);
    let possible = total_possible.load(Ordering::Relaxed);
    let recall = if possible > 0 {
        hits as f64 / possible as f64
    } else {
        0.0
    };
    let qps = queries.len() as f64 / total_time.as_secs_f64();

    // Latency stats
    let mut sorted = latencies.clone();
    sorted.sort();
    let percentile = |p: usize| {
        sorted
            .get(sorted.len() * p / 100)
            .map(|d| d.as_micros())
            .unwrap_or(0)
    };

    println!("\n=== Results ===");
    println!("Recall@{}: {:.4}", config.top_k, recall);
    println!("QPS: {:.2}", qps);
    println!("\nLatency (μs):");
    println!(
        "  Min: {:>8}",
        sorted.first().map(|d| d.as_micros()).unwrap_or(0)
    );
    println!(
        "  Avg: {:>8}",
        latencies.iter().map(|d| d.as_micros()).sum::<u128>() / latencies.len() as u128
    );
    println!("  P50: {:>8}", percentile(50));
    println!("  P95: {:>8}", percentile(95));
    println!("  P99: {:>8}", percentile(99));
    println!(
        "  Max: {:>8}",
        sorted.last().map(|d| d.as_micros()).unwrap_or(0)
    );

    Ok(())
}

fn benchmark_with_sweep(
    index: &MstgRabitqIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<usize>],
    top_k: usize,
) -> CliResult<()> {
    // Define nprobe sweep with larger intervals
    let mut all_nprobes = vec![5, 10, 20, 30, 40, 50];
    for i in (60..200).step_by(20) {
        all_nprobes.push(i);
    }
    for i in (200..=500).step_by(50) {
        all_nprobes.push(i);
    }
    for i in (600..=1000).step_by(100) {
        all_nprobes.push(i);
    }

    println!("\nPerforming nprobe sweep...");
    println!("\n{:<10} {:<12} {:<12} {:<12}", "nprobe", "recall", "QPS", "latency(ms)");
    println!("{}", "-".repeat(50));

    let mut prev_recall = 0.0f32;

    for &nprobe in &all_nprobes {
        let actual_nprobe = nprobe.min(index.cluster_count());
        let params = MstgSearchParams::new(top_k, actual_nprobe);

        let start = Instant::now();

        // Parallel search with latency tracking
        let results: Vec<(usize, Duration)> = queries
            .par_iter()
            .enumerate()
            .map(|(i, query)| {
                let query_start = Instant::now();
                let results = index.search(query, params).expect("search failed");
                let latency = query_start.elapsed();

                let gt_set: HashSet<usize> = groundtruth[i].iter().take(top_k).copied().collect();
                let correct = results
                    .iter()
                    .take(top_k)
                    .filter(|r| gt_set.contains(&r.id))
                    .count();

                (correct, latency)
            })
            .collect();

        let elapsed = start.elapsed().as_secs_f64();
        let total_correct: usize = results.iter().map(|(c, _)| c).sum();
        let avg_latency_ms: f64 = results.iter()
            .map(|(_, l)| l.as_secs_f64() * 1000.0)
            .sum::<f64>() / results.len() as f64;

        let recall = total_correct as f32 / (queries.len() * top_k) as f32;
        let qps = queries.len() as f64 / elapsed;

        println!(
            "{:<10} {:<12.5} {:<12.2} {:<12.2}",
            nprobe, recall, qps, avg_latency_ms
        );

        // Stop if recall > 99% or improvement < 0.001
        if recall > 0.99 || (recall - prev_recall).abs() < 0.001 {
            break;
        }
        prev_recall = recall;
    }

    Ok(())
}

#[derive(Debug)]
struct Config {
    base: PathBuf,
    queries: PathBuf,
    groundtruth: PathBuf,
    k: usize,
    target_cluster_size: usize,
    bits: usize,
    metric: Metric,
    seed: u64,
    top_k: usize,
    nprobe: usize,
    benchmark_mode: bool,
    max_base: Option<usize>,
    max_queries: Option<usize>,
    warmup_queries: usize,
    faster_config: bool,
}

impl Config {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut base = None;
        let mut queries = None;
        let mut groundtruth = None;
        let mut k = 16;
        let mut target_cluster_size = 100;
        let mut bits = 7;
        let mut metric = Metric::L2;
        let mut seed = 0x5a5a_1234_u64;
        let mut top_k = 100;
        let mut nprobe = 64;
        let mut benchmark_mode = false;
        let mut max_base = None;
        let mut max_queries = None;
        let mut warmup_queries = 10;
        let mut faster_config = false;

        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--base" => base = Some(next_path(&mut iter, &arg)?),
                "--queries" => queries = Some(next_path(&mut iter, &arg)?),
                "--groundtruth" | "--gt" => groundtruth = Some(next_path(&mut iter, &arg)?),
                "--k" => k = next_usize(&mut iter, &arg)?,
                "--target-cluster-size" => target_cluster_size = next_usize(&mut iter, &arg)?,
                "--bits" => bits = next_usize(&mut iter, &arg)?,
                "--metric" => {
                    let value = next_value(&mut iter, &arg)?;
                    metric = parse_metric(&value)?;
                }
                "--seed" => seed = next_u64(&mut iter, &arg)?,
                "--top-k" | "--topk" => top_k = next_usize(&mut iter, &arg)?,
                "--nprobe" => nprobe = next_usize(&mut iter, &arg)?,
                "--benchmark" | "--sweep" => benchmark_mode = true,
                "--max-base" => max_base = Some(next_usize(&mut iter, &arg)?),
                "--max-queries" => max_queries = Some(next_usize(&mut iter, &arg)?),
                "--warmup" => warmup_queries = next_usize(&mut iter, &arg)?,
                "--faster-config" => faster_config = true,
                other => {
                    return Err(format!("Unknown argument: {}", other));
                }
            }
        }

        let base = base.ok_or("--base is required")?;
        let queries = queries.ok_or("--queries is required")?;
        let groundtruth = groundtruth.ok_or("--groundtruth or --gt is required")?;

        if bits == 0 || bits > 16 {
            return Err("--bits must be between 1 and 16".to_string());
        }
        if k < 2 {
            return Err("--k must be >= 2".to_string());
        }
        if target_cluster_size == 0 {
            return Err("--target-cluster-size must be positive".to_string());
        }

        Ok(Self {
            base,
            queries,
            groundtruth,
            k,
            target_cluster_size,
            bits,
            metric,
            seed,
            top_k,
            nprobe,
            benchmark_mode,
            max_base,
            max_queries,
            warmup_queries,
            faster_config,
        })
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("missing value for {}", flag))
}

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(next_value(iter, flag)?))
}

fn next_usize(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let value = next_value(iter, flag)?;
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid value for {}: {}", flag, value))
}

fn next_u64(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
    let value = next_value(iter, flag)?;
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid value for {}: {}", flag, value))
}

fn parse_metric(value: &str) -> Result<Metric, String> {
    match value.to_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "ip" | "innerproduct" | "inner_product" => Ok(Metric::InnerProduct),
        other => Err(format!("unsupported metric: {}", other)),
    }
}

fn print_usage() {
    eprintln!("mstg_rabitq - MSTG+RaBitQ index builder and query evaluator\n");
    eprintln!("USAGE:");
    eprintln!("  mstg_rabitq --base <data.fvecs> --queries <Q> --gt <G> [OPTIONS]\n");

    eprintln!("REQUIRED:");
    eprintln!("  --base <path>              Base vectors (.fvecs)");
    eprintln!("  --queries <path>           Query vectors (.fvecs)");
    eprintln!("  --gt <path>                Ground truth (.ivecs)\n");

    eprintln!("INDEX CONFIGURATION:");
    eprintln!("  --k <N>                    Branching factor for tree (default: 16)");
    eprintln!("  --target-cluster-size <N>  Vectors per bottom cluster (default: 100)");
    eprintln!("  --bits <1-16>              Quantization bits (default: 7)");
    eprintln!("  --metric <l2|ip>           Distance metric (default: l2)");
    eprintln!("  --seed <N>                 Random seed (default: 1515870004)");
    eprintln!("  --faster-config            Use faster RaBitQ config\n");

    eprintln!("QUERY CONFIGURATION:");
    eprintln!("  --top-k <N>                Number of neighbors (default: 100)");
    eprintln!("  --nprobe <N>               Clusters to probe (default: 64)");
    eprintln!("  --benchmark                Run nprobe sweep + multi-round benchmark\n");

    eprintln!("LIMITS & TUNING:");
    eprintln!("  --max-base <N>             Limit base vectors");
    eprintln!("  --max-queries <N>          Limit query vectors");
    eprintln!("  --warmup <N>               Warmup queries (default: 10)\n");

    eprintln!("EXAMPLES:");
    eprintln!("  # Build and evaluate on GIST dataset");
    eprintln!("  mstg_rabitq --base data/gist/gist_base.fvecs \\");
    eprintln!("              --queries data/gist/gist_query.fvecs \\");
    eprintln!("              --gt data/gist/gist_groundtruth.ivecs \\");
    eprintln!("              --benchmark\n");

    eprintln!("  # Quick smoke test");
    eprintln!("  mstg_rabitq --base data/gist/gist_base.fvecs \\");
    eprintln!("              --queries data/gist/gist_query.fvecs \\");
    eprintln!("              --gt data/gist/gist_groundtruth.ivecs \\");
    eprintln!("              --max-base 10000 --max-queries 100 --nprobe 64\n");
}
