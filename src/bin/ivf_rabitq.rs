use std::backtrace::Backtrace;
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};

use rabitq_rs::io::{read_fvecs, read_groundtruth, read_ids};
use rabitq_rs::ivf::{IvfRabitqIndex, SearchParams};
use rabitq_rs::Metric;

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    if let Err(err) = install_signal_handlers() {
        eprintln!("Warning: failed to install signal handlers: {err}");
    }

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
    match config.mode {
        Mode::Query => run_query_mode(&config),
        Mode::Index => run_index_mode(&config),
        Mode::IndexAndQuery => run_index_and_query_mode(&config),
    }
}

fn run_query_mode(config: &Config) -> CliResult<()> {
    let index_path = config.load_index.as_ref().unwrap();

    println!("Loading IVF index from {}...", index_path.display());
    let index = IvfRabitqIndex::load_from_path(index_path)?;
    println!(
        "Index loaded ({} vectors, {} clusters)",
        index.len(),
        index.cluster_count()
    );

    let query_path = config.queries.as_ref().unwrap();
    let gt_path = config.groundtruth.as_ref().unwrap();

    println!("\nLoading query vectors from {}...", query_path.display());
    let queries = read_fvecs(query_path, config.max_queries)?;
    println!(
        "Loaded {} queries (dim: {})",
        queries.len(),
        queries.first().map(|v| v.len()).unwrap_or(0)
    );

    println!("Loading ground truth from {}...", gt_path.display());
    let groundtruth = read_groundtruth(gt_path, config.max_queries)?;
    println!(
        "Loaded ground truth ({} entries, top-{})",
        groundtruth.len(),
        groundtruth.first().map(|v| v.len()).unwrap_or(0)
    );

    if queries.is_empty() {
        return Err("No query vectors loaded".into());
    }
    if groundtruth.len() != queries.len() {
        return Err("Ground truth count doesn't match query count".into());
    }

    println!("\n=== Running Query Evaluation ===");
    if config.benchmark_mode {
        benchmark_with_sweep(&index, &queries, &groundtruth, config)?;
    } else {
        evaluate_single_config(&index, &queries, &groundtruth, config)?;
    }

    Ok(())
}

fn run_index_mode(config: &Config) -> CliResult<()> {
    let base_path = &config.base;

    println!("Loading base vectors from {}...", base_path.display());
    let base = read_fvecs(base_path, config.max_base)?;

    if base.is_empty() {
        return Err("No base vectors loaded".into());
    }

    let dim = base.first().unwrap().len();
    if base.iter().any(|v| v.len() != dim) {
        return Err("Inconsistent vector dimensions in base dataset".into());
    }

    println!("Loaded {} vectors (dim: {})", base.len(), dim);

    let start = Instant::now();
    let index = build_index(&base, config)?;
    let elapsed = start.elapsed();

    println!("\n=== Index Built ===");
    println!("Time: {:.2?}", elapsed);
    println!("Vectors: {}", index.len());
    println!("Clusters: {}", index.cluster_count());

    if let Some(save_path) = &config.save_index {
        println!("\nSaving index to {}...", save_path.display());
        index.save_to_path(save_path)?;
        println!("Index saved successfully");
    }

    Ok(())
}

fn run_index_and_query_mode(config: &Config) -> CliResult<()> {
    // Build index first
    let base_path = &config.base;

    println!("Loading base vectors from {}...", base_path.display());
    let base = read_fvecs(base_path, config.max_base)?;

    if base.is_empty() {
        return Err("No base vectors loaded".into());
    }

    let dim = base.first().unwrap().len();
    if base.iter().any(|v| v.len() != dim) {
        return Err("Inconsistent vector dimensions in base dataset".into());
    }

    println!("Loaded {} vectors (dim: {})", base.len(), dim);

    let start = Instant::now();
    let index = build_index(&base, config)?;
    let elapsed = start.elapsed();

    println!("\n=== Index Built ===");
    println!("Time: {:.2?}", elapsed);
    println!("Vectors: {}", index.len());
    println!("Clusters: {}", index.cluster_count());

    if let Some(save_path) = &config.save_index {
        println!("\nSaving index to {}...", save_path.display());
        index.save_to_path(save_path)?;
        println!("Index saved successfully");
    }

    // Now run queries
    let query_path = config.queries.as_ref().unwrap();
    let gt_path = config.groundtruth.as_ref().unwrap();

    println!("\n=== Running Query Evaluation ===");
    println!("Loading query vectors from {}...", query_path.display());
    let queries = read_fvecs(query_path, config.max_queries)?;
    println!("Loaded {} queries", queries.len());

    println!("Loading ground truth from {}...", gt_path.display());
    let groundtruth = read_groundtruth(gt_path, config.max_queries)?;
    println!("Loaded ground truth ({} entries)", groundtruth.len());

    if config.benchmark_mode {
        benchmark_with_sweep(&index, &queries, &groundtruth, config)?;
    } else {
        evaluate_single_config(&index, &queries, &groundtruth, config)?;
    }

    Ok(())
}

fn build_index(base: &[Vec<f32>], config: &Config) -> CliResult<IvfRabitqIndex> {
    let index = if let Some(load_path) = &config.load_index {
        println!("Loading pre-built index from {}...", load_path.display());
        IvfRabitqIndex::load_from_path(load_path)?
    } else if let (Some(centroids_path), Some(assignments_path)) =
        (&config.centroids, &config.assignments)
    {
        println!("Building index with pre-computed clusters...");
        println!("  Centroids: {}", centroids_path.display());
        println!("  Assignments: {}", assignments_path.display());

        let centroids = read_fvecs(centroids_path, None)?;
        let assignments = read_ids(assignments_path, config.max_base)?;

        if assignments.len() != base.len() {
            return Err(format!(
                "Assignment count ({}) doesn't match base count ({})",
                assignments.len(),
                base.len()
            )
            .into());
        }

        IvfRabitqIndex::train_with_clusters(
            base,
            &centroids,
            &assignments,
            config.bits,
            config.metric,
            config.seed,
        )?
    } else if let Some(nlist) = config.nlist {
        println!(
            "Training index with built-in k-means ({} clusters)...",
            nlist
        );
        IvfRabitqIndex::train(base, nlist, config.bits, config.metric, config.seed)?
    } else {
        return Err(
            "Must specify either --load-index, --nlist, or both --centroids and --assignments"
                .into(),
        );
    };

    Ok(index)
}

fn evaluate_single_config(
    index: &IvfRabitqIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<usize>],
    config: &Config,
) -> CliResult<()> {
    let params = SearchParams::new(config.top_k, config.nprobe);

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
    index: &IvfRabitqIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<usize>],
    config: &Config,
) -> CliResult<()> {
    let top_k = config.top_k;
    let test_rounds = 5;

    // Define nprobe sweep
    let mut all_nprobes = vec![5];
    for i in (10..200).step_by(10) {
        all_nprobes.push(i);
    }
    for i in (200..400).step_by(40) {
        all_nprobes.push(i);
    }
    for i in (400..=1500).step_by(100) {
        all_nprobes.push(i);
    }
    for i in (2000..=4000).step_by(500) {
        all_nprobes.push(i);
    }
    all_nprobes.extend_from_slice(&[6000, 10000, 15000]);

    println!("\nPerforming nprobe sweep...");
    let nprobes = determine_nprobes(index, &all_nprobes, queries, groundtruth, top_k);

    println!("\n=== Running {}-round benchmark ===", test_rounds);
    println!("Selected nprobes: {:?}", nprobes);

    let mut all_qps = vec![vec![0.0f32; nprobes.len()]; test_rounds];
    let mut all_recall = vec![vec![0.0f32; nprobes.len()]; test_rounds];

    for round in 0..test_rounds {
        for (idx, &nprobe) in nprobes.iter().enumerate() {
            let actual_nprobe = nprobe.min(index.cluster_count());
            let params = SearchParams::new(top_k, actual_nprobe);

            let mut total_correct = 0;
            let start = Instant::now();

            for (i, query) in queries.iter().enumerate() {
                let results = index.search(query, params)?;
                let gt_set: HashSet<usize> = groundtruth[i].iter().take(top_k).copied().collect();
                total_correct += results
                    .iter()
                    .take(top_k)
                    .filter(|r| gt_set.contains(&r.id))
                    .count();
            }

            let elapsed = start.elapsed().as_secs_f64();
            all_qps[round][idx] = (queries.len() as f64 / elapsed) as f32;
            all_recall[round][idx] = total_correct as f32 / (queries.len() * top_k) as f32;
        }
    }

    // Average across rounds
    let avg_qps: Vec<f32> = (0..nprobes.len())
        .map(|i| all_qps.iter().map(|r| r[i]).sum::<f32>() / test_rounds as f32)
        .collect();
    let avg_recall: Vec<f32> = (0..nprobes.len())
        .map(|i| all_recall.iter().map(|r| r[i]).sum::<f32>() / test_rounds as f32)
        .collect();

    println!("\n{:<10} {:<12} {:<10}", "nprobe", "QPS", "recall");
    println!("{}", "-".repeat(35));
    for (i, &nprobe) in nprobes.iter().enumerate() {
        println!(
            "{:<10} {:<12.2} {:<10.5}",
            nprobe, avg_qps[i], avg_recall[i]
        );
    }

    Ok(())
}

fn determine_nprobes(
    index: &IvfRabitqIndex,
    all_nprobes: &[usize],
    queries: &[Vec<f32>],
    groundtruth: &[Vec<usize>],
    top_k: usize,
) -> Vec<usize> {
    let mut selected = Vec::new();
    let mut prev_recall = 0.0f32;

    println!("\n{:<10} {:<10}", "recall", "nprobe");
    println!("{}", "-".repeat(25));

    for &nprobe in all_nprobes {
        selected.push(nprobe);

        let actual_nprobe = nprobe.min(index.cluster_count());
        let params = SearchParams::new(top_k, actual_nprobe);

        let mut total_correct = 0;
        for (i, query) in queries.iter().enumerate() {
            let results = index.search(query, params).expect("search failed");
            let gt_set: HashSet<usize> = groundtruth[i].iter().take(top_k).copied().collect();
            total_correct += results
                .iter()
                .take(top_k)
                .filter(|r| gt_set.contains(&r.id))
                .count();
        }

        let recall = total_correct as f32 / (queries.len() * top_k) as f32;
        println!("{:<10.5} {:<10}", recall, nprobe);

        // Stop if recall > 99.7% or improvement < 0.00001
        if recall > 0.997 || (recall - prev_recall).abs() < 1e-5 {
            break;
        }
        prev_recall = recall;
    }

    selected
}

#[derive(Debug)]
enum Mode {
    Query,         // Load index and query
    Index,         // Build index (and optionally save)
    IndexAndQuery, // Build index then query
}

#[derive(Debug)]
struct Config {
    mode: Mode,

    // Index building
    base: PathBuf,
    centroids: Option<PathBuf>,
    assignments: Option<PathBuf>,
    nlist: Option<usize>,
    bits: usize,
    metric: Metric,
    seed: u64,

    // Index persistence
    load_index: Option<PathBuf>,
    save_index: Option<PathBuf>,

    // Querying
    queries: Option<PathBuf>,
    groundtruth: Option<PathBuf>,
    top_k: usize,
    nprobe: usize,

    // Evaluation mode
    benchmark_mode: bool, // If true, do nprobe sweep + multi-round benchmark

    // Limits and tuning
    max_base: Option<usize>,
    max_queries: Option<usize>,
    #[allow(dead_code)]
    threads: Option<usize>,
    warmup_queries: usize,
}

impl Config {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut base = None;
        let mut centroids = None;
        let mut assignments = None;
        let mut nlist = None;
        let mut bits = None;
        let mut metric = Metric::L2;
        let mut queries = None;
        let mut groundtruth = None;
        let mut top_k = 100;
        let mut nprobe = 64;
        let mut max_base = None;
        let mut max_queries = None;
        let mut seed = 0x5a5a_1234_u64;
        let mut load_index = None;
        let mut save_index = None;
        let mut threads = None;
        let mut warmup_queries = 10;
        let mut benchmark_mode = false;

        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--base" => base = Some(next_path(&mut iter, &arg)?),
                "--centroids" => centroids = Some(next_path(&mut iter, &arg)?),
                "--assignments" => assignments = Some(next_path(&mut iter, &arg)?),
                "--nlist" => nlist = Some(next_usize(&mut iter, &arg)?),
                "--bits" => bits = Some(next_usize(&mut iter, &arg)?),
                "--metric" => {
                    let value = next_value(&mut iter, &arg)?;
                    metric = parse_metric(&value)?;
                }
                "--queries" => queries = Some(next_path(&mut iter, &arg)?),
                "--groundtruth" | "--gt" => groundtruth = Some(next_path(&mut iter, &arg)?),
                "--top-k" | "--topk" => top_k = next_usize(&mut iter, &arg)?,
                "--nprobe" => nprobe = next_usize(&mut iter, &arg)?,
                "--max-base" => max_base = Some(next_usize(&mut iter, &arg)?),
                "--max-queries" => max_queries = Some(next_usize(&mut iter, &arg)?),
                "--seed" => seed = next_u64(&mut iter, &arg)?,
                "--load-index" | "--load" => load_index = Some(next_path(&mut iter, &arg)?),
                "--save-index" | "--save" => save_index = Some(next_path(&mut iter, &arg)?),
                "--threads" => threads = Some(next_usize(&mut iter, &arg)?),
                "--warmup" => warmup_queries = next_usize(&mut iter, &arg)?,
                "--benchmark" | "--sweep" => benchmark_mode = true,
                other => {
                    return Err(format!("Unknown argument: {}", other));
                }
            }
        }

        // Determine mode
        let mode = if load_index.is_some() && base.is_none() {
            // Query-only mode
            if queries.is_none() || groundtruth.is_none() {
                return Err("Query mode requires --queries and --groundtruth".to_string());
            }
            Mode::Query
        } else if base.is_some() && queries.is_some() && groundtruth.is_some() {
            // Index and query mode
            Mode::IndexAndQuery
        } else if base.is_some() {
            // Index-only mode
            Mode::Index
        } else {
            return Err(
                "Must specify either --base (for indexing) or --load-index (for querying)"
                    .to_string(),
            );
        };

        // Validate based on mode
        match mode {
            Mode::Index | Mode::IndexAndQuery => {
                let base = base.ok_or("--base is required for indexing")?;

                if load_index.is_none() {
                    // Building new index
                    let bits = bits.ok_or("--bits is required when building an index")?;
                    if bits == 0 || bits > 16 {
                        return Err("--bits must be between 1 and 16".to_string());
                    }

                    let has_clusters = centroids.is_some() && assignments.is_some();
                    let has_nlist = nlist.is_some();

                    if !has_clusters && !has_nlist {
                        return Err(
                            "Must specify either --nlist OR both --centroids and --assignments"
                                .to_string(),
                        );
                    }
                    if centroids.is_some() != assignments.is_some() {
                        return Err(
                            "--centroids and --assignments must be used together".to_string()
                        );
                    }

                    Ok(Self {
                        mode,
                        base,
                        centroids,
                        assignments,
                        nlist,
                        bits,
                        metric,
                        seed,
                        load_index,
                        save_index,
                        queries,
                        groundtruth,
                        top_k,
                        nprobe,
                        benchmark_mode,
                        max_base,
                        max_queries,
                        threads,
                        warmup_queries,
                    })
                } else {
                    // Loading existing index
                    Ok(Self {
                        mode,
                        base,
                        centroids: None,
                        assignments: None,
                        nlist: None,
                        bits: bits.unwrap_or(0),
                        metric,
                        seed,
                        load_index,
                        save_index,
                        queries,
                        groundtruth,
                        top_k,
                        nprobe,
                        benchmark_mode,
                        max_base,
                        max_queries,
                        threads,
                        warmup_queries,
                    })
                }
            }
            Mode::Query => Ok(Self {
                mode,
                base: PathBuf::new(),
                centroids: None,
                assignments: None,
                nlist: None,
                bits: 0,
                metric,
                seed,
                load_index,
                save_index: None,
                queries,
                groundtruth,
                top_k,
                nprobe,
                benchmark_mode,
                max_base: None,
                max_queries,
                threads,
                warmup_queries,
            }),
        }
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
    eprintln!("ivf_rabitq - IVF+RaBitQ index builder and query evaluator\n");
    eprintln!("USAGE:");
    eprintln!("  # Build index from scratch");
    eprintln!("  ivf_rabitq --base <data.fvecs> --nlist <K> --bits <B> --save <index>\n");
    eprintln!("  # Build with pre-computed clusters");
    eprintln!("  ivf_rabitq --base <data.fvecs> --centroids <C> --assignments <A> --bits <B> --save <index>\n");
    eprintln!("  # Query existing index");
    eprintln!("  ivf_rabitq --load <index> --queries <Q> --gt <G> [--benchmark]\n");
    eprintln!("  # Build and query in one go");
    eprintln!("  ivf_rabitq --base <data.fvecs> --nlist <K> --bits <B> --queries <Q> --gt <G>\n");

    eprintln!("INDEX BUILDING:");
    eprintln!("  --base <path>         Base vectors (.fvecs)");
    eprintln!("  --bits <1-16>         Quantization bits (required for new index)");
    eprintln!("  --nlist <K>           Number of clusters (built-in k-means)");
    eprintln!("  --centroids <path>    Pre-computed centroids (.fvecs)");
    eprintln!("  --assignments <path>  Pre-computed cluster assignments (.ivecs)");
    eprintln!("  --metric <l2|ip>      Distance metric (default: l2)");
    eprintln!("  --seed <N>            Random seed for rotation (default: 1515870004)");
    eprintln!("  --save <path>         Save index to file\n");

    eprintln!("QUERYING:");
    eprintln!("  --load <path>         Load index from file");
    eprintln!("  --queries <path>      Query vectors (.fvecs)");
    eprintln!("  --gt <path>           Ground truth (.ivecs)");
    eprintln!("  --top-k <N>           Number of neighbors to retrieve (default: 100)");
    eprintln!("  --nprobe <N>          Clusters to probe (default: 64, ignored with --benchmark)");
    eprintln!("  --benchmark           Run nprobe sweep + multi-round benchmark (like C++)\n");

    eprintln!("LIMITS & TUNING:");
    eprintln!("  --max-base <N>        Limit base vectors loaded");
    eprintln!("  --max-queries <N>     Limit query vectors loaded");
    eprintln!("  --threads <N>         Parallel search threads");
    eprintln!("  --warmup <N>          Warmup queries (default: 10, use 0 to disable)\n");

    eprintln!("EXAMPLES:");
    eprintln!("  # Typical workflow with FAISS clustering");
    eprintln!("  python python/ivf.py data/gist_base.fvecs 4096 centroids.fvecs clusters.ivecs l2");
    eprintln!("  ivf_rabitq --base data/gist_base.fvecs --centroids centroids.fvecs \\");
    eprintln!("             --assignments clusters.ivecs --bits 3 --save index.bin");
    eprintln!("  ivf_rabitq --load index.bin --queries data/gist_query.fvecs \\");
    eprintln!("             --gt data/gist_groundtruth.ivecs --benchmark\n");
}

fn install_signal_handlers() -> std::io::Result<()> {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    std::thread::spawn(move || {
        if let Some(signal) = signals.forever().next() {
            eprintln!("\nReceived signal {}; dumping stack trace...", signal);
            let backtrace = Backtrace::force_capture();
            eprintln!("{}", backtrace);
            eprintln!("Exiting due to signal {}.", signal);
            process::exit(130);
        }
    });

    Ok(())
}
