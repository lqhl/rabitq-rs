use std::collections::HashSet;
use std::env;
use std::io::{Error as IoError, ErrorKind as IoErrorKind};
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};

use rabitq::io::{read_fvecs, read_groundtruth, read_ids};
use rabitq::ivf::{IvfRabitqIndex, SearchParams};
use rabitq::Metric;

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
            eprintln!("{message}");
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
        return Err(Box::new(IoError::new(
            IoErrorKind::InvalidInput,
            "no base vectors were loaded",
        )));
    }
    let dim = base[0].len();
    if base.iter().any(|v| v.len() != dim) {
        return Err(Box::new(IoError::new(
            IoErrorKind::InvalidInput,
            "base vectors have inconsistent dimensionality",
        )));
    }

    println!("Loaded {} base vectors with dimension {}.", base.len(), dim);

    println!("Loading centroids from {}...", config.centroids.display());
    let centroids = read_fvecs(&config.centroids, None)?;
    if centroids.is_empty() {
        return Err(Box::new(IoError::new(
            IoErrorKind::InvalidInput,
            "no centroids were loaded",
        )));
    }
    if centroids.iter().any(|c| c.len() != dim) {
        return Err(Box::new(IoError::new(
            IoErrorKind::InvalidInput,
            "centroid dimensionality does not match the base dataset",
        )));
    }

    println!(
        "Loading cluster assignments from {}...",
        config.assignments.display()
    );
    let assignments = read_ids(&config.assignments, config.max_base)?;
    if assignments.len() != base.len() {
        return Err(Box::new(IoError::new(
            IoErrorKind::InvalidInput,
            format!(
                "cluster assignment count ({}) does not match number of base vectors ({})",
                assignments.len(),
                base.len()
            ),
        )));
    }

    let build_start = Instant::now();
    let index = IvfRabitqIndex::train_with_clusters(
        &base,
        &centroids,
        &assignments,
        config.bits,
        config.metric,
        config.seed,
    )?;
    let build_time = build_start.elapsed();

    println!(
        "Constructed IVF+RaBitQ index in {:.2?} ({} vectors across {} clusters).",
        build_time,
        index.len(),
        centroids.len()
    );

    if let (Some(query_path), Some(gt_path)) = (&config.queries, &config.groundtruth) {
        println!("Loading queries from {}...", query_path.display());
        let queries = read_fvecs(query_path, config.max_queries)?;
        if queries.is_empty() {
            return Err(Box::new(IoError::new(
                IoErrorKind::InvalidInput,
                "no query vectors were loaded",
            )));
        }
        if queries.iter().any(|q| q.len() != dim) {
            return Err(Box::new(IoError::new(
                IoErrorKind::InvalidInput,
                "query dimensionality does not match the base dataset",
            )));
        }

        println!("Loading ground truth from {}...", gt_path.display());
        let groundtruth = read_groundtruth(gt_path, config.max_queries)?;
        if groundtruth.len() != queries.len() {
            return Err(Box::new(IoError::new(
                IoErrorKind::InvalidInput,
                "ground truth entries do not match the number of queries",
            )));
        }

        evaluate_search(&index, &queries, &groundtruth, &config)?;
    } else {
        println!("Skipping query evaluation (no queries/ground truth provided).");
    }

    Ok(())
}

fn evaluate_search(
    index: &IvfRabitqIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<usize>],
    config: &Config,
) -> CliResult<()> {
    let params = SearchParams::new(config.top_k, config.nprobe);
    let mut total_hits = 0usize;
    let mut total_possible = 0usize;
    let mut total_time = Duration::default();

    println!(
        "Evaluating {} queries with top_k={} and nprobe={}...",
        queries.len(),
        config.top_k,
        config.nprobe
    );

    for (idx, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(query, params)?;
        total_time += start.elapsed();

        let gt = &groundtruth[idx];
        if gt.is_empty() {
            continue;
        }
        let limit = config.top_k.min(gt.len());
        total_possible += limit;
        let gt_slice: HashSet<usize> = gt[..limit].iter().copied().collect();

        for candidate in results.iter().take(limit) {
            if gt_slice.contains(&candidate.id) {
                total_hits += 1;
            }
        }
    }

    let recall = if total_possible > 0 {
        total_hits as f32 / total_possible as f32
    } else {
        0.0
    };
    let total_seconds = total_time.as_secs_f64();
    let qps = if total_seconds > f64::EPSILON {
        queries.len() as f64 / total_seconds
    } else {
        f64::INFINITY
    };

    println!("Recall@{} = {:.4}, QPS = {:.2}", config.top_k, recall, qps);

    Ok(())
}

#[derive(Debug, Clone)]
struct Config {
    base: PathBuf,
    centroids: PathBuf,
    assignments: PathBuf,
    bits: usize,
    metric: Metric,
    queries: Option<PathBuf>,
    groundtruth: Option<PathBuf>,
    top_k: usize,
    nprobe: usize,
    max_base: Option<usize>,
    max_queries: Option<usize>,
    seed: u64,
}

impl Config {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut base = None;
        let mut centroids = None;
        let mut assignments = None;
        let mut bits = None;
        let mut metric = Metric::L2;
        let mut queries = None;
        let mut groundtruth = None;
        let mut top_k = 10usize;
        let mut nprobe = 64usize;
        let mut max_base = None;
        let mut max_queries = None;
        let mut seed = 0x5a5a_1234_u64;

        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--base" => base = Some(next_path(&mut iter, &arg)?),
                "--centroids" => centroids = Some(next_path(&mut iter, &arg)?),
                "--assignments" => assignments = Some(next_path(&mut iter, &arg)?),
                "--bits" => bits = Some(next_usize(&mut iter, &arg)?),
                "--metric" => {
                    let value = next_value(&mut iter, &arg)?;
                    metric = parse_metric(&value)?;
                }
                "--queries" => queries = Some(next_path(&mut iter, &arg)?),
                "--groundtruth" => groundtruth = Some(next_path(&mut iter, &arg)?),
                "--top-k" | "--topk" => top_k = next_usize(&mut iter, &arg)?,
                "--nprobe" => nprobe = next_usize(&mut iter, &arg)?,
                "--max-base" => max_base = Some(next_usize(&mut iter, &arg)?),
                "--max-queries" => max_queries = Some(next_usize(&mut iter, &arg)?),
                "--seed" => seed = next_u64(&mut iter, &arg)?,
                other => {
                    return Err(format!("unrecognised argument: {other}"));
                }
            }
        }

        let base = base.ok_or_else(|| "missing required argument --base".to_string())?;
        let centroids =
            centroids.ok_or_else(|| "missing required argument --centroids".to_string())?;
        let assignments =
            assignments.ok_or_else(|| "missing required argument --assignments".to_string())?;
        let bits = bits.ok_or_else(|| "missing required argument --bits".to_string())?;

        if bits == 0 || bits > 16 {
            return Err("--bits must be between 1 and 16".to_string());
        }
        if top_k == 0 {
            return Err("--top-k must be positive".to_string());
        }
        if nprobe == 0 {
            return Err("--nprobe must be positive".to_string());
        }
        if queries.is_some() ^ groundtruth.is_some() {
            return Err("--queries and --groundtruth must be provided together".to_string());
        }

        Ok(Self {
            base,
            centroids,
            assignments,
            bits,
            metric,
            queries,
            groundtruth,
            top_k,
            nprobe,
            max_base,
            max_queries,
            seed,
        })
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(next_value(iter, flag)?))
}

fn next_usize(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let value = next_value(iter, flag)?;
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn next_u64(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
    let value = next_value(iter, flag)?;
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn parse_metric(value: &str) -> Result<Metric, String> {
    match value.to_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "ip" | "innerproduct" | "inner_product" => Ok(Metric::InnerProduct),
        other => Err(format!("unsupported metric: {other}")),
    }
}

fn print_usage() {
    eprintln!("Usage: cargo run --bin gist -- [OPTIONS]");
    eprintln!("\nRequired arguments:");
    eprintln!("    --base <path>         Path to gist_base.fvecs");
    eprintln!("    --centroids <path>    Path to centroid .fvecs file");
    eprintln!("    --assignments <path>  Path to cluster id .ivecs file");
    eprintln!("    --bits <value>        Total number of quantisation bits (1-16)");
    eprintln!("\nOptional arguments:");
    eprintln!("    --metric <l2|ip>      Distance metric (default: l2)");
    eprintln!("    --queries <path>      Path to gist_query.fvecs");
    eprintln!("    --groundtruth <path>  Path to gist_groundtruth.ivecs");
    eprintln!("    --top-k <value>       Number of neighbours to retrieve (default: 10)");
    eprintln!("    --nprobe <value>      Number of IVF clusters to probe (default: 64)");
    eprintln!("    --max-base <value>    Limit the number of base vectors loaded");
    eprintln!("    --max-queries <value> Limit the number of query vectors loaded");
    eprintln!("    --seed <value>        Random seed for the rotator");
}
