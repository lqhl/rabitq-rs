use std::backtrace::Backtrace;
use std::collections::HashSet;
use std::env;
use std::io::{Error as IoError, ErrorKind as IoErrorKind};
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};

use rabitq_rs::io::{read_fvecs, read_groundtruth, read_ids};
use rabitq_rs::ivf::{IvfRabitqIndex, SearchParams};
use rabitq_rs::Metric;

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

    let build_start = Instant::now();
    let mut loaded_from_disk = false;
    let index = if let Some(load_path) = &config.load_index {
        println!("Loading IVF+RaBitQ index from {}...", load_path.display());
        loaded_from_disk = true;
        IvfRabitqIndex::load_from_path(load_path)?
    } else if let (Some(centroids_path), Some(assignments_path)) =
        (&config.centroids, &config.assignments)
    {
        println!("Loading centroids from {}...", centroids_path.display());
        let centroids = read_fvecs(centroids_path, None)?;
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
            assignments_path.display()
        );
        println!("  This may take a while; progress updates will be printed periodically.");
        let assignments = read_ids(assignments_path, config.max_base)?;
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

        println!("Loaded {} cluster assignments.", assignments.len());

        IvfRabitqIndex::train_with_clusters(
            &base,
            &centroids,
            &assignments,
            config.bits,
            config.metric,
            config.seed,
        )?
    } else {
        let nlist = config
            .nlist
            .ok_or_else(|| IoError::new(IoErrorKind::InvalidInput, "missing nlist"))?;
        println!(
            "Clustering base vectors into {} lists with the built-in Rust trainer...",
            nlist
        );
        IvfRabitqIndex::train(&base, nlist, config.bits, config.metric, config.seed)?
    };
    let build_time = build_start.elapsed();

    if let Some(save_path) = &config.save_index {
        println!("Persisting index to {}...", save_path.display());
        index.save_to_path(save_path)?;
    }

    let verb = if loaded_from_disk {
        "Loaded"
    } else {
        "Constructed"
    };
    println!(
        "{} IVF+RaBitQ index in {:.2?} ({} vectors across {} clusters).",
        verb,
        build_time,
        index.len(),
        index.cluster_count()
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
    centroids: Option<PathBuf>,
    assignments: Option<PathBuf>,
    nlist: Option<usize>,
    bits: usize,
    metric: Metric,
    queries: Option<PathBuf>,
    groundtruth: Option<PathBuf>,
    top_k: usize,
    nprobe: usize,
    max_base: Option<usize>,
    max_queries: Option<usize>,
    seed: u64,
    load_index: Option<PathBuf>,
    save_index: Option<PathBuf>,
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
        let mut top_k = 10usize;
        let mut nprobe = 64usize;
        let mut max_base = None;
        let mut max_queries = None;
        let mut seed = 0x5a5a_1234_u64;
        let mut load_index = None;
        let mut save_index = None;

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
                "--groundtruth" => groundtruth = Some(next_path(&mut iter, &arg)?),
                "--top-k" | "--topk" => top_k = next_usize(&mut iter, &arg)?,
                "--nprobe" => nprobe = next_usize(&mut iter, &arg)?,
                "--max-base" => max_base = Some(next_usize(&mut iter, &arg)?),
                "--max-queries" => max_queries = Some(next_usize(&mut iter, &arg)?),
                "--seed" => seed = next_u64(&mut iter, &arg)?,
                "--load-index" => load_index = Some(next_path(&mut iter, &arg)?),
                "--save-index" => save_index = Some(next_path(&mut iter, &arg)?),
                other => {
                    return Err(format!("unrecognised argument: {other}"));
                }
            }
        }

        let base = base.ok_or_else(|| "missing required argument --base".to_string())?;
        let bits = match (load_index.as_ref(), bits) {
            (Some(_), Some(value)) => value,
            (Some(_), None) => 0,
            (None, Some(value)) => value,
            (None, None) => return Err("missing required argument --bits".to_string()),
        };

        if load_index.is_none() {
            if bits == 0 || bits > 16 {
                return Err("--bits must be between 1 and 16".to_string());
            }
        } else if bits != 0 && (bits > 16) {
            return Err("--bits must be between 1 and 16 when provided".to_string());
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
        if centroids.is_some() ^ assignments.is_some() {
            return Err("--centroids and --assignments must be provided together".to_string());
        }
        if load_index.is_some() {
            if centroids.is_some() || assignments.is_some() || nlist.is_some() {
                return Err(
                    "--load-index cannot be combined with training arguments (--nlist/--centroids/--assignments)"
                        .to_string(),
                );
            }
        } else if centroids.is_none() && assignments.is_none() && nlist.is_none() {
            return Err(
                "provide either --nlist for built-in training or both --centroids/--assignments"
                    .to_string(),
            );
        }

        Ok(Self {
            base,
            centroids,
            assignments,
            nlist,
            bits,
            metric,
            queries,
            groundtruth,
            top_k,
            nprobe,
            max_base,
            max_queries,
            seed,
            load_index,
            save_index,
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
    eprintln!(
        "    --bits <value>        Total number of quantisation bits (1-16, required unless --load-index)"
    );
    eprintln!("\nChoose one of the following training modes:");
    eprintln!("    --nlist <value>       Number of IVF clusters to learn with the Rust trainer");
    eprintln!("    --centroids <path>    Path to centroid .fvecs file");
    eprintln!("    --assignments <path>  Path to cluster id .ivecs file");
    eprintln!("\nOptional arguments:");
    eprintln!("    --metric <l2|ip>      Distance metric (default: l2)");
    eprintln!("    --queries <path>      Path to gist_query.fvecs");
    eprintln!("    --groundtruth <path>  Path to gist_groundtruth.ivecs");
    eprintln!("    --top-k <value>       Number of neighbours to retrieve (default: 10)");
    eprintln!("    --nprobe <value>      Number of IVF clusters to probe (default: 64)");
    eprintln!("    --max-base <value>    Limit the number of base vectors loaded");
    eprintln!("    --max-queries <value> Limit the number of query vectors loaded");
    eprintln!("    --seed <value>        Random seed for the rotator");
    eprintln!("    --save-index <path>   Persist the trained index to disk");
    eprintln!("    --load-index <path>   Load a persisted index instead of retraining");
}

fn install_signal_handlers() -> std::io::Result<()> {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    std::thread::spawn(move || {
        if let Some(signal) = signals.forever().next() {
            eprintln!("Received signal {signal}; dumping stack trace...");
            let backtrace = Backtrace::force_capture();
            eprintln!("{backtrace}");
            eprintln!("Exiting due to signal {signal}.");
            process::exit(130);
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Config;
    use rabitq_rs::Metric;

    #[test]
    fn parse_config_with_precomputed_clusters() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--centroids".into(),
            "centroids.fvecs".into(),
            "--assignments".into(),
            "assignments.ivecs".into(),
            "--bits".into(),
            "7".into(),
            "--metric".into(),
            "ip".into(),
        ];
        let config = Config::parse(args).expect("parse config");
        assert_eq!(config.base, std::path::PathBuf::from("base.fvecs"));
        assert_eq!(
            config.centroids,
            Some(std::path::PathBuf::from("centroids.fvecs"))
        );
        assert_eq!(
            config.assignments,
            Some(std::path::PathBuf::from("assignments.ivecs"))
        );
        assert_eq!(config.nlist, None);
        assert_eq!(config.bits, 7);
        assert_eq!(config.metric, Metric::InnerProduct);
        assert!(config.load_index.is_none());
        assert!(config.save_index.is_none());
    }

    #[test]
    fn parse_config_with_built_in_training() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--bits".into(),
            "6".into(),
            "--nlist".into(),
            "128".into(),
            "--top-k".into(),
            "20".into(),
        ];
        let config = Config::parse(args).expect("parse config");
        assert_eq!(config.base, std::path::PathBuf::from("base.fvecs"));
        assert_eq!(config.centroids, None);
        assert_eq!(config.assignments, None);
        assert_eq!(config.nlist, Some(128));
        assert_eq!(config.bits, 6);
        assert_eq!(config.top_k, 20);
        assert!(config.load_index.is_none());
        assert!(config.save_index.is_none());
    }

    #[test]
    fn parse_config_requires_training_mode() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--bits".into(),
            "7".into(),
        ];
        let err = Config::parse(args).expect_err("config should fail without training mode");
        assert!(err.contains("provide either --nlist"));
    }

    #[test]
    fn parse_config_rejects_partial_precomputed_arguments() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--bits".into(),
            "7".into(),
            "--centroids".into(),
            "centroids.fvecs".into(),
        ];
        let err = Config::parse(args).expect_err("missing assignments should error");
        assert!(err.contains("--centroids and --assignments must be provided together"));
    }

    #[test]
    fn parse_config_accepts_load_index_without_bits() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--bits".into(),
            "7".into(),
            "--load-index".into(),
            "index.bin".into(),
        ];
        let config = Config::parse(args).expect("parse config");
        assert_eq!(config.bits, 7);
        assert_eq!(
            config.load_index,
            Some(std::path::PathBuf::from("index.bin"))
        );
    }

    #[test]
    fn parse_config_allows_load_index_without_bits_flag() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--load-index".into(),
            "index.bin".into(),
            "--top-k".into(),
            "15".into(),
        ];
        let config = Config::parse(args).expect("parse config");
        assert_eq!(config.bits, 0);
        assert_eq!(config.top_k, 15);
        assert_eq!(
            config.load_index,
            Some(std::path::PathBuf::from("index.bin"))
        );
    }

    #[test]
    fn parse_config_rejects_load_index_with_training_args() {
        let args = vec![
            "--base".into(),
            "base.fvecs".into(),
            "--load-index".into(),
            "index.bin".into(),
            "--bits".into(),
            "7".into(),
            "--nlist".into(),
            "32".into(),
        ];
        let err = Config::parse(args).expect_err("config should reject mixed options");
        assert!(err.contains("--load-index cannot be combined"));
    }
}
