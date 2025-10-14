/// K-Means Benchmark: Compare rabitq-rs vs kentro implementations
///
/// Usage:
///   cargo run --release --example kmeans_bench -- \
///     --data data/gist/gist_base.fvecs \
///     --nlist 4096 \
///     --niter 25 \
///     --max-points 100000

extern crate kentro;
extern crate ndarray;

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    println!("K-Means Benchmark Configuration");
    println!("  Data: {}", args.data_path);
    println!("  Clusters (nlist): {}", args.nlist);
    println!("  Iterations: {}", args.niter);
    println!("  Max points: {}", args.max_points.unwrap_or(usize::MAX));
    println!();

    // Load data
    println!("Loading data...");
    let data = rabitq_rs::io::read_fvecs(&args.data_path, args.max_points)?;
    println!("  Loaded {} vectors of dimension {}", data.len(), data[0].len());
    println!();

    // Benchmark rabitq-rs k-means
    println!("=== Benchmark 1: rabitq-rs (internal k-means) ===");
    let rabitq_result = benchmark_rabitq_kmeans(&data, args.nlist, args.niter, args.seed)?;
    println!();

    // Benchmark kentro k-means
    println!("=== Benchmark 2: kentro k-means ===");
    let kentro_result = benchmark_kentro_kmeans(&data, args.nlist, args.niter)?;
    println!();

    // Summary
    println!("=== Summary ===");
    println!("rabitq-rs:");
    println!("  Time: {:.3}s", rabitq_result.time);
    println!("  Objective: {:.3e}", rabitq_result.objective);
    println!();
    println!("kentro:");
    println!("  Time: {:.3}s", kentro_result.time);
    println!();
    println!("Speedup: {:.2}x {}",
        rabitq_result.time / kentro_result.time,
        if rabitq_result.time < kentro_result.time { "(rabitq-rs faster)" } else { "(kentro faster)" }
    );

    Ok(())
}

struct BenchResult {
    time: f64,
    objective: f64,
    assignments: Vec<usize>,
}

struct ClusterStats {
    min_size: usize,
    max_size: usize,
    mean_size: f64,
    median_size: usize,
    std_dev: f64,
    empty_count: usize,
    imbalance_ratio: f64,
}

fn benchmark_rabitq_kmeans(
    data: &[Vec<f32>],
    nlist: usize,
    niter: usize,
    seed: u64,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    use rabitq_rs::kmeans::{run_kmeans_with_config, KMeansConfig};

    let config = KMeansConfig {
        niter,
        nredo: 1,
        seed,
        spherical: false,
        max_points_per_centroid: 256,
        decode_block_size: 32768,
    };

    let start = Instant::now();
    let result = run_kmeans_with_config(data, nlist, config);
    let elapsed = start.elapsed().as_secs_f64();

    println!("  Time: {:.3}s", elapsed);
    println!("  Objective: {:.3e}", result.objective);

    let stats = analyze_cluster_balance(&result.assignments, nlist);
    print_cluster_stats(&stats);

    Ok(BenchResult {
        time: elapsed,
        objective: result.objective,
        assignments: result.assignments,
    })
}

fn benchmark_kentro_kmeans(
    data: &[Vec<f32>],
    nlist: usize,
    niter: usize,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    // Convert data to ndarray format (flatten Vec<Vec<f32>> to 2D array)
    let n_points = data.len();
    let dim = data[0].len();
    let mut flat_data = Vec::with_capacity(n_points * dim);
    for vec in data {
        flat_data.extend_from_slice(vec);
    }

    let data_array = ndarray::Array2::from_shape_vec((n_points, dim), flat_data)?;

    // Configure kentro k-means
    let mut kmeans = kentro::KMeans::new(nlist)
        .with_iterations(niter)
        .with_euclidean(true)
        .with_verbose(false);

    let start = Instant::now();
    let _clusters = kmeans.train(data_array.view(), None)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("  Time: {:.3}s", elapsed);

    Ok(BenchResult {
        time: elapsed,
        objective: 0.0,  // Not computing objective for kentro to avoid version issues
        assignments: vec![], // kentro doesn't expose assignments easily
    })
}

fn analyze_cluster_balance(assignments: &[usize], nlist: usize) -> ClusterStats {
    let mut cluster_sizes = vec![0usize; nlist];
    for &cluster_id in assignments {
        cluster_sizes[cluster_id] += 1;
    }

    let non_empty_sizes: Vec<usize> = cluster_sizes.iter().copied().filter(|&s| s > 0).collect();
    let empty_count = cluster_sizes.iter().filter(|&&s| s == 0).count();

    let min_size = *non_empty_sizes.iter().min().unwrap_or(&0);
    let max_size = *non_empty_sizes.iter().max().unwrap_or(&0);
    let sum: usize = non_empty_sizes.iter().sum();
    let mean_size = if !non_empty_sizes.is_empty() {
        sum as f64 / non_empty_sizes.len() as f64
    } else {
        0.0
    };

    let mut sorted_sizes = non_empty_sizes.clone();
    sorted_sizes.sort_unstable();
    let median_size = if !sorted_sizes.is_empty() {
        sorted_sizes[sorted_sizes.len() / 2]
    } else {
        0
    };

    let variance = if !non_empty_sizes.is_empty() {
        non_empty_sizes
            .iter()
            .map(|&size| {
                let diff = size as f64 - mean_size;
                diff * diff
            })
            .sum::<f64>()
            / non_empty_sizes.len() as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let imbalance_ratio = if mean_size > 0.0 {
        max_size as f64 / mean_size
    } else {
        0.0
    };

    ClusterStats {
        min_size,
        max_size,
        mean_size,
        median_size,
        std_dev,
        empty_count,
        imbalance_ratio,
    }
}

fn print_cluster_stats(stats: &ClusterStats) {
    println!("  Cluster Balance:");
    println!("    Empty clusters: {}", stats.empty_count);
    println!("    Min size: {}", stats.min_size);
    println!("    Max size: {}", stats.max_size);
    println!("    Mean size: {:.1}", stats.mean_size);
    println!("    Median size: {}", stats.median_size);
    println!("    Std dev: {:.1}", stats.std_dev);
    println!("    Imbalance ratio (max/mean): {:.2}x", stats.imbalance_ratio);
}

struct Args {
    data_path: String,
    nlist: usize,
    niter: usize,
    seed: u64,
    max_points: Option<usize>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut data_path = String::from("data/gist/gist_base.fvecs");
    let mut nlist = 4096;
    let mut niter = 25;
    let mut seed = 42;
    let mut max_points = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                data_path = args[i + 1].clone();
                i += 2;
            }
            "--nlist" => {
                nlist = args[i + 1].parse().expect("Invalid nlist");
                i += 2;
            }
            "--niter" => {
                niter = args[i + 1].parse().expect("Invalid niter");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("Invalid seed");
                i += 2;
            }
            "--max-points" => {
                max_points = Some(args[i + 1].parse().expect("Invalid max-points"));
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    Args {
        data_path,
        nlist,
        niter,
        seed,
        max_points,
    }
}
