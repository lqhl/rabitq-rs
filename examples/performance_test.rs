use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("IVF-RaBitQ Performance Test - Optimized Version");
    println!("================================================\n");

    // Generate synthetic test data
    let dim = 960;
    let base_size = 100_000;
    let query_size = 100;
    let nlist = 4096;
    let nprobe = 64;
    let top_k = 100;
    let total_bits = 7;

    println!("Configuration:");
    println!("  Dimensions: {}", dim);
    println!("  Base vectors: {}", base_size);
    println!("  Query vectors: {}", query_size);
    println!("  IVF clusters: {}", nlist);
    println!("  Probe clusters: {}", nprobe);
    println!("  Top-K: {}", top_k);
    println!("  Bits: {}\n", total_bits);

    // Generate random data
    use rand::distributions::{Distribution, Standard};
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    println!("Generating random data...");
    let base_vecs: Vec<Vec<f32>> = (0..base_size)
        .map(|_| Standard.sample_iter(&mut rng).take(dim).collect())
        .collect();
    let query_vecs: Vec<f32> = Standard
        .sample_iter(&mut rng)
        .take(query_size * dim)
        .collect();

    // Build index
    println!("Building IVF-RaBitQ index (bits={})...", total_bits);

    let train_start = Instant::now();
    let index = IvfRabitqIndex::train(
        &base_vecs,
        nlist,
        total_bits,
        Metric::L2,
        RotatorType::MatrixRotator,
        42,
        false,
    )?;
    let train_time = train_start.elapsed();
    println!("  Training time: {:.2}s", train_time.as_secs_f64());

    // Search test - warmup
    println!("\nWarming up...");
    let params = SearchParams { nprobe, top_k };

    for i in 0..10 {
        let query = &query_vecs[i * dim..(i + 1) * dim];
        let _ = index.search(query, params);
    }

    // Search test - performance measurement
    println!("Running search benchmark...");
    let search_start = Instant::now();
    let mut total_results = 0;

    for i in 0..query_size {
        let query = &query_vecs[i * dim..(i + 1) * dim];
        let results = index.search(query, params)?;
        total_results += results.len();
    }

    let search_time = search_start.elapsed();
    let qps = query_size as f64 / search_time.as_secs_f64();
    let avg_latency_ms = search_time.as_millis() as f64 / query_size as f64;

    println!("\n=== Performance Results ===");
    println!("  Total queries: {}", query_size);
    println!("  Total time: {:.3}s", search_time.as_secs_f64());
    println!("  Average latency: {:.2}ms", avg_latency_ms);
    println!("  QPS: {:.0}", qps);
    println!(
        "  Average results per query: {:.1}",
        total_results as f64 / query_size as f64
    );

    Ok(())
}
