//! Quick benchmark comparing MSTG vs IVF with optimized parameters
//!
//! Usage:
//!   cargo run --release --example benchmark_quick

use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams as MstgSearchParams};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use rand::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

fn generate_test_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn compute_recall(retrieved: &[usize], ground_truth: &[usize]) -> f32 {
    let gt_set: HashSet<_> = ground_truth.iter().copied().collect();
    let matches = retrieved.iter().filter(|id| gt_set.contains(id)).count();
    matches as f32 / ground_truth.len() as f32
}

fn brute_force_search(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, vec)| {
            let dist: f32 = query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().take(k).map(|(i, _)| *i).collect()
}

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║         MSTG vs IVF Quick Benchmark                       ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Test with larger dataset
    let n_base = 100_000;
    let n_queries = 100;
    let dim = 256;

    println!("\n📊 Configuration:");
    println!("  • Vectors:    {:>8}", n_base);
    println!("  • Queries:    {:>8}", n_queries);
    println!("  • Dimensions: {:>8}", dim);

    println!("\n🔄 Generating data...");
    let data = generate_test_data(n_base, dim, 42);
    let queries = generate_test_data(n_queries, dim, 123);

    println!("🔍 Computing ground truth...");
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_search(&data, q, 100))
        .collect();

    // ==================== MSTG ====================
    println!("\n[MSTG] Building index...");
    let mstg_config = MstgConfig {
        max_posting_size: 1000, // Larger for better batching with FastScan
        branching_factor: 8,
        balance_weight: 1.0,
        closure_epsilon: 0.3,
        max_replicas: 5,
        rabitq_bits: 7,
        faster_config: true,
        metric: Metric::L2,
        hnsw_m: 32,
        hnsw_ef_construction: 200,
        centroid_precision: rabitq_rs::mstg::ScalarPrecision::BF16,
        default_ef_search: 200,
        pruning_epsilon: 1.5,
    };

    let start = Instant::now();
    let mstg_index = MstgIndex::build(&data, mstg_config).expect("MSTG build failed");
    let mstg_build_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("[MSTG] Warmup...");
    let mstg_params = MstgSearchParams {
        ef_search: 200,
        pruning_epsilon: 1.5,
        top_k: 100,
    };
    for q in queries.iter().take(10) {
        let _ = mstg_index.search(q, &mstg_params);
    }

    println!("[MSTG] Searching...");
    let start = Instant::now();
    let mut mstg_results = Vec::new();
    for q in queries.iter() {
        let results = mstg_index.search(q, &mstg_params);
        mstg_results.push(results.iter().map(|r| r.vector_id).collect::<Vec<_>>());
    }
    let mstg_search_us = start.elapsed().as_secs_f64() * 1_000_000.0 / queries.len() as f64;

    let mstg_recall: f32 = mstg_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(r, gt)| compute_recall(r, gt))
        .sum::<f32>()
        / queries.len() as f32;

    // ==================== IVF ====================
    println!("\n[IVF] Building index...");
    let nlist = (n_base as f32).sqrt() as usize;

    let start = Instant::now();
    let ivf_index = IvfRabitqIndex::train(
        &data,
        nlist,
        7, // 7 bits
        Metric::L2,
        RotatorType::FhtKacRotator,
        42,
        true, // faster_config
    )
    .expect("IVF train failed");
    let ivf_build_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Find nprobe for similar recall
    println!("[IVF] Tuning nprobe for similar recall...");
    let mut best_nprobe = nlist / 20;
    let mut best_recall_diff = f32::MAX;

    for nprobe_candidate in [nlist / 40, nlist / 20, nlist / 10, nlist / 5] {
        let ivf_params = SearchParams {
            nprobe: nprobe_candidate,
            top_k: 100,
        };

        let test_results: Vec<Vec<usize>> = queries
            .iter()
            .take(10)
            .map(|q| {
                ivf_index
                    .search(q, ivf_params)
                    .unwrap()
                    .iter()
                    .map(|r| r.id)
                    .collect()
            })
            .collect();

        let test_recall: f32 = test_results
            .iter()
            .zip(ground_truth.iter().take(10))
            .map(|(r, gt)| compute_recall(r, gt))
            .sum::<f32>()
            / 10.0;

        let diff = (test_recall - mstg_recall).abs();
        if diff < best_recall_diff {
            best_recall_diff = diff;
            best_nprobe = nprobe_candidate;
        }
    }

    println!(
        "[IVF] Using nprobe={} (targeting recall ~{}%)",
        best_nprobe,
        (mstg_recall * 100.0) as i32
    );

    println!("[IVF] Warmup...");
    let ivf_params = SearchParams {
        nprobe: best_nprobe,
        top_k: 100,
    };
    for q in queries.iter().take(10) {
        let _ = ivf_index.search(q, ivf_params);
    }

    println!("[IVF] Searching...");
    let start = Instant::now();
    let mut ivf_results = Vec::new();
    for q in queries.iter() {
        let results = ivf_index.search(q, ivf_params).unwrap();
        ivf_results.push(results.iter().map(|r| r.id).collect::<Vec<_>>());
    }
    let ivf_search_us = start.elapsed().as_secs_f64() * 1_000_000.0 / queries.len() as f64;

    let ivf_recall: f32 = ivf_results
        .iter()
        .zip(ground_truth.iter())
        .map(|(r, gt)| compute_recall(r, gt))
        .sum::<f32>()
        / queries.len() as f32;

    // ==================== RESULTS ====================
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS                                 ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\n📦 MSTG:");
    println!("  Build time:    {:>8.1} ms", mstg_build_ms);
    println!(
        "  Search time:   {:>8.1} µs  ({:.1} QPS)",
        mstg_search_us,
        1_000_000.0 / mstg_search_us
    );
    println!("  Recall@100:    {:>7.1}%", mstg_recall * 100.0);

    println!("\n📦 IVF (nprobe={}):", best_nprobe);
    println!("  Build time:    {:>8.1} ms", ivf_build_ms);
    println!(
        "  Search time:   {:>8.1} µs  ({:.1} QPS)",
        ivf_search_us,
        1_000_000.0 / ivf_search_us
    );
    println!("  Recall@100:    {:>7.1}%", ivf_recall * 100.0);

    println!("\n🔥 Performance:");
    if mstg_search_us < ivf_search_us {
        println!("  MSTG is {:.2}x FASTER", ivf_search_us / mstg_search_us);
    } else {
        println!("  IVF is {:.2}x FASTER", mstg_search_us / ivf_search_us);
    }

    if mstg_build_ms < ivf_build_ms {
        println!("  MSTG builds {:.2}x FASTER", ivf_build_ms / mstg_build_ms);
    } else {
        println!("  IVF builds {:.2}x FASTER", mstg_build_ms / ivf_build_ms);
    }

    println!("\n✅ Done!");
}
