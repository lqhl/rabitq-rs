use std::io::Cursor;

use rand::prelude::*;

use crate::io::{
    read_fvecs_from_reader, read_groundtruth_from_reader, read_ids_from_reader,
    read_ivecs_from_reader,
};
use crate::ivf::{IvfRabitqIndex, SearchParams};
use crate::kmeans::run_kmeans;
use crate::math::l2_distance_sqr;
use crate::quantizer::{quantize_with_centroid, reconstruct_into};
use crate::rotation::RandomRotator;
use crate::Metric;

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

#[test]
fn quantizer_reconstruction_is_reasonable() {
    let dim = 64;
    let mut rng = StdRng::seed_from_u64(1234);
    let data = random_vector(dim, &mut rng);
    let centroid = vec![0.0f32; dim];
    let quantized = quantize_with_centroid(&data, &centroid, 7, Metric::L2);
    assert_eq!(quantized.code.len(), dim, "quantized code length mismatch");
    assert_eq!(
        quantized.ex_code.len(),
        dim,
        "extended code length mismatch"
    );
    assert_eq!(
        quantized.binary_code.len(),
        dim,
        "binary code length mismatch"
    );
    let mut reconstructed = vec![0.0f32; dim];
    reconstruct_into(&centroid, &quantized, &mut reconstructed);

    let mut diff_sqr = 0.0f32;
    let mut norm_sqr = 0.0f32;
    for (a, b) in data.iter().zip(reconstructed.iter()) {
        let diff = a - b;
        diff_sqr += diff * diff;
        norm_sqr += a * a;
    }
    let rel_error = if norm_sqr > f32::EPSILON {
        (diff_sqr.sqrt()) / norm_sqr.sqrt()
    } else {
        0.0
    };
    assert!(
        rel_error < 0.3,
        "relative reconstruction error too high: {rel_error}"
    );
}

#[test]
fn ivf_search_recovers_identical_vectors() {
    let dim = 32;
    let total = 256;
    let mut rng = StdRng::seed_from_u64(4321);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(&data, 32, 7, Metric::L2, 7777).expect("train index");
    let params = SearchParams::new(1, 32);

    for (idx, vector) in data.iter().take(16).enumerate() {
        let results = index.search(vector, params).expect("search");
        assert!(!results.is_empty(), "no results returned for query {idx}");
        assert_eq!(results[0].id, idx, "failed to retrieve vector {idx}");
    }
}

#[test]
fn search_recovers_shifted_centroid_neighbour() {
    let base = vec![
        vec![10.5f32, 9.5f32],
        vec![9.8f32, 10.2f32],
        vec![11.2f32, 10.8f32],
        vec![9.2f32, 9.1f32],
    ];
    let centroids = vec![vec![10.0f32, 10.0f32]];
    let assignments = vec![0usize; base.len()];

    let index =
        IvfRabitqIndex::train_with_clusters(&base, &centroids, &assignments, 8, Metric::L2, 9876)
            .expect("train index with provided clusters");

    let params = SearchParams::new(1, 1);
    let query = vec![9.0f32, 9.0f32];

    let mut expected_id = 0usize;
    let mut best_dist = f32::INFINITY;
    for (idx, candidate) in base.iter().enumerate() {
        let mut dist = 0.0f32;
        for (a, b) in candidate.iter().zip(query.iter()) {
            let diff = a - b;
            dist += diff * diff;
        }
        if dist < best_dist {
            best_dist = dist;
            expected_id = idx;
        }
    }

    assert_eq!(
        expected_id, 3,
        "brute-force nearest neighbour should be id 3"
    );

    let results = index.search(&query, params).expect("search index");
    assert!(
        !results.is_empty(),
        "no candidates returned for shifted centroid query"
    );
    assert_eq!(
        results[0].id, expected_id,
        "search should recover the brute-force nearest neighbour"
    );
}

#[test]
fn search_estimator_uses_query_coordinates() {
    let base = vec![
        vec![10.5f32, 9.5f32],
        vec![9.8f32, 10.2f32],
        vec![11.2f32, 10.8f32],
        vec![9.2f32, 9.1f32],
    ];
    let centroids = vec![vec![10.0f32, 10.0f32]];
    let assignments = vec![0usize; base.len()];
    let total_bits = 8;

    let index = IvfRabitqIndex::train_with_clusters(
        &base,
        &centroids,
        &assignments,
        total_bits,
        Metric::L2,
        9876,
    )
    .expect("train index with provided clusters");

    let params = SearchParams::new(2, 1);
    let query = vec![9.0f32, 9.0f32];

    let results = index.search(&query, params).expect("search index");
    assert!(
        !results.is_empty(),
        "no candidates returned for estimator check"
    );
    let best_id = results[0].id;

    let rotator = RandomRotator::new(query.len(), 9876);
    let rotated_query = rotator.rotate_inverse(&query);
    let rotated_centroid = rotator.rotate_inverse(&centroids[0]);
    let rotated_base = rotator.rotate_inverse(&base[best_id]);
    let quantized =
        quantize_with_centroid(&rotated_base, &rotated_centroid, total_bits, Metric::L2);

    let centroid_dist = l2_distance_sqr(&rotated_query, &rotated_centroid);
    let g_add = centroid_dist;
    let c1 = -0.5f32;
    let ex_bits = total_bits - 1;
    let binary_scale = (1 << ex_bits) as f32;
    let cb = -((1 << ex_bits) as f32 - 0.5);
    let sum_query: f32 = rotated_query.iter().copied().sum();
    let binary_dot_query: f32 = quantized
        .binary_code
        .iter()
        .zip(rotated_query.iter())
        .map(|(&bit, &q)| (bit as f32) * q)
        .sum();
    let binary_term_query = binary_dot_query + c1 * sum_query;
    let mut distance_query = quantized.f_add + g_add + quantized.f_rescale * binary_term_query;
    if ex_bits > 0 {
        let ex_dot_query: f32 = quantized
            .ex_code
            .iter()
            .zip(rotated_query.iter())
            .map(|(&code, &q)| (code as f32) * q)
            .sum();
        distance_query = quantized.f_add_ex
            + g_add
            + quantized.f_rescale_ex
                * (binary_scale * binary_dot_query + ex_dot_query + cb * sum_query);
    }

    let residuals: Vec<f32> = rotated_query
        .iter()
        .zip(rotated_centroid.iter())
        .map(|(&q, &c)| q - c)
        .collect();
    let sum_residual: f32 = residuals.iter().copied().sum();
    let binary_dot_residual: f32 = quantized
        .binary_code
        .iter()
        .zip(residuals.iter())
        .map(|(&bit, &r)| (bit as f32) * r)
        .sum();
    let binary_term_residual = binary_dot_residual + c1 * sum_residual;
    let mut distance_residual =
        quantized.f_add + g_add + quantized.f_rescale * binary_term_residual;
    if ex_bits > 0 {
        let ex_dot_residual: f32 = quantized
            .ex_code
            .iter()
            .zip(residuals.iter())
            .map(|(&code, &r)| (code as f32) * r)
            .sum();
        distance_residual = quantized.f_add_ex
            + g_add
            + quantized.f_rescale_ex
                * (binary_scale * binary_dot_residual + ex_dot_residual + cb * sum_residual);
    }

    let returned_distance = results
        .iter()
        .find(|res| res.id == best_id)
        .map(|res| res.score)
        .expect("distance for best id");

    assert!(
        (returned_distance - distance_query).abs() < 1e-3,
        "search should evaluate query-based estimator",
    );
    assert!(
        (returned_distance - distance_residual).abs() > 1e-4,
        "search distance should diverge from residual-based estimator",
    );
}

#[test]
fn noisy_queries_recover_original_vectors() {
    let dim = 32;
    let total = 192;
    let mut rng = StdRng::seed_from_u64(314159);
    let normal = rand_distr::Normal::new(0.0, 0.05).expect("create normal distribution");
    let mut base = Vec::with_capacity(total);
    for _ in 0..total {
        base.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(&base, 24, 7, Metric::L2, 1618)
        .expect("train index from random data");
    let params = SearchParams::new(1, 24);

    for (idx, vector) in base.iter().enumerate().take(24) {
        let mut query = vector.clone();
        for value in query.iter_mut() {
            *value += rng.sample::<f32, _>(normal);
        }

        let brute_force = base
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = a
                    .iter()
                    .zip(query.iter())
                    .map(|(x, y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum::<f32>();
                let dist_b = b
                    .iter()
                    .zip(query.iter())
                    .map(|(x, y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum::<f32>();
                dist_a.total_cmp(&dist_b)
            })
            .map(|(best_idx, _)| best_idx)
            .expect("brute-force neighbour");

        let results = index.search(&query, params).expect("fastscan search");
        assert!(
            !results.is_empty(),
            "search returned no results for noisy query {idx}",
        );
        assert_eq!(
            results[0].id, brute_force,
            "fastscan should agree with brute-force for noisy query {idx}",
        );
    }
}

#[test]
fn fastscan_matches_naive_l2() {
    let dim = 48;
    let total = 320;
    let mut rng = StdRng::seed_from_u64(2468);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(&data, 40, 7, Metric::L2, 1357).expect("train index");
    let params = SearchParams::new(5, 12);

    let mut total_evals = 0usize;
    for query in data.iter().take(12) {
        let (fastscan, diag) = index
            .search_with_diagnostics(query, params)
            .expect("fastscan search");
        total_evals += diag.extended_evaluations;
        let naive = index.search_naive(query, params).expect("naive search");
        assert_eq!(fastscan, naive, "fastscan diverged from naive on L2");
    }
    assert!(
        total_evals > 0,
        "extended path never evaluated for L2 dataset"
    );
}

#[test]
fn fastscan_matches_naive_ip() {
    let dim = 24;
    let total = 200;
    let mut rng = StdRng::seed_from_u64(9753);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index =
        IvfRabitqIndex::train(&data, 25, 6, Metric::InnerProduct, 8642).expect("train index");
    let params = SearchParams::new(7, 10);

    let mut total_evals = 0usize;
    for query in data.iter().take(12) {
        let (fastscan, diag) = index
            .search_with_diagnostics(query, params)
            .expect("fastscan search");
        total_evals += diag.extended_evaluations;
        let naive = index.search_naive(query, params).expect("naive search");
        assert_eq!(fastscan, naive, "fastscan diverged from naive on IP");
    }
    assert!(
        total_evals > 0,
        "extended path never evaluated for IP dataset"
    );
}

#[test]
fn one_bit_search_has_no_extended_pruning() {
    let dim = 20;
    let total = 120;
    let mut rng = StdRng::seed_from_u64(4242);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(&data, 16, 1, Metric::L2, 2024).expect("train index");
    let params = SearchParams::new(4, 10);

    for query in data.iter().take(6) {
        let (fastscan, diag) = index
            .search_with_diagnostics(query, params)
            .expect("fastscan search");
        let naive = index.search_naive(query, params).expect("naive search");
        assert_eq!(fastscan, naive, "one-bit search diverged from naive");
        assert_eq!(
            diag.extended_evaluations, 0,
            "extended path should not be evaluated when ex_bits=0"
        );
        assert!(
            diag.estimated > 0,
            "no candidates were fully estimated in one-bit search"
        );
    }
}

#[test]
fn fvecs_reader_parses_vectors() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(3i32).to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    bytes.extend_from_slice(&3.0f32.to_le_bytes());
    bytes.extend_from_slice(&(3i32).to_le_bytes());
    bytes.extend_from_slice(&4.0f32.to_le_bytes());
    bytes.extend_from_slice(&5.0f32.to_le_bytes());
    bytes.extend_from_slice(&6.0f32.to_le_bytes());

    let vectors = read_fvecs_from_reader(Cursor::new(bytes.clone()), None).expect("read fvecs");
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(vectors[1], vec![4.0, 5.0, 6.0]);

    let limited = read_fvecs_from_reader(Cursor::new(bytes), Some(1)).expect("read fvecs limit");
    assert_eq!(limited.len(), 1);
}

#[test]
fn ivecs_reader_parses_vectors() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(2i32).to_le_bytes());
    bytes.extend_from_slice(&7i32.to_le_bytes());
    bytes.extend_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&(2i32).to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&9i32.to_le_bytes());

    let rows = read_ivecs_from_reader(Cursor::new(bytes), None).expect("read ivecs");
    assert_eq!(rows, vec![vec![7, 8], vec![1, 9]]);

    let mut id_bytes = Vec::new();
    id_bytes.extend_from_slice(&(1i32).to_le_bytes());
    id_bytes.extend_from_slice(&5i32.to_le_bytes());
    id_bytes.extend_from_slice(&(1i32).to_le_bytes());
    id_bytes.extend_from_slice(&6i32.to_le_bytes());
    id_bytes.extend_from_slice(&(1i32).to_le_bytes());
    id_bytes.extend_from_slice(&11i32.to_le_bytes());

    let ids = read_ids_from_reader(Cursor::new(id_bytes), None).expect("read ids");
    assert_eq!(ids, vec![5usize, 6usize, 11usize]);

    let mut gt_bytes = Vec::new();
    gt_bytes.extend_from_slice(&(3i32).to_le_bytes());
    gt_bytes.extend_from_slice(&0i32.to_le_bytes());
    gt_bytes.extend_from_slice(&1i32.to_le_bytes());
    gt_bytes.extend_from_slice(&2i32.to_le_bytes());
    gt_bytes.extend_from_slice(&(2i32).to_le_bytes());
    gt_bytes.extend_from_slice(&3i32.to_le_bytes());
    gt_bytes.extend_from_slice(&4i32.to_le_bytes());

    let groundtruth =
        read_groundtruth_from_reader(Cursor::new(gt_bytes), None).expect("read groundtruth");
    assert_eq!(groundtruth, vec![vec![0, 1, 2], vec![3, 4]]);
}

#[test]
fn third_party_kmeans_converges_on_separated_clusters() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.2, -0.1],
        vec![-0.1, 0.3],
        vec![4.0, 4.2],
        vec![3.9, 4.1],
        vec![4.3, 3.8],
    ];

    let mut rng = StdRng::seed_from_u64(0xACED);
    let result = run_kmeans(&data, 2, 25, &mut rng);

    assert_eq!(result.centroids.len(), 2);
    assert_eq!(result.assignments.len(), data.len());

    let mut centroids = result.centroids.clone();
    centroids.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let left = &centroids[0];
    let right = &centroids[1];

    assert!(
        left[0].abs() < 0.5 && left[1].abs() < 0.5,
        "left centroid: {:?}",
        left
    );
    assert!(
        (right[0] - 4.0).abs() < 0.5 && (right[1] - 4.0).abs() < 0.5,
        "right centroid: {:?}",
        right
    );

    let first_cluster = result.assignments[0];
    assert!(result.assignments[1..3]
        .iter()
        .all(|&assignment| assignment == first_cluster));
    let second_cluster = result.assignments[3];
    assert!(result.assignments[4..]
        .iter()
        .all(|&assignment| assignment == second_cluster));
    assert_ne!(first_cluster, second_cluster);
}

#[test]
fn preclustered_training_matches_naive_l2() {
    let dim = 28;
    let total = 240;
    let mut rng = StdRng::seed_from_u64(0xA51CE);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let mut kmeans_rng = StdRng::seed_from_u64(0x5EED);
    let kmeans = run_kmeans(&data, 32, 25, &mut kmeans_rng);

    let index = IvfRabitqIndex::train_with_clusters(
        &data,
        &kmeans.centroids,
        &kmeans.assignments,
        7,
        Metric::L2,
        0xBEEF,
    )
    .expect("train with clusters");

    let params = SearchParams::new(6, 16);
    for query in data.iter().take(8) {
        let (fastscan, _) = index
            .search_with_diagnostics(query, params)
            .expect("fastscan search");
        let naive = index.search_naive(query, params).expect("naive search");
        assert_eq!(fastscan, naive, "preclustered search diverged on L2");
    }
}

#[test]
fn preclustered_training_matches_naive_ip() {
    let dim = 18;
    let total = 180;
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let mut kmeans_rng = StdRng::seed_from_u64(0xB16B00B5);
    let kmeans = run_kmeans(&data, 24, 30, &mut kmeans_rng);

    let index = IvfRabitqIndex::train_with_clusters(
        &data,
        &kmeans.centroids,
        &kmeans.assignments,
        6,
        Metric::InnerProduct,
        0x1234_5678,
    )
    .expect("train with clusters");

    let params = SearchParams::new(5, 12);
    for query in data.iter().take(10) {
        let (fastscan, _) = index
            .search_with_diagnostics(query, params)
            .expect("fastscan search");
        let naive = index.search_naive(query, params).expect("naive search");
        assert_eq!(fastscan, naive, "preclustered search diverged on IP");
    }
}
