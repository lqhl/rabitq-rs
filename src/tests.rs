use rand::prelude::*;

use crate::ivf::{IvfRabitqIndex, SearchParams};
use crate::quantizer::{quantize_with_centroid, reconstruct_into};
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
