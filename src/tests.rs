use std::io::Cursor;

use rand::prelude::*;

use crate::io::{
    read_fvecs_from_reader, read_groundtruth_from_reader, read_ids_from_reader,
    read_ivecs_from_reader,
};
use crate::ivf::{IvfRabitqIndex, SearchParams};
use crate::kmeans::run_kmeans;
use crate::quantizer::{quantize_with_centroid, reconstruct_into, RabitqConfig};
use crate::{Metric, RabitqError, RoaringBitmap, RotatorType};

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

#[test]
fn quantizer_reconstruction_is_reasonable() {
    let dim = 64;
    let mut rng = StdRng::seed_from_u64(1234);
    let data = random_vector(dim, &mut rng);
    let centroid = vec![0.0f32; dim];
    let config = RabitqConfig::new(7);
    let quantized = quantize_with_centroid(&data, &centroid, &config, Metric::L2);
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

    let index = IvfRabitqIndex::train(
        &data,
        32,
        7,
        Metric::L2,
        RotatorType::FhtKacRotator,
        7777,
        false,
    )
    .expect("train index");
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

    let index = IvfRabitqIndex::train(
        &data,
        40,
        7,
        Metric::L2,
        RotatorType::FhtKacRotator,
        1357,
        false,
    )
    .expect("train index");
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

    let index = IvfRabitqIndex::train(
        &data,
        25,
        6,
        Metric::InnerProduct,
        RotatorType::FhtKacRotator,
        8642,
        false,
    )
    .expect("train index");
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

    let index = IvfRabitqIndex::train(
        &data,
        16,
        1,
        Metric::L2,
        RotatorType::FhtKacRotator,
        2024,
        false,
    )
    .expect("train index");
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
fn index_persistence_roundtrip() {
    let dim = 24;
    let total = 200;
    let mut rng = StdRng::seed_from_u64(7412);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(
        &data,
        32,
        7,
        Metric::InnerProduct,
        RotatorType::FhtKacRotator,
        9999,
        false,
    )
    .expect("train index");

    let mut buffer = Vec::new();
    index.save_to_writer(&mut buffer).expect("serialize index");

    let restored = IvfRabitqIndex::load_from_reader(buffer.as_slice()).expect("deserialize index");

    assert_eq!(
        restored.len(),
        index.len(),
        "vector count changed after reload"
    );

    let params = SearchParams::new(5, 12);
    for query in data.iter().take(5) {
        let original = index.search(query, params).expect("search original");
        let roundtrip = restored.search(query, params).expect("search restored");
        assert_eq!(original, roundtrip, "round-tripped results diverged");
    }
}

#[test]
fn index_persistence_detects_corruption() {
    let dim = 16;
    let total = 128;
    let mut rng = StdRng::seed_from_u64(0xFACE);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(
        &data,
        24,
        6,
        Metric::L2,
        RotatorType::FhtKacRotator,
        4242,
        false,
    )
    .expect("train index");
    let mut buffer = Vec::new();
    index.save_to_writer(&mut buffer).expect("serialize index");

    let checksum_offset = buffer.len().saturating_sub(4);
    assert!(checksum_offset > 0, "buffer too small for checksum test");
    buffer[checksum_offset - 1] ^= 0xAA;

    match IvfRabitqIndex::load_from_reader(buffer.as_slice()) {
        Err(RabitqError::InvalidPersistence(msg)) => {
            assert_eq!(msg, "checksum mismatch", "unexpected error message")
        }
        other => panic!("expected checksum mismatch, got {other:?}"),
    }
}

#[test]
fn index_persistence_validates_vector_count() {
    let dim = 12;
    let total = 96;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(
        &data,
        16,
        5,
        Metric::InnerProduct,
        RotatorType::FhtKacRotator,
        9001,
        false,
    )
    .expect("train index");
    let mut buffer = Vec::new();
    index.save_to_writer(&mut buffer).expect("serialize index");

    let vector_count_offset = 4 + 4 + 4 + 4 + 1 + 1 + 1 + 1; // header + dim + padded_dim + metric + rotator_type + ex_bits + total_bits
    let checksum_offset = buffer.len().saturating_sub(4);
    assert!(checksum_offset > vector_count_offset + 8);

    let mut raw = [0u8; 8];
    raw.copy_from_slice(&buffer[vector_count_offset..vector_count_offset + 8]);
    let original = u64::from_le_bytes(raw);
    let updated = original + 1;
    buffer[vector_count_offset..vector_count_offset + 8].copy_from_slice(&updated.to_le_bytes());

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&buffer[8..checksum_offset]);
    let new_checksum = hasher.finalize();
    buffer[checksum_offset..].copy_from_slice(&new_checksum.to_le_bytes());

    match IvfRabitqIndex::load_from_reader(buffer.as_slice()) {
        Err(RabitqError::InvalidPersistence(msg)) => {
            assert_eq!(
                msg, "vector count metadata mismatch",
                "unexpected error message"
            );
        }
        other => panic!("expected vector count mismatch, got {other:?}"),
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
        RotatorType::FhtKacRotator,
        0xBEEF,
        false,
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
        RotatorType::FhtKacRotator,
        0x1234_5678,
        false,
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

#[test]
fn filtered_search_respects_bitmap() {
    let dim = 32;
    let total = 100;
    let mut rng = StdRng::seed_from_u64(9999);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(
        &data,
        16,
        7,
        Metric::L2,
        RotatorType::FhtKacRotator,
        1111,
        false,
    )
    .expect("train index");

    // Create a filter that only includes IDs 10-19
    let mut filter = RoaringBitmap::new();
    for id in 10..20 {
        filter.insert(id);
    }

    let params = SearchParams::new(5, 16);
    let query = &data[0];

    // Perform filtered search
    let filtered_results = index
        .search_filtered(query, params, &filter)
        .expect("filtered search");

    // All returned IDs should be in the filter
    for result in &filtered_results {
        assert!(
            filter.contains(result.id as u32),
            "result id {} not in filter",
            result.id
        );
    }
}

#[test]
fn filtered_search_returns_correct_results() {
    let dim = 32;
    let total = 200;
    let mut rng = StdRng::seed_from_u64(8888);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(
        &data,
        20,
        7,
        Metric::L2,
        RotatorType::FhtKacRotator,
        2222,
        false,
    )
    .expect("train index");

    // Create a filter that includes all even IDs
    let mut filter = RoaringBitmap::new();
    for id in (0..total).step_by(2) {
        filter.insert(id as u32);
    }

    let params = SearchParams::new(10, 20);
    let query = &data[50];

    // Perform filtered search
    let filtered_results = index
        .search_filtered(query, params, &filter)
        .expect("filtered search");

    // Get unfiltered search results
    let unfiltered_results = index.search(query, params).expect("unfiltered search");

    // Verify all filtered results are in the filter
    for result in &filtered_results {
        assert!(
            filter.contains(result.id as u32),
            "filtered result id {} not in filter",
            result.id
        );
    }

    // Verify that filtered results don't include any odd IDs
    for result in &filtered_results {
        assert_eq!(
            result.id % 2,
            0,
            "filtered result id {} is odd but should be even",
            result.id
        );
    }

    // The filtered results should be a subset of unfiltered results (after applying the filter)
    let unfiltered_filtered_ids: Vec<usize> = unfiltered_results
        .iter()
        .filter(|r| filter.contains(r.id as u32))
        .map(|r| r.id)
        .collect();
    let filtered_ids: Vec<usize> = filtered_results.iter().map(|r| r.id).collect();

    // The top results should match
    let min_len = filtered_ids.len().min(unfiltered_filtered_ids.len());
    assert_eq!(
        &filtered_ids[..min_len],
        &unfiltered_filtered_ids[..min_len],
        "filtered search results differ from manually filtered unfiltered results"
    );
}

#[test]
fn filtered_search_with_empty_filter() {
    let dim = 32;
    let total = 50;
    let mut rng = StdRng::seed_from_u64(7777);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(random_vector(dim, &mut rng));
    }

    let index = IvfRabitqIndex::train(
        &data,
        10,
        7,
        Metric::L2,
        RotatorType::FhtKacRotator,
        3333,
        false,
    )
    .expect("train index");

    // Create an empty filter
    let filter = RoaringBitmap::new();

    let params = SearchParams::new(5, 10);
    let query = &data[0];

    // Perform filtered search with empty filter
    let results = index
        .search_filtered(query, params, &filter)
        .expect("filtered search");

    // Should return no results
    assert!(
        results.is_empty(),
        "expected empty results with empty filter, got {}",
        results.len()
    );
}
