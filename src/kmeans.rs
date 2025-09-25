use std::cmp::Ordering;

use matrixmultiply::sgemm;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::RngCore;
use rayon::prelude::*;

const MAX_POINTS_PER_CENTROID: usize = 256;
const ASSIGNMENT_CHUNK_SIZE: usize = 2048;
const RESEED_CANDIDATES: usize = 8;
const ENABLE_SPHERICAL_KMEANS: bool = false;

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

/// Run k-means clustering using a Faiss-inspired pipeline with GEMM-powered
/// assignments, sampling, and progressive-dimension refinement.
pub fn run_kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, rng: &mut StdRng) -> KMeansResult {
    let dim = validate_inputs(data, k, max_iter);
    let total_points = data.len();

    let flattened = flatten_dataset(data, total_points * dim);

    let sampling_seed = rng.next_u64();
    let init_seed = rng.next_u64();
    let reseed_seed = rng.next_u64();

    let mut sampling_rng = StdRng::seed_from_u64(sampling_seed);
    let mut init_rng = StdRng::seed_from_u64(init_seed);
    let mut reseed_rng = StdRng::seed_from_u64(reseed_seed);

    let training_indices = select_training_indices(total_points, k, &mut sampling_rng);
    let training_data = gather_rows(&flattened, &training_indices, dim);
    let training_rows = training_indices.len();

    let schedule = progressive_dim_schedule(dim);
    let stage_iterations = distribute_iterations(max_iter, schedule.len());

    let mut centroids = initialize_centroids(
        &training_data,
        training_rows,
        dim,
        k,
        &schedule,
        &mut init_rng,
    );
    let mut training_assignments = vec![0usize; training_rows];

    for (&stage_dim, &iterations) in schedule.iter().zip(stage_iterations.iter()) {
        if iterations == 0 {
            continue;
        }
        let norms = compute_partial_norms(&training_data, training_rows, dim, stage_dim);
        run_lloyd_stage(
            &mut centroids,
            stage_dim,
            iterations,
            k,
            dim,
            &training_data,
            &norms,
            &mut training_assignments,
            &mut reseed_rng,
        );
    }

    let full_norms = compute_partial_norms(&flattened, total_points, dim, dim);
    let mut centroid_col = Vec::with_capacity(dim * k);
    let mut centroid_norms = Vec::with_capacity(k);
    rebuild_centroid_views(
        &centroids,
        k,
        dim,
        dim,
        &mut centroid_col,
        &mut centroid_norms,
    );
    let assignments = assign_full_dataset(
        &flattened,
        &full_norms,
        k,
        dim,
        &centroid_col,
        &centroid_norms,
    );

    let centroids_vec = centroids.chunks(dim).map(|chunk| chunk.to_vec()).collect();

    KMeansResult {
        centroids: centroids_vec,
        assignments,
    }
}

fn validate_inputs(data: &[Vec<f32>], k: usize, max_iter: usize) -> usize {
    assert!(!data.is_empty(), "k-means requires non-empty data");
    assert!(k > 0, "k must be positive");
    assert!(max_iter > 0, "max_iter must be positive");
    assert!(k <= data.len(), "k cannot exceed number of samples");

    let dim = data[0].len();
    assert!(
        data.iter().all(|v| v.len() == dim),
        "all vectors must share the same dimension",
    );
    dim
}

fn flatten_dataset(data: &[Vec<f32>], capacity: usize) -> Vec<f32> {
    let mut flattened = Vec::with_capacity(capacity);
    for vector in data {
        flattened.extend_from_slice(vector);
    }
    flattened
}

fn select_training_indices(total_points: usize, k: usize, rng: &mut StdRng) -> Vec<usize> {
    let target = total_points.min(k * MAX_POINTS_PER_CENTROID).max(k);
    if target == total_points {
        return (0..total_points).collect();
    }

    let mut indices: Vec<usize> = (0..total_points).collect();
    indices.shuffle(rng);
    indices.truncate(target);
    indices.sort_unstable();
    indices
}

fn gather_rows(data: &[f32], indices: &[usize], dim: usize) -> Vec<f32> {
    let mut gathered = Vec::with_capacity(indices.len() * dim);
    for &idx in indices {
        let start = idx * dim;
        let end = start + dim;
        gathered.extend_from_slice(&data[start..end]);
    }
    gathered
}

fn progressive_dim_schedule(dim: usize) -> Vec<usize> {
    if dim <= 32 {
        return vec![dim];
    }

    let mut schedule = Vec::new();
    let mut current = 32usize;
    while current < dim {
        schedule.push(current.min(dim));
        current *= 2;
    }
    if *schedule.last().unwrap() != dim {
        schedule.push(dim);
    }
    schedule.sort_unstable();
    schedule.dedup();
    schedule
}

fn distribute_iterations(total: usize, stages: usize) -> Vec<usize> {
    if stages == 0 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(stages);
    let mut remaining = total;
    for stage in 0..stages {
        let stages_left = stages - stage;
        let share = if stage == stages - 1 {
            remaining
        } else if remaining < stages_left {
            0
        } else {
            let provisional = remaining / stages_left;
            provisional.max(1)
        };
        result.push(share);
        remaining = remaining.saturating_sub(share);
    }
    result
}

fn initialize_centroids(
    data: &[f32],
    rows: usize,
    dim: usize,
    k: usize,
    schedule: &[usize],
    rng: &mut StdRng,
) -> Vec<f32> {
    let init_dim = schedule.first().copied().unwrap_or(dim);
    kmeans_plus_plus(data, rows, dim, init_dim, k, rng)
}

fn kmeans_plus_plus(
    data: &[f32],
    rows: usize,
    dim: usize,
    init_dim: usize,
    k: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * dim);
    let mut distances = vec![f32::INFINITY; rows];

    let first_idx = rng.gen_range(0..rows);
    centroids.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

    for centroid_idx in 1..k {
        update_distances(
            data,
            dim,
            init_dim,
            &centroids,
            centroid_idx,
            &mut distances,
        );
        let total: f64 = distances.iter().map(|&d| d.max(0.0) as f64).sum();
        if total == 0.0 {
            // All points coincide; duplicate the last centroid.
            let start = (centroid_idx - 1) * dim;
            let end = centroid_idx * dim;
            let duplicate = centroids[start..end].to_vec();
            centroids.extend_from_slice(&duplicate);
            continue;
        }
        let mut threshold = rng.gen::<f64>() * total;
        let mut next_idx = 0;
        for (i, &distance) in distances.iter().enumerate() {
            threshold -= distance.max(0.0) as f64;
            if threshold <= 0.0 {
                next_idx = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[next_idx * dim..(next_idx + 1) * dim]);
    }

    centroids
}

fn update_distances(
    data: &[f32],
    dim: usize,
    effective_dim: usize,
    centroids: &[f32],
    centroid_count: usize,
    distances: &mut [f32],
) {
    for (row_idx, distance) in distances.iter_mut().enumerate() {
        let row = &data[row_idx * dim..row_idx * dim + effective_dim];
        let mut best = *distance;
        for centroid_idx in 0..centroid_count {
            let centroid = &centroids[centroid_idx * dim..centroid_idx * dim + effective_dim];
            let mut dist = 0.0f32;
            for d in 0..effective_dim {
                let delta = row[d] - centroid[d];
                dist += delta * delta;
            }
            if dist < best {
                best = dist;
            }
        }
        *distance = best;
    }
}

fn compute_partial_norms(data: &[f32], rows: usize, dim: usize, effective_dim: usize) -> Vec<f32> {
    let mut norms = vec![0.0f32; rows];
    for (row, norm) in norms.iter_mut().enumerate() {
        let start = row * dim;
        let slice = &data[start..start + effective_dim];
        *norm = slice.iter().map(|value| value * value).sum();
    }
    norms
}

#[allow(clippy::too_many_arguments)]
fn run_lloyd_stage(
    centroids: &mut [f32],
    stage_dim: usize,
    iterations: usize,
    k: usize,
    dim: usize,
    data: &[f32],
    norms: &[f32],
    assignments: &mut [usize],
    rng: &mut StdRng,
) {
    let mut centroid_col = Vec::with_capacity(stage_dim * k);
    let mut centroid_norms = Vec::with_capacity(k);

    for _ in 0..iterations {
        rebuild_centroid_views(
            centroids,
            k,
            dim,
            stage_dim,
            &mut centroid_col,
            &mut centroid_norms,
        );
        let summary = assign_points_for_update(
            data,
            norms,
            assignments,
            k,
            dim,
            stage_dim,
            &centroid_col,
            &centroid_norms,
        );
        update_centroids(centroids, k, dim, stage_dim, data, &summary, rng);
        if ENABLE_SPHERICAL_KMEANS {
            normalize_centroids(centroids, k, dim, stage_dim);
        }
    }
}

fn rebuild_centroid_views(
    centroids: &[f32],
    k: usize,
    dim: usize,
    effective_dim: usize,
    centroid_col: &mut Vec<f32>,
    centroid_norms: &mut Vec<f32>,
) {
    centroid_col.clear();
    centroid_col.resize(effective_dim * k, 0.0);
    centroid_norms.clear();
    centroid_norms.resize(k, 0.0);

    for cluster in 0..k {
        let centroid = &centroids[cluster * dim..cluster * dim + effective_dim];
        let mut norm = 0.0f32;
        for d in 0..effective_dim {
            let value = centroid[d];
            centroid_col[d * k + cluster] = value;
            norm += value * value;
        }
        centroid_norms[cluster] = norm;
    }
}

fn normalize_centroids(centroids: &mut [f32], k: usize, dim: usize, effective_dim: usize) {
    for cluster in 0..k {
        let offset = cluster * dim;
        let centroid = &mut centroids[offset..offset + effective_dim];
        let mut norm = 0.0f32;
        for &value in centroid.iter() {
            norm += value * value;
        }
        if norm > 0.0 {
            let inv = norm.sqrt().recip();
            for value in centroid.iter_mut() {
                *value *= inv;
            }
        }
    }
}

#[derive(Debug)]
struct AssignmentSummary {
    counts: Vec<usize>,
    sums: Vec<f32>,
    candidates: Vec<(f32, usize)>,
}

#[derive(Debug)]
struct ChunkResult {
    start: usize,
    len: usize,
    assignments: Vec<usize>,
    counts: Vec<usize>,
    sums: Vec<f32>,
    candidates: Vec<(f32, usize)>,
}

#[allow(clippy::too_many_arguments)]
fn assign_points_for_update(
    data: &[f32],
    norms: &[f32],
    assignments: &mut [usize],
    k: usize,
    dim: usize,
    effective_dim: usize,
    centroid_col: &[f32],
    centroid_norms: &[f32],
) -> AssignmentSummary {
    let rows = norms.len();
    let num_chunks = (rows + ASSIGNMENT_CHUNK_SIZE - 1) / ASSIGNMENT_CHUNK_SIZE;

    let results: Vec<ChunkResult> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * ASSIGNMENT_CHUNK_SIZE;
            let end = ((chunk_idx + 1) * ASSIGNMENT_CHUNK_SIZE).min(rows);
            let len = end - start;
            let data_chunk = &data[start * dim..end * dim];
            let norms_chunk = &norms[start..end];
            compute_chunk_assignments(
                start,
                data_chunk,
                norms_chunk,
                len,
                k,
                dim,
                effective_dim,
                centroid_col,
                centroid_norms,
            )
        })
        .collect();

    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * effective_dim];
    let mut candidates = Vec::new();

    for result in results {
        let ChunkResult {
            start,
            len,
            assignments: chunk_assignments,
            counts: chunk_counts,
            sums: chunk_sums,
            candidates: mut chunk_candidates,
        } = result;
        let end = start + len;
        assignments[start..end].copy_from_slice(&chunk_assignments);
        for cluster in 0..k {
            counts[cluster] += chunk_counts[cluster];
            let dst_offset = cluster * effective_dim;
            let src_offset = cluster * effective_dim;
            for d in 0..effective_dim {
                sums[dst_offset + d] += chunk_sums[src_offset + d];
            }
        }
        candidates.append(&mut chunk_candidates);
    }

    AssignmentSummary {
        counts,
        sums,
        candidates,
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_chunk_assignments(
    start: usize,
    data_chunk: &[f32],
    norms_chunk: &[f32],
    len: usize,
    k: usize,
    dim: usize,
    effective_dim: usize,
    centroid_col: &[f32],
    centroid_norms: &[f32],
) -> ChunkResult {
    let mut dot_products = vec![0.0f32; len * k];
    unsafe {
        sgemm(
            len,
            effective_dim,
            k,
            1.0,
            data_chunk.as_ptr(),
            dim as isize,
            1,
            centroid_col.as_ptr(),
            k as isize,
            1,
            0.0,
            dot_products.as_mut_ptr(),
            k as isize,
            1,
        );
    }

    let mut assignments = Vec::with_capacity(len);
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * effective_dim];
    let mut candidates: Vec<(f32, usize)> = Vec::new();

    for row in 0..len {
        let norm = norms_chunk[row];
        let mut best_cluster = 0usize;
        let mut best_distance = f32::INFINITY;
        for cluster in 0..k {
            let dot = dot_products[row * k + cluster];
            let mut distance = norm + centroid_norms[cluster] - 2.0 * dot;
            if distance < 0.0 {
                distance = 0.0;
            }
            if distance < best_distance {
                best_distance = distance;
                best_cluster = cluster;
            }
        }
        assignments.push(best_cluster);
        counts[best_cluster] += 1;
        let vector = &data_chunk[row * dim..row * dim + effective_dim];
        let sum_offset = best_cluster * effective_dim;
        for d in 0..effective_dim {
            sums[sum_offset + d] += vector[d];
        }
        insert_candidate(&mut candidates, (best_distance, row));
    }

    let global_candidates = candidates
        .into_iter()
        .map(|(dist, local_idx)| (dist, start + local_idx))
        .collect();

    ChunkResult {
        start,
        len,
        assignments,
        counts,
        sums,
        candidates: global_candidates,
    }
}

fn insert_candidate(candidates: &mut Vec<(f32, usize)>, candidate: (f32, usize)) {
    if candidates.len() < RESEED_CANDIDATES {
        candidates.push(candidate);
        candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        return;
    }
    if let Some((last_dist, _)) = candidates.last() {
        if candidate.0 > *last_dist {
            candidates.pop();
            candidates.push(candidate);
            candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }
    }
}

fn update_centroids(
    centroids: &mut [f32],
    k: usize,
    dim: usize,
    effective_dim: usize,
    data: &[f32],
    summary: &AssignmentSummary,
    rng: &mut StdRng,
) {
    let total_rows = data.len() / dim;
    let mut candidate_pool = summary.candidates.clone();
    candidate_pool.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    let mut used = vec![false; total_rows];
    let mut candidate_indices = Vec::new();
    for (_, idx) in candidate_pool.into_iter() {
        if !used[idx] {
            used[idx] = true;
            candidate_indices.push(idx);
        }
    }
    let mut candidate_iter = candidate_indices.into_iter();

    for cluster in 0..k {
        let offset = cluster * dim;
        let count = summary.counts[cluster];
        if count > 0 {
            let inv = 1.0 / count as f32;
            let sum_offset = cluster * effective_dim;
            for d in 0..effective_dim {
                centroids[offset + d] = summary.sums[sum_offset + d] * inv;
            }
        } else {
            let replacement_index = candidate_iter
                .next()
                .unwrap_or_else(|| rng.gen_range(0..total_rows));
            let source = &data[replacement_index * dim..(replacement_index + 1) * dim];
            centroids[offset..offset + dim].copy_from_slice(source);
        }
    }
}

fn assign_full_dataset(
    data: &[f32],
    norms: &[f32],
    k: usize,
    dim: usize,
    centroid_col: &[f32],
    centroid_norms: &[f32],
) -> Vec<usize> {
    let rows = norms.len();
    let num_chunks = (rows + ASSIGNMENT_CHUNK_SIZE - 1) / ASSIGNMENT_CHUNK_SIZE;
    let results: Vec<(usize, Vec<usize>)> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * ASSIGNMENT_CHUNK_SIZE;
            let end = ((chunk_idx + 1) * ASSIGNMENT_CHUNK_SIZE).min(rows);
            let len = end - start;
            let data_chunk = &data[start * dim..end * dim];
            let norms_chunk = &norms[start..end];
            let assignments = compute_chunk_assignments_only(
                data_chunk,
                norms_chunk,
                len,
                k,
                dim,
                dim,
                centroid_col,
                centroid_norms,
            );
            (start, assignments)
        })
        .collect();

    let mut assignments = vec![0usize; rows];
    for (start, chunk_assignments) in results {
        let end = start + chunk_assignments.len();
        assignments[start..end].copy_from_slice(&chunk_assignments);
    }
    assignments
}

#[allow(clippy::too_many_arguments)]
fn compute_chunk_assignments_only(
    data_chunk: &[f32],
    norms_chunk: &[f32],
    len: usize,
    k: usize,
    dim: usize,
    effective_dim: usize,
    centroid_col: &[f32],
    centroid_norms: &[f32],
) -> Vec<usize> {
    let mut dot_products = vec![0.0f32; len * k];
    unsafe {
        sgemm(
            len,
            effective_dim,
            k,
            1.0,
            data_chunk.as_ptr(),
            dim as isize,
            1,
            centroid_col.as_ptr(),
            k as isize,
            1,
            0.0,
            dot_products.as_mut_ptr(),
            k as isize,
            1,
        );
    }

    let mut assignments = Vec::with_capacity(len);
    for row in 0..len {
        let norm = norms_chunk[row];
        let mut best_cluster = 0usize;
        let mut best_distance = f32::INFINITY;
        for cluster in 0..k {
            let dot = dot_products[row * k + cluster];
            let mut distance = norm + centroid_norms[cluster] - 2.0 * dot;
            if distance < 0.0 {
                distance = 0.0;
            }
            if distance < best_distance {
                best_distance = distance;
                best_cluster = cluster;
            }
        }
        assignments.push(best_cluster);
    }
    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dataset() -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.push(vec![0.0, 0.0]);
            data.push(vec![10.0, 9.5]);
        }
        data
    }

    #[test]
    fn training_indices_are_sampled_and_sorted() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let indices = select_training_indices(10_000, 8, &mut rng);
        assert_eq!(indices.len(), 8 * MAX_POINTS_PER_CENTROID);
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn progressive_schedule_includes_full_dimension() {
        let schedule = progressive_dim_schedule(960);
        assert!(schedule.contains(&960));
        assert!(schedule
            .iter()
            .zip(schedule.iter().skip(1))
            .all(|(a, b)| a < b));
    }

    #[test]
    fn kmeans_converges_on_simple_dataset() {
        let data = simple_dataset();
        let mut rng = StdRng::seed_from_u64(0xBAD5EED);
        let result = run_kmeans(&data, 2, 20, &mut rng);
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), data.len());
        let mut centroids = result.centroids.clone();
        centroids.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        let left = &centroids[0];
        let right = &centroids[1];
        assert!(left[0].abs() < 0.5 && left[1].abs() < 0.5);
        assert!((right[0] - 10.0).abs() < 0.5 && (right[1] - 9.5).abs() < 0.5);
    }

    #[test]
    fn runs_are_deterministic_given_seed() {
        let data = simple_dataset();
        let mut rng1 = StdRng::seed_from_u64(0x1234_5678);
        let mut rng2 = StdRng::seed_from_u64(0x1234_5678);
        let result1 = run_kmeans(&data, 2, 20, &mut rng1);
        let result2 = run_kmeans(&data, 2, 20, &mut rng2);
        assert_eq!(result1.assignments, result2.assignments);
        assert_eq!(result1.centroids, result2.centroids);
    }
}
