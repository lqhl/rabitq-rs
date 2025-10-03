use std::cmp::Ordering;

use matrixmultiply::sgemm;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::RngCore;
use rayon::prelude::*;

const RESEED_CANDIDATES: usize = 8;
const DEFAULT_MAX_POINTS_PER_CENTROID: usize = 256;
const DEFAULT_DECODE_BLOCK_SIZE: usize = 32768;

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub niter: usize,
    pub nredo: usize,
    pub seed: u64,
    pub spherical: bool,
    /// Maximum points sampled per centroid during training
    /// (Faiss default: 256)
    pub max_points_per_centroid: usize,
    /// Batch size for assignment computation (Faiss: decode_block_size)
    /// Larger = better throughput, higher memory; default 32768
    pub decode_block_size: usize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            niter: 25,
            nredo: 1,
            seed: 42,
            spherical: false,
            max_points_per_centroid: DEFAULT_MAX_POINTS_PER_CENTROID,
            decode_block_size: DEFAULT_DECODE_BLOCK_SIZE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub objective: f64,
}

/// Run k-means clustering using a Faiss-inspired pipeline with GEMM-powered
/// assignments and multi-restart for robustness.
pub fn run_kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, rng: &mut StdRng) -> KMeansResult {
    let config = KMeansConfig {
        niter: max_iter,
        nredo: 1,
        seed: rng.next_u64(),
        spherical: false,
        max_points_per_centroid: DEFAULT_MAX_POINTS_PER_CENTROID,
        decode_block_size: DEFAULT_DECODE_BLOCK_SIZE,
    };
    run_kmeans_with_config(data, k, config)
}

/// Run k-means with explicit configuration
pub fn run_kmeans_with_config(data: &[Vec<f32>], k: usize, config: KMeansConfig) -> KMeansResult {
    let dim = validate_inputs(data, k, config.niter);
    let total_points = data.len();

    let flattened = flatten_dataset(data, total_points * dim);

    // Select training subset
    let mut sampling_rng = StdRng::seed_from_u64(config.seed);
    let training_indices = select_training_indices(
        total_points,
        k,
        config.max_points_per_centroid,
        &mut sampling_rng,
    );
    let training_data = gather_rows(&flattened, &training_indices, dim);
    let training_rows = training_indices.len();

    println!(
        "  K-means: {} points, {} clusters, {} iterations, {} restarts",
        training_rows, k, config.niter, config.nredo
    );

    // Multi-restart: run nredo times, pick best by objective
    let mut best_result: Option<KMeansResult> = None;

    for redo_idx in 0..config.nredo {
        let redo_seed = config
            .seed
            .wrapping_add((redo_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let mut redo_rng = StdRng::seed_from_u64(redo_seed);

        // Random Forgy initialization
        let mut centroids =
            initialize_centroids_random(&training_data, training_rows, dim, k, &mut redo_rng);

        let mut training_assignments = vec![0usize; training_rows];
        let norms = compute_norms(&training_data, training_rows, dim);

        // Lloyd iterations
        run_lloyd_iterations(
            &mut centroids,
            config.niter,
            k,
            dim,
            &training_data,
            &norms,
            &mut training_assignments,
            &mut redo_rng,
            config.spherical,
            config.decode_block_size,
        );

        // Assign full dataset and compute objective
        let full_norms = compute_norms(&flattened, total_points, dim);
        let mut centroid_col = Vec::with_capacity(dim * k);
        let mut centroid_norms = Vec::with_capacity(k);
        rebuild_centroid_views(&centroids, k, dim, &mut centroid_col, &mut centroid_norms);

        let assignments = assign_full_dataset(
            &flattened,
            &full_norms,
            k,
            dim,
            &centroid_col,
            &centroid_norms,
            config.decode_block_size,
        );

        let objective = compute_objective(&flattened, &centroids, &assignments, total_points, dim);

        let centroids_vec = centroids.chunks(dim).map(|c| c.to_vec()).collect();
        let result = KMeansResult {
            centroids: centroids_vec,
            assignments,
            objective,
        };

        if redo_idx == 0 {
            println!(
                "    Restart {}/{}: objective = {:.2e}",
                redo_idx + 1,
                config.nredo,
                objective
            );
            best_result = Some(result);
        } else {
            let is_better = objective < best_result.as_ref().unwrap().objective;
            println!(
                "    Restart {}/{}: objective = {:.2e} {}",
                redo_idx + 1,
                config.nredo,
                objective,
                if is_better { "(new best)" } else { "" }
            );
            if is_better {
                best_result = Some(result);
            }
        }
    }

    best_result.unwrap()
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

fn select_training_indices(
    total_points: usize,
    k: usize,
    max_points_per_centroid: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let target = total_points.min(k * max_points_per_centroid).max(k);
    if target == total_points {
        return (0..total_points).collect();
    }

    let mut indices: Vec<usize> = (0..total_points).collect();
    indices.shuffle(rng);
    indices.truncate(target);
    indices.sort_unstable();
    indices
}

/// Random Forgy initialization: pick k random points as initial centroids
fn initialize_centroids_random(
    data: &[f32],
    rows: usize,
    dim: usize,
    k: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..rows).collect();
    indices.shuffle(rng);
    indices.truncate(k);

    let mut centroids = Vec::with_capacity(k * dim);
    for &idx in &indices {
        centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
    }
    centroids
}

fn compute_norms(data: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut norms = vec![0.0f32; rows];
    for (row, norm) in norms.iter_mut().enumerate() {
        let start = row * dim;
        let slice = &data[start..start + dim];
        *norm = slice.iter().map(|v| v * v).sum();
    }
    norms
}

fn compute_objective(
    data: &[f32],
    centroids: &[f32],
    assignments: &[usize],
    rows: usize,
    dim: usize,
) -> f64 {
    let mut total = 0.0f64;
    for row in 0..rows {
        let cluster = assignments[row];
        let point = &data[row * dim..(row + 1) * dim];
        let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
        let mut dist = 0.0f64;
        for d in 0..dim {
            let delta = (point[d] - centroid[d]) as f64;
            dist += delta * delta;
        }
        total += dist;
    }
    total
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

/// Run Lloyd iterations for k-means
#[allow(clippy::too_many_arguments)]
fn run_lloyd_iterations(
    centroids: &mut [f32],
    iterations: usize,
    k: usize,
    dim: usize,
    data: &[f32],
    norms: &[f32],
    assignments: &mut [usize],
    rng: &mut StdRng,
    spherical: bool,
    decode_block_size: usize,
) {
    let mut centroid_col = Vec::with_capacity(dim * k);
    let mut centroid_norms = Vec::with_capacity(k);

    for _iter in 0..iterations {
        rebuild_centroid_views(centroids, k, dim, &mut centroid_col, &mut centroid_norms);

        let summary = assign_points_for_update(
            data,
            norms,
            assignments,
            k,
            dim,
            &centroid_col,
            &centroid_norms,
            decode_block_size,
        );

        update_centroids(centroids, k, dim, data, &summary, rng);

        if spherical {
            normalize_centroids(centroids, k, dim);
        }
    }
}

fn rebuild_centroid_views(
    centroids: &[f32],
    k: usize,
    dim: usize,
    centroid_col: &mut Vec<f32>,
    centroid_norms: &mut Vec<f32>,
) {
    centroid_col.clear();
    centroid_col.resize(dim * k, 0.0);
    centroid_norms.clear();
    centroid_norms.resize(k, 0.0);

    for cluster in 0..k {
        let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
        let mut norm = 0.0f32;
        for d in 0..dim {
            let value = centroid[d];
            centroid_col[d * k + cluster] = value;
            norm += value * value;
        }
        centroid_norms[cluster] = norm;
    }
}

fn normalize_centroids(centroids: &mut [f32], k: usize, dim: usize) {
    for cluster in 0..k {
        let offset = cluster * dim;
        let centroid = &mut centroids[offset..offset + dim];
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
    centroid_col: &[f32],
    centroid_norms: &[f32],
    decode_block_size: usize,
) -> AssignmentSummary {
    let rows = norms.len();
    let num_chunks = (rows + decode_block_size - 1) / decode_block_size;

    let results: Vec<ChunkResult> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * decode_block_size;
            let end = ((chunk_idx + 1) * decode_block_size).min(rows);
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
                centroid_col,
                centroid_norms,
            )
        })
        .collect();

    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * dim];
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
            let dst_offset = cluster * dim;
            let src_offset = cluster * dim;
            for d in 0..dim {
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
    centroid_col: &[f32],
    centroid_norms: &[f32],
) -> ChunkResult {
    let mut dot_products = vec![0.0f32; len * k];
    unsafe {
        sgemm(
            len,
            dim,
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
    let mut sums = vec![0.0f32; k * dim];
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
        let vector = &data_chunk[row * dim..(row + 1) * dim];
        let sum_offset = best_cluster * dim;
        for d in 0..dim {
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
            let sum_offset = cluster * dim;
            for d in 0..dim {
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
    decode_block_size: usize,
) -> Vec<usize> {
    let rows = norms.len();
    let num_chunks = (rows + decode_block_size - 1) / decode_block_size;
    let results: Vec<(usize, Vec<usize>)> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * decode_block_size;
            let end = ((chunk_idx + 1) * decode_block_size).min(rows);
            let len = end - start;
            let data_chunk = &data[start * dim..end * dim];
            let norms_chunk = &norms[start..end];
            let assignments = compute_chunk_assignments_only(
                data_chunk,
                norms_chunk,
                len,
                k,
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
    centroid_col: &[f32],
    centroid_norms: &[f32],
) -> Vec<usize> {
    let mut dot_products = vec![0.0f32; len * k];
    unsafe {
        sgemm(
            len,
            dim,
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
        let indices = select_training_indices(10_000, 8, 256, &mut rng);
        assert_eq!(indices.len(), 8 * 256);
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn training_indices_respect_max_points_per_centroid() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        // Custom sampling: 64 points per centroid
        let indices = select_training_indices(1_000_000, 4096, 64, &mut rng);
        assert_eq!(indices.len(), 4096 * 64);

        // Custom sampling: 128 points per centroid
        let indices = select_training_indices(1_000_000, 1024, 128, &mut rng);
        assert_eq!(indices.len(), 1024 * 128);

        // Default Faiss: 256 points per centroid
        let indices = select_training_indices(1_000_000, 512, 256, &mut rng);
        assert_eq!(indices.len(), 512 * 256);
    }

    #[test]
    fn kmeans_converges_on_simple_dataset() {
        let data = simple_dataset();
        let config = KMeansConfig {
            niter: 20,
            nredo: 3,
            seed: 0xBAD5EED,
            spherical: false,
            max_points_per_centroid: DEFAULT_MAX_POINTS_PER_CENTROID,
            decode_block_size: DEFAULT_DECODE_BLOCK_SIZE,
        };
        let result = run_kmeans_with_config(&data, 2, config);
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), data.len());
        assert!(result.objective >= 0.0);
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
        let config1 = KMeansConfig {
            niter: 20,
            nredo: 1,
            seed: 0x1234_5678,
            spherical: false,
            max_points_per_centroid: DEFAULT_MAX_POINTS_PER_CENTROID,
            decode_block_size: DEFAULT_DECODE_BLOCK_SIZE,
        };
        let config2 = KMeansConfig {
            niter: 20,
            nredo: 1,
            seed: 0x1234_5678,
            spherical: false,
            max_points_per_centroid: DEFAULT_MAX_POINTS_PER_CENTROID,
            decode_block_size: DEFAULT_DECODE_BLOCK_SIZE,
        };
        let result1 = run_kmeans_with_config(&data, 2, config1);
        let result2 = run_kmeans_with_config(&data, 2, config2);
        assert_eq!(result1.assignments, result2.assignments);
        assert_eq!(result1.centroids, result2.centroids);
        assert_eq!(result1.objective, result2.objective);
    }
}
