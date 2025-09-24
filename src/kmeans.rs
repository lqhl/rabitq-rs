use rand::prelude::*;
use rayon::prelude::*;

use crate::math::l2_distance_sqr;

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

/// Run k-means clustering with k-means++ initialisation.
pub fn run_kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, rng: &mut StdRng) -> KMeansResult {
    assert!(!data.is_empty(), "k-means requires non-empty data");
    assert!(k > 0, "k must be positive");
    let dim = data[0].len();
    assert!(data.iter().all(|v| v.len() == dim));

    let mut centroids = initialise_plus_plus(data, k, rng);
    let mut assignments = vec![0usize; data.len()];

    for _ in 0..max_iter {
        // Assignment step
        let changed = assignments
            .par_iter_mut()
            .enumerate()
            .map(|(idx, assignment)| {
                let (best_cluster, _) = nearest_centroid(&data[idx], &centroids);
                if *assignment != best_cluster {
                    *assignment = best_cluster;
                    1usize
                } else {
                    0usize
                }
            })
            .sum::<usize>();

        if changed == 0 {
            break;
        }

        // Update step
        recompute_centroids(data, &assignments, &mut centroids, rng);
    }

    KMeansResult {
        centroids,
        assignments,
    }
}

fn initialise_plus_plus(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    let mut chosen = vec![false; data.len()];

    let first = rng.gen_range(0..data.len());
    centroids.push(data[first].clone());
    chosen[first] = true;

    while centroids.len() < k {
        let mut distances: Vec<f64> = Vec::with_capacity(data.len());
        for vector in data {
            let (_, dist) = nearest_centroid(vector, &centroids);
            distances.push(dist as f64);
        }
        let dist_sum: f64 = distances.iter().sum();
        if dist_sum <= f64::EPSILON {
            // All points identical; fill remaining centroids with copies.
            while centroids.len() < k {
                centroids.push(centroids[0].clone());
            }
            break;
        }
        let mut target = rng.gen::<f64>() * dist_sum;
        let mut next_idx = 0usize;
        for (idx, weight) in distances.iter().enumerate() {
            target -= *weight;
            if target <= 0.0 {
                next_idx = idx;
                break;
            }
        }
        if chosen[next_idx] {
            next_idx = (0..data.len()).find(|i| !chosen[*i]).unwrap_or(next_idx);
        }
        centroids.push(data[next_idx].clone());
        chosen[next_idx] = true;
    }

    centroids
}

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
    let mut best_cluster = 0usize;
    let mut best_distance = f32::MAX;
    for (cid, centroid) in centroids.iter().enumerate() {
        let dist = l2_distance_sqr(vector, centroid);
        if dist < best_distance {
            best_distance = dist;
            best_cluster = cid;
        }
    }
    (best_cluster, best_distance)
}

fn recompute_centroids(
    data: &[Vec<f32>],
    assignments: &[usize],
    centroids: &mut [Vec<f32>],
    rng: &mut StdRng,
) {
    let k = centroids.len();
    let dim = centroids[0].len();

    let (sums, counts) = data
        .par_iter()
        .zip(assignments.par_iter())
        .fold(
            || (vec![vec![0.0f32; dim]; k], vec![0usize; k]),
            |mut acc, (vector, &cluster)| {
                acc.1[cluster] += 1;
                let cluster_sum = &mut acc.0[cluster];
                for (sum, value) in cluster_sum.iter_mut().zip(vector.iter()) {
                    *sum += *value;
                }
                acc
            },
        )
        .reduce(
            || (vec![vec![0.0f32; dim]; k], vec![0usize; k]),
            |mut left, (right_sums, right_counts)| {
                for (dst, src) in left.0.iter_mut().zip(right_sums.iter()) {
                    for (dst_val, src_val) in dst.iter_mut().zip(src.iter()) {
                        *dst_val += *src_val;
                    }
                }
                for (dst_count, src_count) in left.1.iter_mut().zip(right_counts.iter()) {
                    *dst_count += *src_count;
                }
                left
            },
        );

    for cid in 0..k {
        if counts[cid] == 0 {
            let idx = rng.gen_range(0..data.len());
            centroids[cid] = data[idx].clone();
        } else {
            let inv = 1.0f32 / counts[cid] as f32;
            for d in 0..dim {
                centroids[cid][d] = sums[cid][d] * inv;
            }
        }
    }
}
