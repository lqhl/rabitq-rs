use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::prelude::*;

use crate::kmeans::{run_kmeans, KMeansResult};
use crate::math::{dot, l2_distance_sqr};
use crate::quantizer::{quantize_with_centroid, QuantizedVector};
use crate::rotation::RandomRotator;
use crate::{Metric, RabitqError};

/// Parameters for IVF search.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub top_k: usize,
    pub nprobe: usize,
}

impl SearchParams {
    pub fn new(top_k: usize, nprobe: usize) -> Self {
        Self { top_k, nprobe }
    }
}

/// Result entry returned by IVF search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct SearchDiagnostics {
    pub estimated: usize,
    pub skipped_by_lower_bound: usize,
    pub extended_evaluations: usize,
}

#[derive(Debug, Clone)]
struct ClusterEntry {
    centroid: Vec<f32>,
    ids: Vec<usize>,
    vectors: Vec<QuantizedVector>,
}

impl ClusterEntry {
    fn new(centroid: Vec<f32>) -> Self {
        Self {
            centroid,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct HeapCandidate {
    id: usize,
    distance: f32,
    score: f32,
}

#[derive(Debug, Clone)]
struct HeapEntry {
    candidate: HeapCandidate,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.candidate
            .distance
            .to_bits()
            .eq(&other.candidate.distance.to_bits())
            && self.candidate.id == other.candidate.id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.candidate.distance.total_cmp(&other.candidate.distance)
    }
}

/// IVF + RaBitQ index implemented in Rust.
#[derive(Debug, Clone)]
pub struct IvfRabitqIndex {
    dim: usize,
    metric: Metric,
    rotator: RandomRotator,
    clusters: Vec<ClusterEntry>,
    ex_bits: usize,
}

impl IvfRabitqIndex {
    /// Train a new index from the provided dataset.
    pub fn train(
        data: &[Vec<f32>],
        nlist: usize,
        total_bits: usize,
        metric: Metric,
        seed: u64,
    ) -> Result<Self, RabitqError> {
        if data.is_empty() {
            return Err(RabitqError::InvalidConfig(
                "training data must be non-empty",
            ));
        }
        if nlist == 0 {
            return Err(RabitqError::InvalidConfig("nlist must be positive"));
        }
        if total_bits == 0 || total_bits > 16 {
            return Err(RabitqError::InvalidConfig(
                "total_bits must be between 1 and 16",
            ));
        }

        let dim = data[0].len();
        if data.iter().any(|v| v.len() != dim) {
            return Err(RabitqError::InvalidConfig(
                "input vectors must share the same dimension",
            ));
        }
        if nlist > data.len() {
            return Err(RabitqError::InvalidConfig(
                "nlist cannot exceed number of vectors",
            ));
        }

        let rotator = RandomRotator::new(dim, seed);
        let rotated_data: Vec<Vec<f32>> = data.iter().map(|v| rotator.rotate(v)).collect();

        let mut rng = StdRng::seed_from_u64(seed ^ 0x5a5a_5a5a5a5a5a5a);
        let KMeansResult {
            centroids,
            assignments,
        } = run_kmeans(&rotated_data, nlist, 30, &mut rng);

        Self::build_from_rotated(
            dim,
            metric,
            rotator,
            centroids,
            &rotated_data,
            &assignments,
            total_bits,
        )
    }

    /// Train an index using externally provided centroids and cluster assignments.
    pub fn train_with_clusters(
        data: &[Vec<f32>],
        centroids: &[Vec<f32>],
        assignments: &[usize],
        total_bits: usize,
        metric: Metric,
        seed: u64,
    ) -> Result<Self, RabitqError> {
        if data.is_empty() {
            return Err(RabitqError::InvalidConfig(
                "training data must be non-empty",
            ));
        }
        if centroids.is_empty() {
            return Err(RabitqError::InvalidConfig("centroids must be non-empty"));
        }
        if assignments.len() != data.len() {
            return Err(RabitqError::InvalidConfig(
                "assignments length must match data length",
            ));
        }
        if total_bits == 0 || total_bits > 16 {
            return Err(RabitqError::InvalidConfig(
                "total_bits must be between 1 and 16",
            ));
        }

        let dim = data[0].len();
        if data.iter().any(|v| v.len() != dim) {
            return Err(RabitqError::InvalidConfig(
                "input vectors must share the same dimension",
            ));
        }
        if centroids.iter().any(|c| c.len() != dim) {
            return Err(RabitqError::InvalidConfig(
                "centroids must match the data dimensionality",
            ));
        }

        let nlist = centroids.len();
        if nlist == 0 {
            return Err(RabitqError::InvalidConfig("nlist must be positive"));
        }
        if nlist > data.len() {
            return Err(RabitqError::InvalidConfig(
                "nlist cannot exceed number of vectors",
            ));
        }
        if assignments.iter().any(|&cid| cid >= nlist) {
            return Err(RabitqError::InvalidConfig(
                "assignments reference invalid cluster ids",
            ));
        }

        let rotator = RandomRotator::new(dim, seed);
        let rotated_data: Vec<Vec<f32>> = data.iter().map(|v| rotator.rotate(v)).collect();
        let rotated_centroids: Vec<Vec<f32>> =
            centroids.iter().map(|c| rotator.rotate(c)).collect();

        Self::build_from_rotated(
            dim,
            metric,
            rotator,
            rotated_centroids,
            &rotated_data,
            assignments,
            total_bits,
        )
    }

    fn build_from_rotated(
        dim: usize,
        metric: Metric,
        rotator: RandomRotator,
        rotated_centroids: Vec<Vec<f32>>,
        rotated_data: &[Vec<f32>],
        assignments: &[usize],
        total_bits: usize,
    ) -> Result<Self, RabitqError> {
        if assignments.len() != rotated_data.len() {
            return Err(RabitqError::InvalidConfig(
                "assignments length must match number of vectors",
            ));
        }
        if total_bits == 0 || total_bits > 16 {
            return Err(RabitqError::InvalidConfig(
                "total_bits must be between 1 and 16",
            ));
        }

        let mut clusters: Vec<ClusterEntry> = rotated_centroids
            .into_iter()
            .map(ClusterEntry::new)
            .collect();
        if clusters.is_empty() {
            return Err(RabitqError::InvalidConfig("nlist must be positive"));
        }

        for (idx, rotated_vec) in rotated_data.iter().enumerate() {
            let cluster_id = assignments[idx];
            let cluster = clusters
                .get_mut(cluster_id)
                .ok_or(RabitqError::InvalidConfig(
                    "assignments reference invalid cluster ids",
                ))?;
            let quantized =
                quantize_with_centroid(rotated_vec, &cluster.centroid, total_bits, metric);
            cluster.ids.push(idx);
            cluster.vectors.push(quantized);
        }

        Ok(Self {
            dim,
            metric,
            rotator,
            clusters,
            ex_bits: total_bits.saturating_sub(1),
        })
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.clusters.iter().map(|c| c.ids.len()).sum()
    }

    /// Check whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Search for the nearest neighbours of the provided query vector.
    pub fn search(
        &self,
        query: &[f32],
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, RabitqError> {
        self.search_internal(query, params, None)
    }

    fn search_internal(
        &self,
        query: &[f32],
        params: SearchParams,
        mut diagnostics: Option<&mut SearchDiagnostics>,
    ) -> Result<Vec<SearchResult>, RabitqError> {
        if self.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let rotated_query = self.rotator.rotate(query);
        let mut cluster_scores: Vec<(usize, f32)> = Vec::with_capacity(self.clusters.len());
        for (cid, cluster) in self.clusters.iter().enumerate() {
            let score = match self.metric {
                Metric::L2 => l2_distance_sqr(&rotated_query, &cluster.centroid),
                Metric::InnerProduct => dot(&rotated_query, &cluster.centroid),
            };
            cluster_scores.push((cid, score));
        }

        match self.metric {
            Metric::L2 => cluster_scores.sort_by(|a, b| a.1.total_cmp(&b.1)),
            Metric::InnerProduct => cluster_scores.sort_by(|a, b| b.1.total_cmp(&a.1)),
        }

        let nprobe = params.nprobe.max(1).min(self.clusters.len());
        if params.top_k == 0 {
            return Ok(Vec::new());
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let sum_query: f32 = rotated_query.iter().sum();
        let query_norm: f32 = rotated_query.iter().map(|v| v * v).sum::<f32>().sqrt();
        let c1 = -0.5f32;
        let binary_scale = (1 << self.ex_bits) as f32;
        let cb = -((1 << self.ex_bits) as f32 - 0.5);

        for &(cid, _) in cluster_scores.iter().take(nprobe) {
            let cluster = &self.clusters[cid];
            let centroid_dist = l2_distance_sqr(&rotated_query, &cluster.centroid);
            let dot_query_centroid = dot(&rotated_query, &cluster.centroid);
            let g_add = match self.metric {
                Metric::L2 => centroid_dist,
                Metric::InnerProduct => -dot_query_centroid,
            };
            let centroid_norm = centroid_dist.sqrt();
            let g_error = centroid_norm;

            for (local_idx, quantized) in cluster.vectors.iter().enumerate() {
                let mut binary_dot = 0.0f32;
                for (&bit, &q_val) in quantized.binary_code.iter().zip(rotated_query.iter()) {
                    binary_dot += (bit as f32) * q_val;
                }
                let binary_term = binary_dot + c1 * sum_query;
                let est_distance = quantized.f_add + g_add + quantized.f_rescale * binary_term;
                let mut lower_bound = est_distance - quantized.f_error * g_error;
                let safe_bound = match self.metric {
                    Metric::L2 => {
                        let diff = (centroid_norm - quantized.residual_norm).max(0.0);
                        diff * diff
                    }
                    Metric::InnerProduct => {
                        let max_dot = dot_query_centroid + query_norm * quantized.residual_norm;
                        -max_dot
                    }
                };
                if !lower_bound.is_finite() {
                    lower_bound = safe_bound;
                } else {
                    lower_bound = lower_bound.min(safe_bound);
                }

                let distk = if heap.len() < params.top_k {
                    f32::INFINITY
                } else {
                    heap.peek()
                        .map(|entry| entry.candidate.distance)
                        .unwrap_or(f32::INFINITY)
                };

                if lower_bound >= distk {
                    if let Some(diag) = diagnostics.as_deref_mut() {
                        diag.skipped_by_lower_bound += 1;
                    }
                    continue;
                }

                if self.ex_bits > 0 {
                    if let Some(diag) = diagnostics.as_deref_mut() {
                        diag.extended_evaluations += 1;
                    }
                }

                let mut distance = est_distance;
                let mut score = match self.metric {
                    Metric::L2 => distance,
                    Metric::InnerProduct => -distance,
                };

                if self.ex_bits > 0 {
                    let mut ex_dot = 0.0f32;
                    for (&code, &q_val) in quantized.ex_code.iter().zip(rotated_query.iter()) {
                        ex_dot += (code as f32) * q_val;
                    }
                    let total_term = binary_scale * binary_dot + ex_dot + cb * sum_query;
                    distance = quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term;
                    score = match self.metric {
                        Metric::L2 => distance,
                        Metric::InnerProduct => -distance,
                    };
                }

                if !distance.is_finite() {
                    continue;
                }

                if let Some(diag) = diagnostics.as_deref_mut() {
                    diag.estimated += 1;
                }

                heap.push(HeapEntry {
                    candidate: HeapCandidate {
                        id: cluster.ids[local_idx],
                        distance,
                        score,
                    },
                });

                if heap.len() > params.top_k {
                    heap.pop();
                }
            }
        }

        let mut candidates: Vec<HeapCandidate> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.candidate)
            .collect();

        match self.metric {
            Metric::L2 => candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance)),
            Metric::InnerProduct => candidates.sort_by(|a, b| b.score.total_cmp(&a.score)),
        }

        Ok(candidates
            .into_iter()
            .map(|candidate| SearchResult {
                id: candidate.id,
                score: match self.metric {
                    Metric::L2 => candidate.distance,
                    Metric::InnerProduct => candidate.score,
                },
            })
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn search_with_diagnostics(
        &self,
        query: &[f32],
        params: SearchParams,
    ) -> Result<(Vec<SearchResult>, SearchDiagnostics), RabitqError> {
        let mut diagnostics = SearchDiagnostics::default();
        let results = self.search_internal(query, params, Some(&mut diagnostics))?;
        Ok((results, diagnostics))
    }

    #[cfg(test)]
    pub(crate) fn search_naive(
        &self,
        query: &[f32],
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, RabitqError> {
        if self.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let rotated_query = self.rotator.rotate(query);
        let mut cluster_scores: Vec<(usize, f32)> = Vec::with_capacity(self.clusters.len());
        for (cid, cluster) in self.clusters.iter().enumerate() {
            let score = match self.metric {
                Metric::L2 => l2_distance_sqr(&rotated_query, &cluster.centroid),
                Metric::InnerProduct => dot(&rotated_query, &cluster.centroid),
            };
            cluster_scores.push((cid, score));
        }

        match self.metric {
            Metric::L2 => cluster_scores.sort_by(|a, b| a.1.total_cmp(&b.1)),
            Metric::InnerProduct => cluster_scores.sort_by(|a, b| b.1.total_cmp(&a.1)),
        }

        let nprobe = params.nprobe.max(1).min(self.clusters.len());
        let mut candidates = Vec::new();
        let sum_query: f32 = rotated_query.iter().sum();
        let c1 = -0.5f32;
        let binary_scale = (1 << self.ex_bits) as f32;
        let cb = -((1 << self.ex_bits) as f32 - 0.5);
        for &(cid, _) in cluster_scores.iter().take(nprobe) {
            let cluster = &self.clusters[cid];
            let centroid_dist = l2_distance_sqr(&rotated_query, &cluster.centroid);
            let dot_query_centroid = dot(&rotated_query, &cluster.centroid);
            let g_add = match self.metric {
                Metric::L2 => centroid_dist,
                Metric::InnerProduct => -dot_query_centroid,
            };
            for (local_idx, quantized) in cluster.vectors.iter().enumerate() {
                let mut binary_dot = 0.0f32;
                for (&bit, &q_val) in quantized.binary_code.iter().zip(rotated_query.iter()) {
                    binary_dot += (bit as f32) * q_val;
                }
                let binary_term = binary_dot + c1 * sum_query;
                let mut distance = quantized.f_add + g_add + quantized.f_rescale * binary_term;
                if self.ex_bits > 0 {
                    let mut ex_dot = 0.0f32;
                    for (&code, &q_val) in quantized.ex_code.iter().zip(rotated_query.iter()) {
                        ex_dot += (code as f32) * q_val;
                    }
                    let total_term = binary_scale * binary_dot + ex_dot + cb * sum_query;
                    distance = quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term;
                }
                if !distance.is_finite() {
                    continue;
                }
                let score = match self.metric {
                    Metric::L2 => distance,
                    Metric::InnerProduct => -distance,
                };
                candidates.push(SearchResult {
                    id: cluster.ids[local_idx],
                    score,
                });
            }
        }

        match self.metric {
            Metric::L2 => candidates.sort_by(|a, b| a.score.total_cmp(&b.score)),
            Metric::InnerProduct => candidates.sort_by(|a, b| b.score.total_cmp(&a.score)),
        }

        candidates.truncate(params.top_k.min(candidates.len()));
        Ok(candidates)
    }
}
