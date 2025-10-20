use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher;
use instant_distance::{Builder, HnswMap, Point, Search};
use rand::prelude::*;
use rayon::prelude::*;

use crate::ivf::{
    metric_to_tag, read_f32, read_u32, read_u64, read_u8, tag_to_metric, unpack_binary_code,
    unpack_ex_code, write_f32, write_u32, write_u64,
};
use crate::kmeans::{run_kmeans, KMeansResult};
use crate::math::{dot, l2_distance_sqr};
use crate::quantizer::{quantize_with_centroid, QuantizedVector, RabitqConfig};
use crate::rotation::{DynamicRotator, RotatorType};
use crate::{Metric, RabitqError};

/// Configuration for MSTG index
#[derive(Debug, Clone, Copy)]
pub struct MstgConfig {
    /// Branching factor for the multi-level k-means tree (default: 16)
    pub k: usize,
    /// Target vectors per cluster at the bottom level (default: 100)
    pub target_cluster_size: usize,
    /// RaBitQ quantization bits (default: 7)
    pub total_bits: usize,
    /// Use faster RaBitQ configuration (default: false)
    pub use_faster_config: bool,
    /// HNSW M parameter (default: 16)
    pub hnsw_m: usize,
    /// HNSW ef_construction (default: 200)
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search - controls search quality (default: 400)
    /// Higher values = better recall but slower search
    /// Should be >= max expected nprobe value
    pub hnsw_ef_search: usize,
}

impl Default for MstgConfig {
    fn default() -> Self {
        Self {
            k: 16,
            target_cluster_size: 100,
            total_bits: 7,
            use_faster_config: false,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 1000, // High ef_search for better recall (HNSW sets this internally)
        }
    }
}

/// Parameters for MSTG search
#[derive(Debug, Clone, Copy)]
pub struct MstgSearchParams {
    /// Number of top-k results to return
    pub top_k: usize,
    /// Number of clusters to probe (passed to HNSW search as ef_search)
    pub nprobe: usize,
}

impl MstgSearchParams {
    pub fn new(top_k: usize, nprobe: usize) -> Self {
        Self { top_k, nprobe }
    }
}

/// Search result for MSTG
pub type MstgSearchResult = crate::ivf::SearchResult;

/// Medoid wrapper for HNSW
#[derive(Debug, Clone)]
struct Medoid {
    cluster_id: usize,
    vector: Vec<f32>,
}

impl Point for Medoid {
    fn distance(&self, other: &Self) -> f32 {
        // Use L2 distance for medoid selection in HNSW
        l2_distance_sqr(&self.vector, &other.vector).sqrt()
    }
}

/// Bottom-level cluster containing quantized vectors
#[derive(Debug, Clone)]
struct MstgCluster {
    /// Medoid (representative) vector
    medoid: Vec<f32>,
    /// Vector IDs in this cluster
    ids: Vec<usize>,
    /// Quantized vectors
    vectors: Vec<QuantizedVector>,
}

impl MstgCluster {
    fn new(medoid: Vec<f32>) -> Self {
        Self {
            medoid,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }
}

/// MSTG (Multi-Scale Tree Graph) + RaBitQ index
pub struct MstgRabitqIndex {
    dim: usize,
    padded_dim: usize,
    metric: Metric,
    rotator: DynamicRotator,
    clusters: Vec<MstgCluster>,
    hnsw: HnswMap<Medoid, usize>,
    ex_bits: usize,
}

impl MstgRabitqIndex {
    /// Train a new MSTG index from scratch
    pub fn train(
        data: &[Vec<f32>],
        config: MstgConfig,
        metric: Metric,
        rotator_type: RotatorType,
        seed: u64,
    ) -> Result<Self, RabitqError> {
        if data.is_empty() {
            return Err(RabitqError::InvalidConfig(
                "training data must be non-empty",
            ));
        }
        if config.k < 2 {
            return Err(RabitqError::InvalidConfig("k must be >= 2"));
        }
        if config.target_cluster_size == 0 {
            return Err(RabitqError::InvalidConfig(
                "target_cluster_size must be positive",
            ));
        }
        if config.total_bits == 0 || config.total_bits > 16 {
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

        let n = data.len();
        let target_num_clusters = (n / config.target_cluster_size).max(1);

        println!("Training MSTG index:");
        println!("  Data: {} vectors, {} dimensions", n, dim);
        println!("  Target clusters: {}", target_num_clusters);
        println!("  Branching factor K: {}", config.k);

        // Rotate all vectors once
        let rotator = DynamicRotator::new(dim, rotator_type, seed);
        let padded_dim = rotator.padded_dim();
        println!("Rotating vectors...");
        let rotated_data: Vec<Vec<f32>> = data.par_iter().map(|v| rotator.rotate(v)).collect();

        // Multi-level k-means tree construction
        println!("Building multi-level k-means tree...");
        let (bottom_centroids, bottom_assignments) =
            Self::build_multilevel_kmeans(&rotated_data, config.k, target_num_clusters, seed)?;

        println!(
            "Tree construction complete. Bottom level: {} clusters",
            bottom_centroids.len()
        );

        // Extract medoids and build HNSW
        println!("Extracting medoids and building HNSW index...");
        let (clusters, medoids) = Self::build_clusters_and_medoids(
            &rotated_data,
            &bottom_centroids,
            &bottom_assignments,
            padded_dim,
        )?;

        let hnsw = Self::build_hnsw(
            &medoids,
            config.hnsw_m,
            config.hnsw_ef_construction,
            config.hnsw_ef_search,
        )?;
        println!(
            "HNSW index built with {} medoids (ef_search={})",
            medoids.len(),
            config.hnsw_ef_search
        );

        // Quantize vectors in each cluster
        println!("Quantizing vectors with RaBitQ...");
        let rabitq_config = if config.use_faster_config {
            RabitqConfig::faster(padded_dim, config.total_bits, seed)
        } else {
            RabitqConfig::new(config.total_bits)
        };

        let clusters = Self::quantize_clusters(
            clusters,
            &rotated_data,
            &bottom_assignments,
            &rabitq_config,
            metric,
        )?;
        println!("Quantization complete");

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            clusters,
            hnsw,
            ex_bits: config.total_bits.saturating_sub(1),
        })
    }

    /// Build multi-level k-means tree (recursive splitting)
    fn build_multilevel_kmeans(
        data: &[Vec<f32>],
        k: usize,
        target_num_clusters: usize,
        seed: u64,
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>), RabitqError> {
        let n = data.len();

        // Calculate conservative max depth
        // We'll use early stopping to avoid overshooting
        let max_depth = ((n as f64).ln() / (k as f64).ln()).ceil() as usize;

        println!(
            "  Multi-level tree: target={}, max_depth={}",
            target_num_clusters, max_depth
        );

        // Start with all data in one cluster
        let mut current_level: Vec<(Vec<usize>, Vec<f32>)> = vec![(
            (0..n).collect(),
            Self::compute_centroid(data, &(0..n).collect::<Vec<_>>()),
        )];

        let _rng = StdRng::seed_from_u64(seed);

        for level in 0..max_depth {
            println!("  Level {}: {} clusters", level, current_level.len());

            // Early stopping BEFORE splitting if we're close to target
            // Allow some overshoot (1.5x) but stop if we'd create too many clusters
            let estimated_next_level_size = current_level.len() * k;
            if current_level.len() >= target_num_clusters / 2
                && estimated_next_level_size > target_num_clusters * 2
            {
                println!(
                    "  Early stop: current={}, estimated_next={}, target={}",
                    current_level.len(),
                    estimated_next_level_size,
                    target_num_clusters
                );
                break;
            }

            let mut next_level = Vec::new();

            // Process each cluster in parallel
            let split_results: Vec<Vec<(Vec<usize>, Vec<f32>)>> = current_level
                .par_iter()
                .map(|(indices, _)| {
                    if indices.len() <= k {
                        // Too small to split, keep as-is
                        vec![(indices.clone(), Self::compute_centroid(data, indices))]
                    } else {
                        // Run k-means on this subset
                        let subset: Vec<Vec<f32>> =
                            indices.iter().map(|&idx| data[idx].clone()).collect();
                        let mut local_rng =
                            StdRng::seed_from_u64(seed ^ (level as u64) ^ (indices[0] as u64));
                        let k_to_use = k.min(indices.len());
                        let KMeansResult {
                            centroids,
                            assignments,
                            ..
                        } = run_kmeans(&subset, k_to_use, 10, &mut local_rng);

                        // Group indices by assignment
                        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k_to_use];
                        for (local_idx, &cluster_id) in assignments.iter().enumerate() {
                            groups[cluster_id].push(indices[local_idx]);
                        }

                        groups
                            .into_iter()
                            .zip(centroids.into_iter())
                            .filter(|(g, _)| !g.is_empty())
                            .collect()
                    }
                })
                .collect();

            for split_result in split_results {
                next_level.extend(split_result);
            }

            current_level = next_level;

            // Also check after splitting
            if current_level.len() >= target_num_clusters {
                break;
            }
        }

        // Extract final centroids and assignments
        let mut assignments = vec![0usize; n];
        let centroids: Vec<Vec<f32>> = current_level
            .iter()
            .enumerate()
            .map(|(cluster_id, (indices, centroid))| {
                for &idx in indices {
                    assignments[idx] = cluster_id;
                }
                centroid.clone()
            })
            .collect();

        Ok((centroids, assignments))
    }

    fn compute_centroid(data: &[Vec<f32>], indices: &[usize]) -> Vec<f32> {
        if indices.is_empty() {
            return Vec::new();
        }
        let dim = data[0].len();
        let mut centroid = vec![0.0; dim];
        for &idx in indices {
            for (i, &val) in data[idx].iter().enumerate() {
                centroid[i] += val;
            }
        }
        let count = indices.len() as f32;
        for val in &mut centroid {
            *val /= count;
        }
        centroid
    }

    fn build_clusters_and_medoids(
        data: &[Vec<f32>],
        centroids: &[Vec<f32>],
        assignments: &[usize],
        _padded_dim: usize,
    ) -> Result<(Vec<MstgCluster>, Vec<Medoid>), RabitqError> {
        let num_clusters = centroids.len();
        let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];

        for (idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id >= num_clusters {
                return Err(RabitqError::InvalidConfig("invalid cluster assignment"));
            }
            cluster_indices[cluster_id].push(idx);
        }

        // For each cluster, find the medoid (closest vector to centroid)
        let clusters_and_medoids: Vec<(MstgCluster, Medoid)> = centroids
            .par_iter()
            .enumerate()
            .map(|(cluster_id, centroid)| {
                let indices = &cluster_indices[cluster_id];

                // Find medoid: vector closest to centroid
                let (_medoid_idx, medoid_vec) = if indices.is_empty() {
                    // Empty cluster, use centroid itself
                    (0, centroid.clone())
                } else {
                    let mut best_idx = indices[0];
                    let mut best_dist = l2_distance_sqr(&data[indices[0]], centroid);

                    for &idx in indices.iter().skip(1) {
                        let dist = l2_distance_sqr(&data[idx], centroid);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = idx;
                        }
                    }

                    (best_idx, data[best_idx].clone())
                };

                let cluster = MstgCluster::new(centroid.clone());
                let medoid = Medoid {
                    cluster_id,
                    vector: medoid_vec,
                };

                (cluster, medoid)
            })
            .collect();

        let (clusters, medoids) = clusters_and_medoids.into_iter().unzip();
        Ok((clusters, medoids))
    }

    fn build_hnsw(
        medoids: &[Medoid],
        _m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<HnswMap<Medoid, usize>, RabitqError> {
        // Build HnswMap: points are Medoids, values are cluster IDs
        let cluster_ids: Vec<usize> = medoids.iter().map(|m| m.cluster_id).collect();
        let hnsw = Builder::default()
            .ef_construction(ef_construction)
            .ef_search(ef_search)
            .build(medoids.to_vec(), cluster_ids);
        Ok(hnsw)
    }

    fn quantize_clusters(
        mut clusters: Vec<MstgCluster>,
        data: &[Vec<f32>],
        assignments: &[usize],
        config: &RabitqConfig,
        metric: Metric,
    ) -> Result<Vec<MstgCluster>, RabitqError> {
        // Group indices by cluster using the provided assignments
        let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
        for (idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id >= clusters.len() {
                return Err(RabitqError::InvalidConfig("invalid cluster assignment"));
            }
            cluster_indices[cluster_id].push(idx);
        }

        // Quantize each cluster in parallel
        let cluster_data: Vec<(Vec<usize>, Vec<QuantizedVector>)> = clusters
            .par_iter()
            .enumerate()
            .map(|(cluster_id, cluster)| {
                let indices = &cluster_indices[cluster_id];
                let quantized_vectors: Vec<QuantizedVector> = indices
                    .par_iter()
                    .map(|&idx| quantize_with_centroid(&data[idx], &cluster.medoid, config, metric))
                    .collect();
                (indices.clone(), quantized_vectors)
            })
            .collect();

        // Populate clusters
        for (cluster_id, (indices, quantized_vectors)) in cluster_data.into_iter().enumerate() {
            clusters[cluster_id].ids = indices;
            clusters[cluster_id].vectors = quantized_vectors;
        }

        Ok(clusters)
    }

    /// Search for nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        params: MstgSearchParams,
    ) -> Result<Vec<MstgSearchResult>, RabitqError> {
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.clusters.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }

        // Rotate query
        let rotated_query = self.rotator.rotate(query);

        // Use HNSW to find nearest clusters (via medoids)
        let query_medoid = Medoid {
            cluster_id: usize::MAX, // Dummy ID for query
            vector: rotated_query.clone(),
        };

        // Use HNSW to find nearest clusters
        // Note: The instant-distance library's HNSW.search() internally sets search.ef
        // to the ef_search value configured in the Builder. We set a high default (1000)
        // to ensure good recall across different nprobe values. The HNSW will return up to
        // ef_search neighbors, which we then filter down to nprobe below.
        let mut search = Search::default();
        let neighbors: Vec<_> = self.hnsw.search(&query_medoid, &mut search).collect();

        // Take top nprobe clusters
        let nprobe = params.nprobe.min(neighbors.len()).min(self.clusters.len());
        let cluster_ids: Vec<usize> = neighbors
            .into_iter()
            .take(nprobe)
            .map(|item| *item.value)
            .collect();

        // Debug: check if we got valid cluster IDs
        if cluster_ids.iter().any(|&id| id >= self.clusters.len()) {
            return Err(RabitqError::InvalidConfig(
                "HNSW returned invalid cluster ID",
            ));
        }

        // Search within selected clusters using RaBitQ
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        for &cluster_id in &cluster_ids {
            let cluster = &self.clusters[cluster_id];
            let centroid_dist = l2_distance_sqr(&rotated_query, &cluster.medoid);
            let dot_query_centroid = dot(&rotated_query, &cluster.medoid);
            let g_add = match self.metric {
                Metric::L2 => centroid_dist,
                Metric::InnerProduct => -dot_query_centroid,
            };

            for (local_idx, quantized) in cluster.vectors.iter().enumerate() {
                let vector_id = cluster.ids[local_idx];

                // Binary code evaluation (simplified from IVF implementation)
                let binary_code = quantized.unpack_binary_code();
                let mut binary_dot = 0.0f32;
                for (&bit, &q_val) in binary_code.iter().zip(rotated_query.iter()) {
                    binary_dot += (bit as f32) * q_val;
                }
                let sum_query: f32 = rotated_query.iter().sum();
                let c1 = -0.5f32;
                let binary_term = binary_dot + c1 * sum_query;
                let mut distance = quantized.f_add + g_add + quantized.f_rescale * binary_term;

                // Extended code evaluation if needed
                if self.ex_bits > 0 {
                    let ex_code = quantized.unpack_ex_code();
                    let mut ex_dot = 0.0f32;
                    for (&code, &q_val) in ex_code.iter().zip(rotated_query.iter()) {
                        ex_dot += (code as f32) * q_val;
                    }
                    let binary_scale = (1 << self.ex_bits) as f32;
                    let cb = -((1 << self.ex_bits) as f32 - 0.5);
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

                heap.push(HeapEntry {
                    id: vector_id,
                    distance,
                    score,
                });

                if heap.len() > params.top_k {
                    heap.pop();
                }
            }
        }

        // Extract results
        let mut results: Vec<MstgSearchResult> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|entry| MstgSearchResult {
                id: entry.id,
                score: match self.metric {
                    Metric::L2 => entry.distance,
                    Metric::InnerProduct => entry.score,
                },
            })
            .collect();

        match self.metric {
            Metric::L2 => results.sort_by(|a, b| a.score.total_cmp(&b.score)),
            Metric::InnerProduct => results.sort_by(|a, b| b.score.total_cmp(&a.score)),
        }

        results.truncate(params.top_k);
        Ok(results)
    }

    pub fn len(&self) -> usize {
        self.clusters.iter().map(|c| c.ids.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Save the MSTG index to a file
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), RabitqError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.save_to_writer(writer)
    }

    /// Save the MSTG index to a writer
    pub fn save_to_writer<W: Write>(&self, writer: W) -> Result<(), RabitqError> {
        let mut writer = BufWriter::new(writer);

        // Magic and version
        const PERSIST_MAGIC: [u8; 4] = *b"MSTG";
        const PERSIST_VERSION: u32 = 1;
        writer.write_all(&PERSIST_MAGIC)?;
        write_u32(&mut writer, PERSIST_VERSION, None)?;

        let mut hasher = Hasher::new();

        // Basic metadata
        write_u32(&mut writer, self.dim as u32, Some(&mut hasher))?;
        write_u32(&mut writer, self.padded_dim as u32, Some(&mut hasher))?;
        writer.write_all(&[metric_to_tag(self.metric)])?;
        hasher.update(&[metric_to_tag(self.metric)]);

        let rotator_type_tag = self.rotator.rotator_type() as u8;
        writer.write_all(&[rotator_type_tag])?;
        hasher.update(&[rotator_type_tag]);

        writer.write_all(&[self.ex_bits as u8])?;
        hasher.update(&[self.ex_bits as u8]);

        // Save rotator
        let rotator_data = self.rotator.serialize();
        write_u64(&mut writer, rotator_data.len() as u64, Some(&mut hasher))?;
        writer.write_all(&rotator_data)?;
        hasher.update(&rotator_data);

        // Save clusters
        write_u64(&mut writer, self.clusters.len() as u64, Some(&mut hasher))?;

        for cluster in &self.clusters {
            // Save medoid
            for &val in &cluster.medoid {
                write_f32(&mut writer, val, Some(&mut hasher))?;
            }

            // Save ids and vectors
            write_u64(&mut writer, cluster.ids.len() as u64, Some(&mut hasher))?;
            for &id in &cluster.ids {
                write_u64(&mut writer, id as u64, Some(&mut hasher))?;
            }

            for vector in &cluster.vectors {
                // Save packed codes
                writer.write_all(&vector.binary_code_packed)?;
                hasher.update(&vector.binary_code_packed);
                writer.write_all(&vector.ex_code_packed)?;
                hasher.update(&vector.ex_code_packed);

                // Save metadata
                write_f32(&mut writer, vector.delta, Some(&mut hasher))?;
                write_f32(&mut writer, vector.vl, Some(&mut hasher))?;
                write_f32(&mut writer, vector.f_add, Some(&mut hasher))?;
                write_f32(&mut writer, vector.f_rescale, Some(&mut hasher))?;
                write_f32(&mut writer, vector.f_error, Some(&mut hasher))?;
                write_f32(&mut writer, vector.residual_norm, Some(&mut hasher))?;
                write_f32(&mut writer, vector.f_add_ex, Some(&mut hasher))?;
                write_f32(&mut writer, vector.f_rescale_ex, Some(&mut hasher))?;
            }
        }

        // Write checksum
        let checksum = hasher.finalize();
        write_u32(&mut writer, checksum, None)?;

        writer.flush()?;
        Ok(())
    }

    /// Load the MSTG index from a file
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader)
    }

    /// Load the MSTG index from a reader
    pub fn load_from_reader<R: Read>(reader: R) -> Result<Self, RabitqError> {
        let mut reader = BufReader::new(reader);

        // Check magic and version
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != *b"MSTG" {
            return Err(RabitqError::InvalidPersistence("unrecognized file header"));
        }

        let version = read_u32(&mut reader, None)?;
        if version != 1 {
            return Err(RabitqError::InvalidPersistence("unsupported version"));
        }

        let mut hasher = Hasher::new();

        // Load metadata
        let dim = read_u32(&mut reader, Some(&mut hasher))? as usize;
        let padded_dim = read_u32(&mut reader, Some(&mut hasher))? as usize;

        let metric_tag = read_u8(&mut reader, Some(&mut hasher))?;
        let metric = tag_to_metric(metric_tag)
            .ok_or(RabitqError::InvalidPersistence("unknown metric tag"))?;

        let rotator_type_tag = read_u8(&mut reader, Some(&mut hasher))?;
        let rotator_type = RotatorType::from_u8(rotator_type_tag)
            .ok_or(RabitqError::InvalidPersistence("unknown rotator type"))?;

        let ex_bits = read_u8(&mut reader, Some(&mut hasher))? as usize;

        // Load rotator
        let rotator_len = read_u64(&mut reader, Some(&mut hasher))? as usize;
        let mut rotator_data = vec![0u8; rotator_len];
        reader.read_exact(&mut rotator_data)?;
        hasher.update(&rotator_data);
        let rotator = DynamicRotator::deserialize(dim, padded_dim, rotator_type, &rotator_data)?;

        // Load clusters
        let num_clusters = read_u64(&mut reader, Some(&mut hasher))? as usize;
        let mut clusters = Vec::with_capacity(num_clusters);

        for _ in 0..num_clusters {
            // Load medoid
            let mut medoid = vec![0f32; padded_dim];
            for val in medoid.iter_mut() {
                *val = read_f32(&mut reader, Some(&mut hasher))?;
            }

            // Load ids
            let num_ids = read_u64(&mut reader, Some(&mut hasher))? as usize;
            let mut ids = Vec::with_capacity(num_ids);
            for _ in 0..num_ids {
                ids.push(read_u64(&mut reader, Some(&mut hasher))? as usize);
            }

            // Load vectors
            let mut vectors = Vec::with_capacity(num_ids);
            for _ in 0..num_ids {
                // Load packed codes
                let binary_packed_size = (padded_dim + 7) / 8;
                let mut binary_packed = vec![0u8; binary_packed_size];
                reader.read_exact(&mut binary_packed)?;
                hasher.update(&binary_packed);
                let binary_code = unpack_binary_code(&binary_packed, padded_dim);

                let ex_packed_size = if ex_bits > 0 {
                    ((padded_dim * ex_bits) + 7) / 8
                } else {
                    0
                };
                let mut ex_packed = vec![0u8; ex_packed_size];
                reader.read_exact(&mut ex_packed)?;
                hasher.update(&ex_packed);
                let ex_code = unpack_ex_code(&ex_packed, padded_dim, ex_bits as u8);

                // Reconstruct code
                let code: Vec<u16> = binary_code
                    .iter()
                    .zip(ex_code.iter())
                    .map(|(&bin, &ex)| ex + ((bin as u16) << ex_bits))
                    .collect();

                // Load metadata
                let delta = read_f32(&mut reader, Some(&mut hasher))?;
                let vl = read_f32(&mut reader, Some(&mut hasher))?;
                let f_add = read_f32(&mut reader, Some(&mut hasher))?;
                let f_rescale = read_f32(&mut reader, Some(&mut hasher))?;
                let f_error = read_f32(&mut reader, Some(&mut hasher))?;
                let residual_norm = read_f32(&mut reader, Some(&mut hasher))?;
                let f_add_ex = read_f32(&mut reader, Some(&mut hasher))?;
                let f_rescale_ex = read_f32(&mut reader, Some(&mut hasher))?;

                // Pack codes again for memory
                use crate::simd;
                let mut binary_code_packed2 = vec![0u8; binary_packed_size];
                simd::pack_binary_code(&binary_code, &mut binary_code_packed2, padded_dim);

                let mut ex_code_packed2 = vec![0u8; ex_packed_size];
                if ex_bits > 0 {
                    simd::pack_ex_code(&ex_code, &mut ex_code_packed2, padded_dim, ex_bits as u8);
                }

                vectors.push(QuantizedVector {
                    code,
                    binary_code_packed: binary_code_packed2,
                    ex_code_packed: ex_code_packed2,
                    ex_bits: ex_bits as u8,
                    dim: padded_dim,
                    delta,
                    vl,
                    f_add,
                    f_rescale,
                    f_error,
                    residual_norm,
                    f_add_ex,
                    f_rescale_ex,
                });
            }

            clusters.push(MstgCluster {
                medoid,
                ids,
                vectors,
            });
        }

        // Verify checksum
        let computed_checksum = hasher.finalize();
        let stored_checksum = read_u32(&mut reader, None)?;
        if computed_checksum != stored_checksum {
            return Err(RabitqError::InvalidPersistence("checksum mismatch"));
        }

        // Rebuild HNSW from medoids
        println!("Rebuilding HNSW index from {} medoids...", clusters.len());
        let medoids: Vec<Medoid> = clusters
            .iter()
            .enumerate()
            .map(|(id, cluster)| Medoid {
                cluster_id: id,
                vector: cluster.medoid.clone(),
            })
            .collect();

        // Use high ef_search when rebuilding to ensure good recall
        let hnsw = Self::build_hnsw(&medoids, 16, 200, 1000)?;
        println!("HNSW index rebuilt (ef_search=1000)");

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            clusters,
            hnsw,
            ex_bits,
        })
    }
}

#[derive(Debug, Clone)]
struct HeapEntry {
    id: usize,
    distance: f32,
    score: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits().eq(&other.distance.to_bits()) && self.id == other.id
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
        self.distance.total_cmp(&other.distance)
    }
}
