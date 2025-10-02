use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher;

use rand::prelude::*;

use crate::kmeans::{run_kmeans, KMeansResult};
use crate::math::{dot, l2_distance_sqr};
use crate::quantizer::{quantize_with_centroid, QuantizedVector};
use crate::rotation::{DynamicRotator, RotatorType};
use crate::{Metric, RabitqError};

/// Parameters for IVF search.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub top_k: usize,
    pub nprobe: usize,
}

fn write_u16<W: Write>(writer: &mut W, value: u16, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn write_u32<W: Write>(writer: &mut W, value: u32, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn write_u64<W: Write>(writer: &mut W, value: u64, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn write_f32<W: Write>(writer: &mut W, value: f32, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = value.to_le_bytes();
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    writer.write_all(&bytes)
}

fn read_u8<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(buf[0])
}

fn read_u16<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    if let Some(h) = hasher {
        h.update(&buf);
    }
    Ok(f32::from_le_bytes(buf))
}

fn usize_from_u64(value: u64) -> Result<usize, RabitqError> {
    usize::try_from(value)
        .map_err(|_| RabitqError::InvalidPersistence("value exceeds platform limits"))
}

fn metric_to_tag(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::InnerProduct => 1,
    }
}

fn tag_to_metric(tag: u8) -> Option<Metric> {
    match tag {
        0 => Some(Metric::L2),
        1 => Some(Metric::InnerProduct),
        _ => None,
    }
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

const PERSIST_MAGIC: [u8; 4] = *b"RBQ1";
const PERSIST_VERSION: u32 = 1;

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
    padded_dim: usize,
    metric: Metric,
    rotator: DynamicRotator,
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
        rotator_type: RotatorType,
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

        let rotator = DynamicRotator::new(dim, rotator_type, seed);
        let padded_dim = rotator.padded_dim();
        let rotated_data: Vec<Vec<f32>> = data.iter().map(|v| rotator.rotate(v)).collect();

        let mut rng = StdRng::seed_from_u64(seed ^ 0x5a5a_5a5a5a5a5a5a);
        let KMeansResult {
            centroids,
            assignments,
        } = run_kmeans(&rotated_data, nlist, 30, &mut rng);

        Self::build_from_rotated(
            dim,
            padded_dim,
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
        rotator_type: RotatorType,
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

        let rotator = DynamicRotator::new(dim, rotator_type, seed);
        let padded_dim = rotator.padded_dim();
        let rotated_data: Vec<Vec<f32>> = data.iter().map(|v| rotator.rotate(v)).collect();
        let rotated_centroids: Vec<Vec<f32>> =
            centroids.iter().map(|c| rotator.rotate(c)).collect();

        Self::build_from_rotated(
            dim,
            padded_dim,
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
        padded_dim: usize,
        metric: Metric,
        rotator: DynamicRotator,
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
            padded_dim,
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

    /// Number of IVF clusters maintained by the index.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Persist the index to the provided filesystem path.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), RabitqError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.save_to_writer(writer)
    }

    /// Persist the index using the supplied writer.
    pub fn save_to_writer<W: Write>(&self, writer: W) -> Result<(), RabitqError> {
        let mut writer = BufWriter::new(writer);
        writer.write_all(&PERSIST_MAGIC)?;
        write_u32(&mut writer, PERSIST_VERSION, None)?;

        let mut hasher = Hasher::new();

        let dim = u32::try_from(self.dim)
            .map_err(|_| RabitqError::InvalidPersistence("dimension exceeds persistence limits"))?;
        write_u32(&mut writer, dim, Some(&mut hasher))?;

        let padded_dim = u32::try_from(self.padded_dim)
            .map_err(|_| RabitqError::InvalidPersistence("padded_dim exceeds persistence limits"))?;
        write_u32(&mut writer, padded_dim, Some(&mut hasher))?;

        let metric_tag = metric_to_tag(self.metric);
        writer.write_all(&[metric_tag])?;
        hasher.update(&[metric_tag]);

        let rotator_type_tag = self.rotator.rotator_type() as u8;
        writer.write_all(&[rotator_type_tag])?;
        hasher.update(&[rotator_type_tag]);

        let ex_bits = u8::try_from(self.ex_bits)
            .map_err(|_| RabitqError::InvalidPersistence("ex_bits exceeds persistence limits"))?;
        writer.write_all(&[ex_bits])?;
        hasher.update(&[ex_bits]);

        let total_bits = self
            .ex_bits
            .checked_add(1)
            .ok_or(RabitqError::InvalidPersistence("total_bits overflow"))?;
        let total_bits_u8 = u8::try_from(total_bits).map_err(|_| {
            RabitqError::InvalidPersistence("total_bits exceeds persistence limits")
        })?;
        writer.write_all(&[total_bits_u8])?;
        hasher.update(&[total_bits_u8]);

        let vector_count = u64::try_from(self.len()).map_err(|_| {
            RabitqError::InvalidPersistence("vector count exceeds persistence limits")
        })?;
        write_u64(&mut writer, vector_count, Some(&mut hasher))?;

        let cluster_count = u64::try_from(self.clusters.len()).map_err(|_| {
            RabitqError::InvalidPersistence("cluster count exceeds persistence limits")
        })?;
        write_u64(&mut writer, cluster_count, Some(&mut hasher))?;

        // Save rotator state (much smaller than full matrix for FHT)
        let rotator_data = self.rotator.serialize();
        let rotator_len = u64::try_from(rotator_data.len())
            .map_err(|_| RabitqError::InvalidPersistence("rotator data too large"))?;
        write_u64(&mut writer, rotator_len, Some(&mut hasher))?;
        writer.write_all(&rotator_data)?;
        hasher.update(&rotator_data);

        for cluster in &self.clusters {
            if cluster.centroid.len() != self.padded_dim {
                return Err(RabitqError::InvalidPersistence(
                    "cluster centroid dimension mismatch",
                ));
            }
            if cluster.ids.len() != cluster.vectors.len() {
                return Err(RabitqError::InvalidPersistence(
                    "cluster id/vector count mismatch",
                ));
            }

            for &value in &cluster.centroid {
                write_f32(&mut writer, value, Some(&mut hasher))?;
            }

            let entry_count = u64::try_from(cluster.ids.len()).map_err(|_| {
                RabitqError::InvalidPersistence("cluster entry count exceeds persistence limits")
            })?;
            write_u64(&mut writer, entry_count, Some(&mut hasher))?;

            for &id in &cluster.ids {
                let encoded = u64::try_from(id).map_err(|_| {
                    RabitqError::InvalidPersistence("vector id exceeds persistence limits")
                })?;
                write_u64(&mut writer, encoded, Some(&mut hasher))?;
            }

            for vector in &cluster.vectors {
                if vector.code.len() != self.padded_dim
                    || vector.binary_code.len() != self.padded_dim
                    || vector.ex_code.len() != self.padded_dim
                {
                    return Err(RabitqError::InvalidPersistence(
                        "quantized vector dimension mismatch",
                    ));
                }

                for &value in &vector.code {
                    write_u16(&mut writer, value, Some(&mut hasher))?;
                }

                writer.write_all(&vector.binary_code)?;
                hasher.update(&vector.binary_code);

                for &value in &vector.ex_code {
                    write_u16(&mut writer, value, Some(&mut hasher))?;
                }

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

        let checksum = hasher.finalize();
        write_u32(&mut writer, checksum, None)?;

        writer.flush()?;
        Ok(())
    }

    /// Load an index from the provided filesystem path.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader)
    }

    /// Load an index from a persisted byte stream.
    pub fn load_from_reader<R: Read>(reader: R) -> Result<Self, RabitqError> {
        let mut reader = BufReader::new(reader);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != PERSIST_MAGIC {
            return Err(RabitqError::InvalidPersistence("unrecognized file header"));
        }

        let version = read_u32(&mut reader, None)?;
        if version != PERSIST_VERSION {
            return Err(RabitqError::InvalidPersistence(
                "unsupported index format version",
            ));
        }

        let mut hasher = Hasher::new();

        let dim = read_u32(&mut reader, Some(&mut hasher))? as usize;
        if dim == 0 {
            return Err(RabitqError::InvalidPersistence(
                "dimension must be positive",
            ));
        }

        let padded_dim = read_u32(&mut reader, Some(&mut hasher))? as usize;
        if padded_dim < dim {
            return Err(RabitqError::InvalidPersistence(
                "padded_dim must be >= dim",
            ));
        }

        let metric_tag = read_u8(&mut reader, Some(&mut hasher))?;
        let metric = tag_to_metric(metric_tag)
            .ok_or(RabitqError::InvalidPersistence("unknown metric tag"))?;

        let rotator_type_tag = read_u8(&mut reader, Some(&mut hasher))?;
        let rotator_type = RotatorType::from_u8(rotator_type_tag)
            .ok_or(RabitqError::InvalidPersistence("unknown rotator type tag"))?;

        let ex_bits = read_u8(&mut reader, Some(&mut hasher))? as usize;
        if ex_bits > 16 {
            return Err(RabitqError::InvalidPersistence("ex_bits out of range"));
        }

        let total_bits = read_u8(&mut reader, Some(&mut hasher))? as usize;
        if total_bits == 0 || total_bits > 16 {
            return Err(RabitqError::InvalidPersistence("total_bits out of range"));
        }
        if total_bits.saturating_sub(1) != ex_bits {
            return Err(RabitqError::InvalidPersistence(
                "total_bits does not match ex_bits",
            ));
        }

        let expected_vectors = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;

        let cluster_count = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
        let rotator_data_len = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;

        let mut rotator_data = vec![0u8; rotator_data_len];
        reader.read_exact(&mut rotator_data)?;
        hasher.update(&rotator_data);

        let rotator = DynamicRotator::deserialize(dim, padded_dim, rotator_type, &rotator_data)?;

        let mut clusters = Vec::with_capacity(cluster_count);
        for _ in 0..cluster_count {
            let mut centroid = vec![0f32; padded_dim];
            for value in centroid.iter_mut() {
                *value = read_f32(&mut reader, Some(&mut hasher))?;
            }

            let entry_count = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;

            let mut ids = Vec::with_capacity(entry_count);
            for _ in 0..entry_count {
                ids.push(usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?);
            }

            let mut vectors = Vec::with_capacity(entry_count);
            for _ in 0..entry_count {
                let mut code = vec![0u16; padded_dim];
                for value in code.iter_mut() {
                    *value = read_u16(&mut reader, Some(&mut hasher))?;
                }

                let mut binary_code = vec![0u8; padded_dim];
                reader.read_exact(&mut binary_code)?;
                hasher.update(&binary_code);

                let mut ex_code = vec![0u16; padded_dim];
                for value in ex_code.iter_mut() {
                    *value = read_u16(&mut reader, Some(&mut hasher))?;
                }

                let delta = read_f32(&mut reader, Some(&mut hasher))?;
                let vl = read_f32(&mut reader, Some(&mut hasher))?;
                let f_add = read_f32(&mut reader, Some(&mut hasher))?;
                let f_rescale = read_f32(&mut reader, Some(&mut hasher))?;
                let f_error = read_f32(&mut reader, Some(&mut hasher))?;
                let residual_norm = read_f32(&mut reader, Some(&mut hasher))?;
                let f_add_ex = read_f32(&mut reader, Some(&mut hasher))?;
                let f_rescale_ex = read_f32(&mut reader, Some(&mut hasher))?;

                vectors.push(QuantizedVector {
                    code,
                    binary_code,
                    ex_code,
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

            if ids.len() != vectors.len() {
                return Err(RabitqError::InvalidPersistence(
                    "cluster id/vector count mismatch",
                ));
            }

            clusters.push(ClusterEntry {
                centroid,
                ids,
                vectors,
            });
        }

        let actual_vectors: usize = clusters.iter().map(|c| c.ids.len()).sum();
        if actual_vectors != expected_vectors {
            return Err(RabitqError::InvalidPersistence(
                "vector count metadata mismatch",
            ));
        }

        let computed_checksum = hasher.finalize();
        let stored_checksum = read_u32(&mut reader, None)?;
        if computed_checksum != stored_checksum {
            return Err(RabitqError::InvalidPersistence("checksum mismatch"));
        }

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            clusters,
            ex_bits,
        })
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
