use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::ivf::{unpack_binary_code, unpack_ex_code};
use crate::quantizer::{quantize_with_centroid, QuantizedVector, RabitqConfig};
use crate::rotation::{DynamicRotator, RotatorType};
use crate::{Metric, RabitqError};

const PERSIST_MAGIC: [u8; 4] = *b"RBF1";
const PERSIST_VERSION: u32 = 1;

/// Parameters for brute-force search.
#[derive(Debug, Clone, Copy)]
pub struct BruteForceSearchParams {
    pub top_k: usize,
}

impl BruteForceSearchParams {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }
}

/// Result entry returned by brute-force search.
#[derive(Debug, Clone, PartialEq)]
pub struct BruteForceSearchResult {
    pub id: usize,
    pub score: f32,
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

/// Precomputed query constants to avoid repeated calculations during search.
struct QueryPrecomputed {
    rotated_query: Vec<f32>,
    k1x_sum_q: f32, // c1 × sum_query (precomputed)
    kbx_sum_q: f32, // cb × sum_query (precomputed)
    binary_scale: f32,
}

impl QueryPrecomputed {
    fn new(rotated_query: Vec<f32>, ex_bits: usize) -> Self {
        let sum_query: f32 = rotated_query.iter().sum();
        let c1 = -0.5f32;
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let binary_scale = (1 << ex_bits) as f32;

        Self {
            rotated_query,
            k1x_sum_q: c1 * sum_query,
            kbx_sum_q: cb * sum_query,
            binary_scale,
        }
    }
}

fn write_u8<W: Write>(writer: &mut W, value: u8, hasher: Option<&mut Hasher>) -> io::Result<()> {
    let bytes = [value];
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

/// Reconstruct full code from binary_code and ex_code
fn reconstruct_code(binary_code: &[u8], ex_code: &[u16], ex_bits: u8) -> Vec<u16> {
    binary_code
        .iter()
        .zip(ex_code.iter())
        .map(|(&bin, &ex)| ex + ((bin as u16) << ex_bits))
        .collect()
}

/// Brute-force RaBitQ index without clustering.
///
/// This index quantizes all vectors relative to a zero centroid and performs
/// exhaustive search over all vectors. It's suitable for small to medium datasets
/// (e.g., <200K vectors with 1024 dimensions) where IVF clustering overhead is
/// not justified.
#[derive(Debug, Clone)]
pub struct BruteForceRabitqIndex {
    dim: usize,
    padded_dim: usize,
    metric: Metric,
    rotator: DynamicRotator,
    vectors: Vec<QuantizedVector>,
    ex_bits: usize,
}

impl BruteForceRabitqIndex {
    /// Train a new brute-force index from the provided dataset.
    pub fn train(
        data: &[Vec<f32>],
        total_bits: usize,
        metric: Metric,
        rotator_type: RotatorType,
        seed: u64,
        use_faster_config: bool,
    ) -> Result<Self, RabitqError> {
        if data.is_empty() {
            return Err(RabitqError::InvalidConfig(
                "training data must be non-empty",
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

        let rotator = DynamicRotator::new(dim, rotator_type, seed);
        let padded_dim = rotator.padded_dim();

        println!("Rotating vectors...");
        let rotated_data: Vec<Vec<f32>> = data.par_iter().map(|v| rotator.rotate(v)).collect();

        let config = if use_faster_config {
            RabitqConfig::faster(padded_dim, total_bits, seed)
        } else {
            RabitqConfig::new(total_bits)
        };

        // Use zero centroid for brute-force quantization
        let zero_centroid = vec![0.0f32; padded_dim];

        println!("Quantizing vectors...");
        use std::sync::atomic::{AtomicUsize, Ordering};
        let progress_counter = AtomicUsize::new(0);
        let total_vectors = rotated_data.len();

        let vectors: Vec<QuantizedVector> = rotated_data
            .par_iter()
            .map(|vec| {
                let quantized = quantize_with_centroid(vec, &zero_centroid, &config, metric);

                let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % (total_vectors / 20).max(1) == 0 || completed == total_vectors {
                    println!(
                        "  Quantized {}/{} vectors ({:.1}%)",
                        completed,
                        total_vectors,
                        100.0 * completed as f64 / total_vectors as f64
                    );
                }

                quantized
            })
            .collect();
        println!("Quantization complete");

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            vectors,
            ex_bits: total_bits.saturating_sub(1),
        })
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
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

        let padded_dim = u32::try_from(self.padded_dim).map_err(|_| {
            RabitqError::InvalidPersistence("padded_dim exceeds persistence limits")
        })?;
        write_u32(&mut writer, padded_dim, Some(&mut hasher))?;

        let metric_tag = metric_to_tag(self.metric);
        write_u8(&mut writer, metric_tag, Some(&mut hasher))?;

        let rotator_type_tag = self.rotator.rotator_type() as u8;
        write_u8(&mut writer, rotator_type_tag, Some(&mut hasher))?;

        let ex_bits = u8::try_from(self.ex_bits)
            .map_err(|_| RabitqError::InvalidPersistence("ex_bits exceeds persistence limits"))?;
        write_u8(&mut writer, ex_bits, Some(&mut hasher))?;

        let total_bits = self
            .ex_bits
            .checked_add(1)
            .ok_or(RabitqError::InvalidPersistence("total_bits overflow"))?;
        let total_bits_u8 = u8::try_from(total_bits).map_err(|_| {
            RabitqError::InvalidPersistence("total_bits exceeds persistence limits")
        })?;
        write_u8(&mut writer, total_bits_u8, Some(&mut hasher))?;

        let vector_count = u64::try_from(self.len()).map_err(|_| {
            RabitqError::InvalidPersistence("vector count exceeds persistence limits")
        })?;
        write_u64(&mut writer, vector_count, Some(&mut hasher))?;

        // Save rotator state
        let rotator_data = self.rotator.serialize();
        let rotator_len = u64::try_from(rotator_data.len())
            .map_err(|_| RabitqError::InvalidPersistence("rotator data too large"))?;
        write_u64(&mut writer, rotator_len, Some(&mut hasher))?;
        writer.write_all(&rotator_data)?;
        hasher.update(&rotator_data);

        // Save vectors
        for vector in &self.vectors {
            if vector.dim != self.padded_dim {
                return Err(RabitqError::InvalidPersistence(
                    "quantized vector dimension mismatch",
                ));
            }

            // Write packed binary_code (already in packed format)
            writer.write_all(&vector.binary_code_packed)?;
            hasher.update(&vector.binary_code_packed);

            // Write packed ex_code (already in packed format)
            writer.write_all(&vector.ex_code_packed)?;
            hasher.update(&vector.ex_code_packed);

            // Metadata (8 × f32 = 32 bytes)
            write_f32(&mut writer, vector.delta, Some(&mut hasher))?;
            write_f32(&mut writer, vector.vl, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_add, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_rescale, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_error, Some(&mut hasher))?;
            write_f32(&mut writer, vector.residual_norm, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_add_ex, Some(&mut hasher))?;
            write_f32(&mut writer, vector.f_rescale_ex, Some(&mut hasher))?;
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
        if version != 1 {
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
            return Err(RabitqError::InvalidPersistence("padded_dim must be >= dim"));
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

        let vector_count = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;

        let rotator_data_len = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
        let mut rotator_data = vec![0u8; rotator_data_len];
        reader.read_exact(&mut rotator_data)?;
        hasher.update(&rotator_data);

        let rotator = DynamicRotator::deserialize(dim, padded_dim, rotator_type, &rotator_data)?;

        let mut vectors = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
            // Bit-packed format
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

            // Reconstruct full code
            let code = reconstruct_code(&binary_code, &ex_code, ex_bits as u8);

            let delta = read_f32(&mut reader, Some(&mut hasher))?;
            let vl = read_f32(&mut reader, Some(&mut hasher))?;
            let f_add = read_f32(&mut reader, Some(&mut hasher))?;
            let f_rescale = read_f32(&mut reader, Some(&mut hasher))?;
            let f_error = read_f32(&mut reader, Some(&mut hasher))?;
            let residual_norm = read_f32(&mut reader, Some(&mut hasher))?;
            let f_add_ex = read_f32(&mut reader, Some(&mut hasher))?;
            let f_rescale_ex = read_f32(&mut reader, Some(&mut hasher))?;

            // Pack the codes for in-memory storage
            use crate::simd;
            let binary_code_packed_size = (padded_dim + 7) / 8;
            let mut binary_code_packed = vec![0u8; binary_code_packed_size];
            simd::pack_binary_code(&binary_code, &mut binary_code_packed, padded_dim);

            let ex_code_packed_size = if ex_bits > 0 {
                ((padded_dim * ex_bits) + 7) / 8
            } else {
                0
            };
            let mut ex_code_packed = vec![0u8; ex_code_packed_size];
            if ex_bits > 0 {
                simd::pack_ex_code(&ex_code, &mut ex_code_packed, padded_dim, ex_bits as u8);
            }

            vectors.push(QuantizedVector {
                code,
                binary_code_packed,
                ex_code_packed,
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
            vectors,
            ex_bits,
        })
    }

    /// Search for the nearest neighbours of the provided query vector.
    pub fn search(
        &self,
        query: &[f32],
        params: BruteForceSearchParams,
    ) -> Result<Vec<BruteForceSearchResult>, RabitqError> {
        self.search_internal(query, params, None)
    }

    /// Search for the nearest neighbours of the provided query vector,
    /// filtering results to only include vector IDs present in the provided bitmap.
    pub fn search_filtered(
        &self,
        query: &[f32],
        params: BruteForceSearchParams,
        filter: &RoaringBitmap,
    ) -> Result<Vec<BruteForceSearchResult>, RabitqError> {
        self.search_internal(query, params, Some(filter))
    }

    fn search_internal(
        &self,
        query: &[f32],
        params: BruteForceSearchParams,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<BruteForceSearchResult>, RabitqError> {
        if self.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        if params.top_k == 0 {
            return Ok(Vec::new());
        }

        // Precompute query constants once
        let rotated_query = self.rotator.rotate(query);
        let query_precomp = QueryPrecomputed::new(rotated_query, self.ex_bits);

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        // Zero centroid for brute-force (g_add = 0)
        let g_add = 0.0f32;

        for (vector_id, quantized) in self.vectors.iter().enumerate() {
            // Skip if not in filter
            if let Some(filter_bitmap) = filter {
                if !filter_bitmap.contains(vector_id as u32) {
                    continue;
                }
            }

            // Unpack binary code for computation
            let binary_code = quantized.unpack_binary_code();
            let mut binary_dot = 0.0f32;
            for (&bit, &q_val) in binary_code.iter().zip(query_precomp.rotated_query.iter()) {
                binary_dot += (bit as f32) * q_val;
            }
            let binary_term = binary_dot + query_precomp.k1x_sum_q;
            let mut distance = quantized.f_add + g_add + quantized.f_rescale * binary_term;

            let mut score = match self.metric {
                Metric::L2 => distance,
                Metric::InnerProduct => -distance,
            };

            if self.ex_bits > 0 {
                // Unpack ex code for computation
                let ex_code = quantized.unpack_ex_code();
                let mut ex_dot = 0.0f32;
                for (&code, &q_val) in ex_code.iter().zip(query_precomp.rotated_query.iter()) {
                    ex_dot += (code as f32) * q_val;
                }
                let total_term =
                    query_precomp.binary_scale * binary_dot + ex_dot + query_precomp.kbx_sum_q;
                distance = quantized.f_add_ex + g_add + quantized.f_rescale_ex * total_term;
                score = match self.metric {
                    Metric::L2 => distance,
                    Metric::InnerProduct => -distance,
                };
            }

            if !distance.is_finite() {
                continue;
            }

            heap.push(HeapEntry {
                candidate: HeapCandidate {
                    id: vector_id,
                    distance,
                    score,
                },
            });

            if heap.len() > params.top_k {
                heap.pop();
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
            .map(|candidate| BruteForceSearchResult {
                id: candidate.id,
                score: match self.metric {
                    Metric::L2 => candidate.distance,
                    Metric::InnerProduct => candidate.score,
                },
            })
            .collect())
    }
}
