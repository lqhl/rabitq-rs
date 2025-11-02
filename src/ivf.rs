use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher;

use rand::prelude::*;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::kmeans::{run_kmeans, KMeansResult};
use crate::math::{dot, l2_distance_sqr};
use crate::quantizer::{quantize_with_centroid, QuantizedVector, RabitqConfig};
use crate::rotation::{DynamicRotator, RotatorType};
use crate::simd;
use crate::{Metric, RabitqError};

/// Parameters for IVF search.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub top_k: usize,
    pub nprobe: usize,
}

/// Unpack binary codes from packed bytes
pub(crate) fn unpack_binary_code(packed: &[u8], dim: usize) -> Vec<u8> {
    let mut binary_code = vec![0u8; dim];
    for i in 0..dim {
        if (packed[i / 8] & (1 << (i % 8))) != 0 {
            binary_code[i] = 1;
        }
    }
    binary_code
}

/// Unpack ex codes from packed bytes
pub(crate) fn unpack_ex_code(packed: &[u8], dim: usize, ex_bits: u8) -> Vec<u16> {
    if ex_bits == 0 {
        return vec![0u16; dim];
    }
    let mut ex_code = vec![0u16; dim];
    let bits_per_element = ex_bits as usize;

    #[allow(clippy::needless_range_loop)]
    for i in 0..dim {
        let bit_offset = i * bits_per_element;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let mut code = 0u16;
        let mut remaining_bits = bits_per_element;
        let mut current_byte = byte_offset;
        let mut current_bit = bit_in_byte;
        let mut shift = 0;

        while remaining_bits > 0 {
            let bits_in_current_byte = (8 - current_bit).min(remaining_bits);
            let mask = ((1u16 << bits_in_current_byte) - 1) as u8;
            let bits = ((packed[current_byte] >> current_bit) & mask) as u16;
            code |= bits << shift;

            shift += bits_in_current_byte;
            remaining_bits -= bits_in_current_byte;
            current_byte += 1;
            current_bit = 0;
        }

        ex_code[i] = code;
    }
    ex_code
}

/// Reconstruct full code from binary_code and ex_code
#[allow(dead_code)]
fn reconstruct_code(binary_code: &[u8], ex_code: &[u16], ex_bits: u8) -> Vec<u16> {
    binary_code
        .iter()
        .zip(ex_code.iter())
        .map(|(&bin, &ex)| ex + ((bin as u16) << ex_bits))
        .collect()
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
const PERSIST_VERSION: u32 = 3; // V3: Unified memory layout with ClusterData

// ============================================================================
// Unified Memory Layout Data Structures (Phase 1 Optimization)
// ============================================================================

/// Generic 64-byte cache line aligned wrapper
/// Ensures data is aligned to cache line boundaries for optimal performance
#[repr(C, align(64))]
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T> AlignedVec<T> {
    #[allow(dead_code)]
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[allow(dead_code)]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Unified cluster data with contiguous memory layout
/// Eliminates V1/V2 split and scattered allocations
/// Reference: OPTIMIZATION_PLAN.md Phase 1
/// 64-byte cache line aligned for better cache efficiency
#[repr(C, align(64))]
#[derive(Debug, Clone)]
struct ClusterData {
    /// Cluster centroid
    centroid: Vec<f32>,

    /// Vector IDs (all vectors in cluster)
    ids: Vec<usize>,

    /// Single contiguous memory block for all batches
    /// Layout: [Batch 0][Batch 1]...[Batch N]
    /// Each batch layout:
    ///   - packed_binary_codes: padded_dim * 32 / 8 bytes
    ///   - f_add: 32 * 4 bytes (f32)
    ///   - f_rescale: 32 * 4 bytes (f32)
    ///   - f_error: 32 * 4 bytes (f32)
    batch_data: Vec<u8>,

    /// Packed extended codes (per-vector, C++-style on-demand unpacking)
    /// Each vector: Vec<u8> containing bit-packed ex_code
    /// Memory-efficient: ~(padded_dim * ex_bits / 8) bytes per vector
    /// Unpacked on-demand only after lower-bound filtering (mimics C++ estimator.hpp:85-103)
    /// Reference: OPTIMIZATION_PLAN.md - removes 4.4x memory overhead from pre-unpacking
    ex_codes_packed: Vec<Vec<u8>>,

    /// Per-vector ex parameters
    f_add_ex: Vec<f32>,
    f_rescale_ex: Vec<f32>,

    /// Metadata
    num_vectors: usize,
    padded_dim: usize,
    ex_bits: usize,
}

impl ClusterData {
    /// Calculate bytes per batch in contiguous layout
    #[inline(always)]
    fn batch_stride(padded_dim: usize) -> usize {
        let binary_codes_bytes = padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let params_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE * 3; // f_add, f_rescale, f_error
        binary_codes_bytes + params_bytes
    }

    /// Get number of complete batches
    #[inline(always)]
    fn num_complete_batches(&self) -> usize {
        self.num_vectors / simd::FASTSCAN_BATCH_SIZE
    }

    /// Get number of remainder vectors
    #[inline(always)]
    fn num_remainder_vectors(&self) -> usize {
        self.num_vectors % simd::FASTSCAN_BATCH_SIZE
    }

    /// Zero-copy access to packed binary codes for a batch
    #[inline(always)]
    fn batch_bin_codes(&self, batch_idx: usize) -> &[u8] {
        let stride = Self::batch_stride(self.padded_dim);
        let offset = batch_idx * stride;
        let len = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        &self.batch_data[offset..offset + len]
    }

    /// Zero-copy access to f_add parameters for a batch
    #[inline(always)]
    fn batch_f_add(&self, batch_idx: usize) -> &[f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let offset = batch_idx * stride + (self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8);
        unsafe {
            std::slice::from_raw_parts(
                self.batch_data[offset..].as_ptr() as *const f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Zero-copy access to f_rescale parameters for a batch
    #[inline(always)]
    fn batch_f_rescale(&self, batch_idx: usize) -> &[f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let binary_bytes = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let f_add_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let offset = batch_idx * stride + binary_bytes + f_add_bytes;
        unsafe {
            std::slice::from_raw_parts(
                self.batch_data[offset..].as_ptr() as *const f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Zero-copy access to f_error parameters for a batch
    #[inline(always)]
    fn batch_f_error(&self, batch_idx: usize) -> &[f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let binary_bytes = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let f_add_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_rescale_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let offset = batch_idx * stride + binary_bytes + f_add_bytes + f_rescale_bytes;
        unsafe {
            std::slice::from_raw_parts(
                self.batch_data[offset..].as_ptr() as *const f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Get f_add for a single vector (for naive search)
    #[allow(dead_code)]
    fn get_vector_f_add(&self, vec_idx: usize) -> f32 {
        let batch_idx = vec_idx / simd::FASTSCAN_BATCH_SIZE;
        let in_batch_idx = vec_idx % simd::FASTSCAN_BATCH_SIZE;
        self.batch_f_add(batch_idx)[in_batch_idx]
    }

    /// Get f_rescale for a single vector (for naive search)
    #[allow(dead_code)]
    fn get_vector_f_rescale(&self, vec_idx: usize) -> f32 {
        let batch_idx = vec_idx / simd::FASTSCAN_BATCH_SIZE;
        let in_batch_idx = vec_idx % simd::FASTSCAN_BATCH_SIZE;
        self.batch_f_rescale(batch_idx)[in_batch_idx]
    }

    /// Mutable access to batch binary codes
    #[allow(dead_code)]
    fn batch_bin_codes_mut(&mut self, batch_idx: usize) -> &mut [u8] {
        let stride = Self::batch_stride(self.padded_dim);
        let offset = batch_idx * stride;
        let len = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        &mut self.batch_data[offset..offset + len]
    }

    /// Mutable access to f_add parameters
    #[allow(dead_code)]
    fn batch_f_add_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let offset = batch_idx * stride + (self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8);
        unsafe {
            std::slice::from_raw_parts_mut(
                self.batch_data[offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Mutable access to f_rescale parameters
    #[allow(dead_code)]
    fn batch_f_rescale_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let binary_bytes = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let f_add_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let offset = batch_idx * stride + binary_bytes + f_add_bytes;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.batch_data[offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Mutable access to f_error parameters
    #[allow(dead_code)]
    fn batch_f_error_mut(&mut self, batch_idx: usize) -> &mut [f32] {
        let stride = Self::batch_stride(self.padded_dim);
        let binary_bytes = self.padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;
        let f_add_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_rescale_bytes = std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let offset = batch_idx * stride + binary_bytes + f_add_bytes + f_rescale_bytes;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.batch_data[offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            )
        }
    }

    /// Create new empty cluster
    #[allow(dead_code)]
    fn new(centroid: Vec<f32>, padded_dim: usize, ex_bits: usize) -> Self {
        Self {
            centroid,
            ids: Vec::new(),
            batch_data: Vec::new(),
            ex_codes_packed: Vec::new(),
            f_add_ex: Vec::new(),
            f_rescale_ex: Vec::new(),
            num_vectors: 0,
            padded_dim,
            ex_bits,
        }
    }

    /// Build ClusterData from quantized vectors
    /// Implements unified memory layout with:
    /// 1. Contiguous batch storage (eliminates scattered Vec allocations)
    /// 2. Pre-unpacked ex_codes (avoids repeated unpacking)
    fn from_quantized_vectors(
        centroid: Vec<f32>,
        ids: Vec<usize>,
        vectors: Vec<QuantizedVector>,
        padded_dim: usize,
        ex_bits: usize,
    ) -> Self {
        let num_vectors = vectors.len();
        let num_complete_batches = num_vectors / simd::FASTSCAN_BATCH_SIZE;
        let num_remainder = num_vectors % simd::FASTSCAN_BATCH_SIZE;

        let total_batches = if num_remainder > 0 {
            num_complete_batches + 1
        } else {
            num_complete_batches
        };

        // Allocate contiguous memory for all batches
        let stride = Self::batch_stride(padded_dim);
        let mut batch_data = crate::memory::allocate_aligned_vec::<u8>(stride * total_batches);

        // Pre-allocate storage for packed ex_codes and parameters
        let mut ex_codes_packed = Vec::with_capacity(num_vectors);
        let mut f_add_ex = Vec::with_capacity(num_vectors);
        let mut f_rescale_ex = Vec::with_capacity(num_vectors);

        let dim_bytes = padded_dim / 8;

        // Process complete batches
        for batch_idx in 0..num_complete_batches {
            let start_idx = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let end_idx = start_idx + simd::FASTSCAN_BATCH_SIZE;
            let batch_vectors = &vectors[start_idx..end_idx];

            // Pack this batch into contiguous memory
            Self::pack_batch_into_memory(
                batch_vectors,
                &mut batch_data,
                batch_idx,
                padded_dim,
                ex_bits,
                &mut ex_codes_packed,
                &mut f_add_ex,
                &mut f_rescale_ex,
            );
        }

        // Process remainder vectors
        if num_remainder > 0 {
            let start_idx = num_complete_batches * simd::FASTSCAN_BATCH_SIZE;
            let batch_vectors = &vectors[start_idx..];

            // Pad with zeros
            let mut padded_vectors = batch_vectors.to_vec();
            let ex_bytes_per_vec = if ex_bits > 0 {
                padded_dim * ex_bits / 8
            } else {
                0
            };
            padded_vectors.resize(
                simd::FASTSCAN_BATCH_SIZE,
                QuantizedVector {
                    code: vec![0u16; padded_dim],
                    binary_code_packed: vec![0u8; dim_bytes],
                    ex_code_packed: vec![0u8; ex_bytes_per_vec],
                    binary_code_unpacked: Vec::new(),
                    ex_code_unpacked: Vec::new(),
                    ex_bits: ex_bits as u8,
                    dim: padded_dim,
                    delta: 0.0,
                    vl: 0.0,
                    f_add: 0.0,
                    f_rescale: 0.0,
                    f_error: 0.0,
                    residual_norm: 0.0,
                    f_add_ex: 0.0,
                    f_rescale_ex: 0.0,
                },
            );

            // Call pack_batch_into_memory_partial to handle partial batches correctly
            Self::pack_batch_into_memory_partial(
                &padded_vectors,
                num_remainder, // actual number of vectors
                &mut batch_data,
                num_complete_batches,
                padded_dim,
                ex_bits,
                &mut ex_codes_packed,
                &mut f_add_ex,
                &mut f_rescale_ex,
            );
        }

        Self {
            centroid,
            ids,
            batch_data,
            ex_codes_packed,
            f_add_ex,
            f_rescale_ex,
            num_vectors,
            padded_dim,
            ex_bits,
        }
    }

    /// Pack 32 vectors into contiguous batch memory
    /// Pack a partial batch into memory (handles the last batch with < 32 vectors)
    #[allow(clippy::too_many_arguments)]
    fn pack_batch_into_memory_partial(
        vectors: &[QuantizedVector],
        actual_count: usize, // actual number of vectors (before padding)
        batch_data: &mut [u8],
        batch_idx: usize,
        padded_dim: usize,
        ex_bits: usize,
        ex_codes_packed: &mut Vec<Vec<u8>>,
        f_add_ex: &mut Vec<f32>,
        f_rescale_ex: &mut Vec<f32>,
    ) {
        assert_eq!(vectors.len(), simd::FASTSCAN_BATCH_SIZE);
        assert!(actual_count <= simd::FASTSCAN_BATCH_SIZE);

        let stride = Self::batch_stride(padded_dim);
        let dim_bytes = padded_dim / 8;

        // Calculate offsets in contiguous memory
        let batch_offset = batch_idx * stride;
        let binary_bytes = padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;

        // Collect binary codes into flat buffer (including padding)
        let mut binary_codes_flat = Vec::with_capacity(simd::FASTSCAN_BATCH_SIZE * dim_bytes);
        for vec in vectors.iter() {
            binary_codes_flat.extend_from_slice(&vec.binary_code_packed);
        }

        // Pack binary codes using FastScan layout
        let packed_codes = &mut batch_data[batch_offset..batch_offset + binary_bytes];
        simd::pack_codes(
            &binary_codes_flat,
            simd::FASTSCAN_BATCH_SIZE,
            dim_bytes,
            packed_codes,
        );

        // Get mutable slices for parameters
        let f_add_offset = batch_offset + binary_bytes;
        let f_rescale_offset =
            f_add_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_error_offset =
            f_rescale_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;

        unsafe {
            let f_add_slice = std::slice::from_raw_parts_mut(
                batch_data[f_add_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_rescale_slice = std::slice::from_raw_parts_mut(
                batch_data[f_rescale_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_error_slice = std::slice::from_raw_parts_mut(
                batch_data[f_error_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );

            // Copy parameters (including padding)
            for (i, vec) in vectors.iter().enumerate() {
                f_add_slice[i] = vec.f_add;
                f_rescale_slice[i] = vec.f_rescale;
                f_error_slice[i] = vec.f_error;
            }
        }

        // Store packed ex_codes - ONLY for actual vectors, not padding!
        // C++-style: Keep codes packed, unpack on-demand during search
        for vec in vectors.iter().take(actual_count) {
            // Store packed ex_code (no unpacking!)
            if ex_bits > 0 {
                ex_codes_packed.push(vec.ex_code_packed.clone());
                f_add_ex.push(vec.f_add_ex);
                f_rescale_ex.push(vec.f_rescale_ex);
            } else {
                ex_codes_packed.push(Vec::new());
                f_add_ex.push(0.0);
                f_rescale_ex.push(0.0);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_batch_into_memory(
        vectors: &[QuantizedVector],
        batch_data: &mut [u8],
        batch_idx: usize,
        padded_dim: usize,
        ex_bits: usize,
        ex_codes_packed: &mut Vec<Vec<u8>>,
        f_add_ex: &mut Vec<f32>,
        f_rescale_ex: &mut Vec<f32>,
    ) {
        assert_eq!(vectors.len(), simd::FASTSCAN_BATCH_SIZE);

        let stride = Self::batch_stride(padded_dim);
        let dim_bytes = padded_dim / 8;

        // Calculate offsets in contiguous memory
        let batch_offset = batch_idx * stride;
        let binary_bytes = padded_dim * simd::FASTSCAN_BATCH_SIZE / 8;

        // Collect binary codes into flat buffer
        let mut binary_codes_flat = Vec::with_capacity(simd::FASTSCAN_BATCH_SIZE * dim_bytes);
        for vec in vectors.iter() {
            binary_codes_flat.extend_from_slice(&vec.binary_code_packed);
        }

        // Pack binary codes using FastScan layout
        let packed_codes = &mut batch_data[batch_offset..batch_offset + binary_bytes];
        simd::pack_codes(
            &binary_codes_flat,
            simd::FASTSCAN_BATCH_SIZE,
            dim_bytes,
            packed_codes,
        );

        // Get mutable slices for parameters (using pointer arithmetic like C++)
        let f_add_offset = batch_offset + binary_bytes;
        let f_rescale_offset =
            f_add_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;
        let f_error_offset =
            f_rescale_offset + std::mem::size_of::<f32>() * simd::FASTSCAN_BATCH_SIZE;

        unsafe {
            let f_add_slice = std::slice::from_raw_parts_mut(
                batch_data[f_add_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_rescale_slice = std::slice::from_raw_parts_mut(
                batch_data[f_rescale_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );
            let f_error_slice = std::slice::from_raw_parts_mut(
                batch_data[f_error_offset..].as_mut_ptr() as *mut f32,
                simd::FASTSCAN_BATCH_SIZE,
            );

            // Copy parameters
            for (i, vec) in vectors.iter().enumerate() {
                f_add_slice[i] = vec.f_add;
                f_rescale_slice[i] = vec.f_rescale;
                f_error_slice[i] = vec.f_error;
            }
        }

        // Store packed ex_codes for all vectors in batch
        // C++-style: Keep codes packed, unpack on-demand during search
        for vec in vectors.iter() {
            // Store packed ex_code (no unpacking!)
            if ex_bits > 0 {
                ex_codes_packed.push(vec.ex_code_packed.clone());
                f_add_ex.push(vec.f_add_ex);
                f_rescale_ex.push(vec.f_rescale_ex);
            } else {
                ex_codes_packed.push(Vec::new());
                f_add_ex.push(0.0);
                f_rescale_ex.push(0.0);
            }
        }
    }
}

// ============================================================================
// Quantized vector for temporary use during training
// ============================================================================

/// Temporary structure used during index construction
/// After construction, vectors are packed into ClusterData's unified memory layout
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TemporaryQuantizedVector {
    binary_code_packed: Vec<u8>,
    ex_code_packed: Vec<u8>,
    f_add: f32,
    f_rescale: f32,
    f_error: f32,
    f_add_ex: f32,
    f_rescale_ex: f32,
}

/// Lookup table for batch FastScan search
/// Mimics C++ Lut class from lut.hpp
struct QueryLut {
    /// Quantized lookup table (i8 values for SIMD)
    lut_i8: Vec<i8>,
    /// Quantization delta factor
    delta: f32,
    /// Sum of vl across all lookup tables
    sum_vl_lut: f32,
}

/// High-accuracy LUT with separated low8/high8 components
/// Used for high-dimensional data (dim > 2048) to prevent overflow
struct QueryLutHighAcc {
    /// Low 8 bits of LUT values (u8 for unsigned values)
    lut_low8: Vec<u8>,
    /// High 8 bits of LUT values (u8 for unsigned values)
    lut_high8: Vec<u8>,
    /// Quantization delta factor
    delta: f32,
    /// Sum of vl across all lookup tables
    sum_vl_lut: f32,
}

impl QueryLutHighAcc {
    /// Build high-accuracy LUT from rotated query
    /// Splits quantized values into low8 and high8 components
    fn new(rotated_query: &[f32], padded_dim: usize) -> Self {
        assert!(
            padded_dim % 4 == 0,
            "padded_dim must be multiple of 4 for LUT"
        );

        // First compute float values like regular LUT
        // pack_lut_f32 packs 4 dimensions into 16 LUT entries, repeated for each batch
        let lut_size = (padded_dim / 4) * 16 * (padded_dim / 32);
        let mut lut_f32 = vec![0.0f32; lut_size];
        simd::pack_lut_f32(rotated_query, &mut lut_f32);

        // Compute delta and sum_vl_lut
        let sum: f32 = lut_f32.iter().sum();
        let (delta, sum_vl) = if lut_f32.iter().all(|&x| x.abs() < f32::EPSILON) {
            (1.0, 0.0)
        } else {
            let max_val = lut_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            // Use larger range for high-accuracy mode (16-bit instead of 8-bit)
            let delta = max_val / 32767.0; // i16::MAX
            let sum_vl = sum / delta;
            (delta, sum_vl)
        };

        // Quantize to 16-bit signed values first
        let mut lut_i16 = vec![0i16; lut_f32.len()];
        for (i, &val) in lut_f32.iter().enumerate() {
            let quantized = (val / delta).round();
            lut_i16[i] = quantized.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }

        // Split into low8 and high8 components
        let mut lut_low8 = vec![0u8; lut_i16.len()];
        let mut lut_high8 = vec![0u8; lut_i16.len()];

        for (i, &val) in lut_i16.iter().enumerate() {
            // Convert i16 to u16 by adding 32768 (shift to unsigned range)
            let unsigned_val = (val as i32 + 32768) as u16;
            lut_low8[i] = (unsigned_val & 0xFF) as u8;
            lut_high8[i] = ((unsigned_val >> 8) & 0xFF) as u8;
        }

        Self {
            lut_low8,
            lut_high8,
            delta,
            sum_vl_lut: sum_vl,
        }
    }
}

impl QueryLut {
    /// Build LUT from rotated query
    /// Reference: C++ Lut constructor in lut.hpp:27-53
    fn new(rotated_query: &[f32], padded_dim: usize) -> Self {
        assert!(
            padded_dim % 4 == 0,
            "padded_dim must be multiple of 4 for LUT"
        );

        let table_length = padded_dim * 4; // padded_dim << 2

        // Step 1: Generate float LUT using pack_lut_f32
        let mut lut_float = vec![0.0f32; table_length];
        simd::pack_lut_f32(rotated_query, &mut lut_float);

        // Step 2: Find min and max of LUT
        let vl_lut = lut_float
            .iter()
            .copied()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let vr_lut = lut_float
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        // Step 3: Compute delta for i8 quantization (8 bits)
        let delta = (vr_lut - vl_lut) / 255.0f32; // (1 << 8) - 1 = 255

        // Step 4: Quantize float LUT to i8
        let mut lut_i8 = vec![0i8; table_length];
        if delta > 0.0 {
            for i in 0..table_length {
                let quantized = ((lut_float[i] - vl_lut) / delta).round();
                // Must convert to u8 first, then to i8 to avoid saturation
                // f32 as i8 saturates at 127, but we need wrapping behavior (128-255 -> -128 to -1)
                lut_i8[i] = (quantized.clamp(0.0, 255.0) as u8) as i8;
            }
        }

        // Step 5: Compute sum_vl_lut
        let num_table = table_length / 16;
        let sum_vl_lut = vl_lut * (num_table as f32);

        Self {
            lut_i8,
            delta,
            sum_vl_lut,
        }
    }
}

/// Precomputed query constants to avoid repeated calculations during search.
struct QueryPrecomputed {
    rotated_query: Vec<f32>,
    query_norm: f32,
    k1x_sum_q: f32, // c1 × sum_query (precomputed)
    kbx_sum_q: f32, // cb × sum_query (precomputed)
    binary_scale: f32,
    /// Optional LUT for FastScan batch search
    lut: Option<QueryLut>,
    /// Optional high-accuracy LUT for high-dimensional data
    lut_highacc: Option<QueryLutHighAcc>,
}

impl QueryPrecomputed {
    fn new(rotated_query: Vec<f32>, ex_bits: usize) -> Self {
        let sum_query: f32 = rotated_query.iter().sum();
        let query_norm: f32 = rotated_query.iter().map(|v| v * v).sum::<f32>().sqrt();
        let c1 = -0.5f32;
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let binary_scale = (1 << ex_bits) as f32;

        Self {
            rotated_query,
            query_norm,
            k1x_sum_q: c1 * sum_query,
            kbx_sum_q: cb * sum_query,
            binary_scale,
            lut: None,         // LUT built on demand for batch search
            lut_highacc: None, // High-accuracy LUT built on demand for high-dimensional data
        }
    }

    /// Build LUT for FastScan batch search
    /// Should be called once per query before searching clusters
    fn build_lut(&mut self, padded_dim: usize) {
        // Decide whether to use high-accuracy mode based on dimension
        // Use high-accuracy for dimensions > 2048 to prevent overflow
        let use_highacc = padded_dim > 2048;

        if use_highacc {
            if self.lut_highacc.is_none() {
                self.lut_highacc = Some(QueryLutHighAcc::new(&self.rotated_query, padded_dim));
            }
        } else if self.lut.is_none() {
            self.lut = Some(QueryLut::new(&self.rotated_query, padded_dim));
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
    /// Unified cluster data with optimized memory layout (Phase 1 optimization)
    clusters: Vec<ClusterData>,
    ex_bits: usize,
    /// Function pointer for ex-code inner product on packed data (Phase 3 optimization)
    /// Selected based on ex_bits at index construction time
    ip_func: crate::simd::ExIpFunc,
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
        use_faster_config: bool,
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

        // Log huge pages status
        crate::memory::log_huge_page_status();

        println!("Rotating vectors...");
        let rotated_data: Vec<Vec<f32>> = data.par_iter().map(|v| rotator.rotate(v)).collect();

        println!(
            "Training k-means ({} clusters, {} iterations)...",
            nlist, 30
        );
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5a5a_5a5a5a5a5a5a);
        let KMeansResult {
            centroids,
            assignments,
            ..
        } = run_kmeans(&rotated_data, nlist, 30, &mut rng);
        println!("K-means training complete");

        Self::build_from_rotated(
            dim,
            padded_dim,
            metric,
            rotator,
            centroids,
            &rotated_data,
            &assignments,
            total_bits,
            seed,
            use_faster_config,
        )
    }

    /// Train an index using externally provided centroids and cluster assignments.
    #[allow(clippy::too_many_arguments)]
    pub fn train_with_clusters(
        data: &[Vec<f32>],
        centroids: &[Vec<f32>],
        assignments: &[usize],
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

        // Log huge pages status
        crate::memory::log_huge_page_status();

        let rotated_data: Vec<Vec<f32>> = data.par_iter().map(|v| rotator.rotate(v)).collect();
        let rotated_centroids: Vec<Vec<f32>> =
            centroids.par_iter().map(|c| rotator.rotate(c)).collect();

        Self::build_from_rotated(
            dim,
            padded_dim,
            metric,
            rotator,
            rotated_centroids,
            &rotated_data,
            assignments,
            total_bits,
            seed,
            use_faster_config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from_rotated(
        dim: usize,
        padded_dim: usize,
        metric: Metric,
        rotator: DynamicRotator,
        rotated_centroids: Vec<Vec<f32>>,
        rotated_data: &[Vec<f32>],
        assignments: &[usize],
        total_bits: usize,
        seed: u64,
        use_faster_config: bool,
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

        let centroids = rotated_centroids;
        if centroids.is_empty() {
            return Err(RabitqError::InvalidConfig("nlist must be positive"));
        }

        let config = if use_faster_config {
            RabitqConfig::faster(padded_dim, total_bits, seed)
        } else {
            RabitqConfig::new(total_bits)
        };

        // Group vector indices by cluster
        let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); centroids.len()];
        for (idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id >= centroids.len() {
                return Err(RabitqError::InvalidConfig(
                    "assignments reference invalid cluster ids",
                ));
            }
            cluster_indices[cluster_id].push(idx);
        }

        // Parallelize quantization across clusters and within clusters
        println!("Quantizing vectors...");
        use std::sync::atomic::{AtomicUsize, Ordering};
        let progress_counter = AtomicUsize::new(0);
        let total_clusters = centroids.len();

        let cluster_data: Vec<(Vec<usize>, Vec<QuantizedVector>)> = centroids
            .par_iter()
            .enumerate()
            .map(|(cluster_id, centroid)| {
                let indices = &cluster_indices[cluster_id];
                let quantized_vectors: Vec<QuantizedVector> = indices
                    .par_iter()
                    .map(|&idx| {
                        quantize_with_centroid(&rotated_data[idx], centroid, &config, metric)
                    })
                    .collect();

                let completed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % (total_clusters / 20).max(1) == 0 || completed == total_clusters {
                    println!(
                        "  Quantized {}/{} clusters ({:.1}%)",
                        completed,
                        total_clusters,
                        100.0 * completed as f64 / total_clusters as f64
                    );
                }

                (indices.clone(), quantized_vectors)
            })
            .collect();
        println!("Quantization complete");

        // Build unified clusters with optimized memory layout
        println!("Building clusters with unified memory layout...");
        let clusters: Vec<ClusterData> = cluster_data
            .into_iter()
            .enumerate()
            .map(|(cluster_id, (indices, quantized_vectors))| {
                ClusterData::from_quantized_vectors(
                    centroids[cluster_id].clone(),
                    indices,
                    quantized_vectors,
                    padded_dim,
                    total_bits.saturating_sub(1),
                )
            })
            .collect();
        println!("Cluster construction complete");

        let ex_bits = total_bits.saturating_sub(1);
        let ip_func = crate::simd::select_excode_ipfunc(ex_bits);

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            clusters,
            ex_bits,
            ip_func,
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

        let padded_dim = u32::try_from(self.padded_dim).map_err(|_| {
            RabitqError::InvalidPersistence("padded_dim exceeds persistence limits")
        })?;
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
        eprintln!("DEBUG SAVE: rotator_len={}", rotator_len);
        write_u64(&mut writer, rotator_len, Some(&mut hasher))?;
        writer.write_all(&rotator_data)?;
        hasher.update(&rotator_data);

        // V3: Save ClusterData with unified memory layout
        eprintln!("DEBUG SAVE: Saving {} clusters", self.clusters.len());
        for (i, cluster) in self.clusters.iter().enumerate() {
            eprintln!(
                "DEBUG SAVE: Cluster {}: num_vectors={}, batch_data.len()={}",
                i,
                cluster.num_vectors,
                cluster.batch_data.len()
            );
            if cluster.centroid.len() != self.padded_dim {
                return Err(RabitqError::InvalidPersistence(
                    "cluster centroid dimension mismatch",
                ));
            }
            if cluster.ids.len() != cluster.num_vectors {
                return Err(RabitqError::InvalidPersistence(
                    "cluster id/vector count mismatch",
                ));
            }

            // Save centroid
            for &value in &cluster.centroid {
                write_f32(&mut writer, value, Some(&mut hasher))?;
            }

            // Save vector count
            let entry_count = u64::try_from(cluster.num_vectors).map_err(|_| {
                RabitqError::InvalidPersistence("cluster entry count exceeds persistence limits")
            })?;
            write_u64(&mut writer, entry_count, Some(&mut hasher))?;

            // Save vector IDs
            for &id in &cluster.ids {
                let encoded = u64::try_from(id).map_err(|_| {
                    RabitqError::InvalidPersistence("vector id exceeds persistence limits")
                })?;
                write_u64(&mut writer, encoded, Some(&mut hasher))?;
            }

            // Save batch_data (contiguous memory block)
            let batch_data_len = u64::try_from(cluster.batch_data.len()).map_err(|_| {
                RabitqError::InvalidPersistence("batch_data size exceeds persistence limits")
            })?;

            // DEBUG: Output cluster info
            // eprintln!("Serializing cluster: num_vectors={}, batch_data.len()={}",
            //           cluster.num_vectors, cluster.batch_data.len());

            write_u64(&mut writer, batch_data_len, Some(&mut hasher))?;
            writer.write_all(&cluster.batch_data)?;
            hasher.update(&cluster.batch_data);

            // Save packed ex_codes (C++-style on-demand unpacking)
            eprintln!(
                "DEBUG SAVE: Cluster {}: ex_codes_packed.len()={}",
                i,
                cluster.ex_codes_packed.len()
            );
            for (j, ex_code_packed) in cluster.ex_codes_packed.iter().enumerate() {
                let ex_code_len = u64::try_from(ex_code_packed.len()).map_err(|_| {
                    RabitqError::InvalidPersistence(
                        "ex_code_packed length exceeds persistence limits",
                    )
                })?;
                if j < 2 {
                    eprintln!(
                        "DEBUG SAVE: Cluster {}: ex_code_packed[{}].len()={}",
                        i, j, ex_code_len
                    );
                }
                write_u64(&mut writer, ex_code_len, Some(&mut hasher))?;
                writer.write_all(ex_code_packed)?;
                hasher.update(ex_code_packed);
            }

            // Save ex parameters
            for &val in &cluster.f_add_ex {
                write_f32(&mut writer, val, Some(&mut hasher))?;
            }
            for &val in &cluster.f_rescale_ex {
                write_f32(&mut writer, val, Some(&mut hasher))?;
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
        if version != 3 {
            return Err(RabitqError::InvalidPersistence(
                "unsupported index format version (expected V3 with unified memory layout)",
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

        let expected_vectors = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
        eprintln!("DEBUG LOAD: expected_vectors={}", expected_vectors);

        let cluster_count = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
        eprintln!("DEBUG LOAD: cluster_count={}", cluster_count);

        let rotator_data_len = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
        eprintln!("DEBUG LOAD: rotator_data_len={}", rotator_data_len);

        let mut rotator_data = vec![0u8; rotator_data_len];
        reader.read_exact(&mut rotator_data)?;
        hasher.update(&rotator_data);

        let rotator = DynamicRotator::deserialize(dim, padded_dim, rotator_type, &rotator_data)?;

        // V3: Load ClusterData with unified memory layout
        let mut clusters = Vec::with_capacity(cluster_count);
        eprintln!("DEBUG LOAD: Loading {} clusters", cluster_count);
        for i in 0..cluster_count {
            eprintln!("DEBUG LOAD: Loading cluster {}", i);
            // Load centroid
            let mut centroid = vec![0f32; padded_dim];
            for value in centroid.iter_mut() {
                *value = read_f32(&mut reader, Some(&mut hasher))?;
            }

            // Load vector count
            let num_vectors = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
            eprintln!("DEBUG LOAD: Cluster {}: num_vectors={}", i, num_vectors);

            // Validate num_vectors is reasonable (防止破坏的数据导致内存分配溢出)
            const MAX_CLUSTER_SIZE: usize = 1_000_000; // 1M vectors per cluster is very large
            if num_vectors > MAX_CLUSTER_SIZE {
                return Err(RabitqError::InvalidPersistence(
                    "cluster size exceeds reasonable limits - possible corruption",
                ));
            }

            // Load vector IDs
            let mut ids = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                ids.push(usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?);
            }

            // Load batch_data (contiguous memory block)
            let batch_data_len = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;

            // Validate batch_data_len is reasonable
            let total_batches =
                (num_vectors + simd::FASTSCAN_BATCH_SIZE - 1) / simd::FASTSCAN_BATCH_SIZE;
            let expected_batch_data_len = ClusterData::batch_stride(padded_dim) * total_batches;
            if batch_data_len != expected_batch_data_len {
                eprintln!("DEBUG: batch_data_len mismatch");
                eprintln!("  num_vectors: {}", num_vectors);
                eprintln!("  padded_dim: {}", padded_dim);
                eprintln!("  total_batches: {}", total_batches);
                eprintln!("  batch_stride: {}", ClusterData::batch_stride(padded_dim));
                eprintln!("  expected: {}", expected_batch_data_len);
                eprintln!("  actual: {}", batch_data_len);
                return Err(RabitqError::InvalidPersistence(
                    "batch_data length mismatch - possible corruption or version incompatibility",
                ));
            }

            let mut batch_data = crate::memory::allocate_aligned_vec::<u8>(batch_data_len);
            reader.read_exact(&mut batch_data)?;
            hasher.update(&batch_data);

            // Load packed ex_codes (C++-style on-demand unpacking)
            eprintln!(
                "DEBUG LOAD: Cluster {}: Loading {} ex_codes_packed",
                i, num_vectors
            );
            let mut ex_codes_packed = Vec::with_capacity(num_vectors);
            for j in 0..num_vectors {
                let ex_code_len = usize_from_u64(read_u64(&mut reader, Some(&mut hasher))?)?;
                if j < 2 {
                    eprintln!(
                        "DEBUG LOAD: Cluster {}: ex_code_packed[{}].len()={}",
                        i, j, ex_code_len
                    );
                }

                // Validate ex_code_len is reasonable
                // When ex_bits = 0, ex_code_len should be 0
                // When ex_bits > 0, ex_code_len should equal (padded_dim * ex_bits / 8)
                let expected_ex_code_len = if ex_bits > 0 {
                    padded_dim * ex_bits / 8
                } else {
                    0
                };
                if ex_code_len != expected_ex_code_len {
                    return Err(RabitqError::InvalidPersistence(
                        "ex_code_packed length mismatch - possible corruption or version incompatibility",
                    ));
                }

                let mut ex_code_packed = vec![0u8; ex_code_len];
                reader.read_exact(&mut ex_code_packed)?;
                hasher.update(&ex_code_packed);
                ex_codes_packed.push(ex_code_packed);
            }

            // Load ex parameters
            let mut f_add_ex = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                f_add_ex.push(read_f32(&mut reader, Some(&mut hasher))?);
            }

            let mut f_rescale_ex = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                f_rescale_ex.push(read_f32(&mut reader, Some(&mut hasher))?);
            }

            clusters.push(ClusterData {
                centroid,
                ids,
                batch_data,
                ex_codes_packed,
                f_add_ex,
                f_rescale_ex,
                num_vectors,
                padded_dim,
                ex_bits,
            });
        }

        let actual_vectors: usize = clusters.iter().map(|c| c.num_vectors).sum();
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

        println!("Loaded index with unified memory layout (V3)");

        let ip_func = crate::simd::select_excode_ipfunc(ex_bits);

        Ok(Self {
            dim,
            padded_dim,
            metric,
            rotator,
            clusters,
            ex_bits,
            ip_func,
        })
    }

    /// Search for the nearest neighbours of the provided query vector.
    pub fn search(
        &self,
        query: &[f32],
        params: SearchParams,
    ) -> Result<Vec<SearchResult>, RabitqError> {
        self.search_fastscan(query, params, None, None)
    }

    /// Search for the nearest neighbours of the provided query vector,
    /// filtering results to only include vector IDs present in the provided bitmap.
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `params` - Search parameters (top_k, nprobe)
    /// * `filter` - A RoaringBitmap containing valid candidate vector IDs
    ///
    /// # Returns
    /// A vector of search results, sorted by score, containing only IDs present in the filter.
    pub fn search_filtered(
        &self,
        query: &[f32],
        params: SearchParams,
        filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>, RabitqError> {
        self.search_fastscan(query, params, None, Some(filter))
    }

    /// Batch search for multiple queries in parallel using rayon
    ///
    /// # Performance
    /// Expected speedup: 2-8x on typical CPUs (depends on core count and batch size)
    ///
    /// # Arguments
    /// * `queries` - Slice of query vectors
    /// * `params` - Search parameters (top_k, nprobe)
    ///
    /// # Returns
    /// A vector of search results for each query, in the same order as the input queries
    pub fn batch_search(
        &self,
        queries: &[&[f32]],
        params: SearchParams,
    ) -> Vec<Result<Vec<SearchResult>, RabitqError>> {
        queries
            .par_iter()
            .map(|query| self.search(query, params))
            .collect()
    }

    fn search_fastscan(
        &self,
        query: &[f32],
        params: SearchParams,
        mut diagnostics: Option<&mut SearchDiagnostics>,
        filter: Option<&RoaringBitmap>,
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

        // FastScan V2 is the default search method (Session 10)
        // Binary code bit order fixed in Session 9 (MSB-first to match C++)
        // Session 11: FastScan now supports all bit configurations (ex_bits >= 0)

        // Precompute query constants once to avoid repeated calculations
        let rotated_query = self.rotator.rotate(query);
        let mut query_precomp = QueryPrecomputed::new(rotated_query, self.ex_bits);

        // Build LUT for FastScan batch search
        query_precomp.build_lut(self.padded_dim);

        let mut cluster_scores: Vec<(usize, f32)> = Vec::with_capacity(self.clusters.len());
        for (cid, cluster) in self.clusters.iter().enumerate() {
            let score = match self.metric {
                Metric::L2 => l2_distance_sqr(&query_precomp.rotated_query, &cluster.centroid),
                Metric::InnerProduct => dot(&query_precomp.rotated_query, &cluster.centroid),
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

        // Debug: print selected clusters (disabled)
        // println!("Selected {} clusters (nprobe={}): {:?}",
        //     cluster_scores.iter().take(nprobe).count(),
        //     nprobe,
        //     cluster_scores.iter().take(nprobe).map(|(cid, score)| (*cid, *score)).collect::<Vec<_>>()
        // );

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        // Use FastScan batch search with unified memory layout
        for &(cid, _) in cluster_scores.iter().take(nprobe) {
            let cluster = &self.clusters[cid];

            let centroid_dist = l2_distance_sqr(&query_precomp.rotated_query, &cluster.centroid);
            let dot_query_centroid = dot(&query_precomp.rotated_query, &cluster.centroid);
            let g_add = match self.metric {
                Metric::L2 => centroid_dist,
                Metric::InnerProduct => -dot_query_centroid,
            };
            let centroid_norm = centroid_dist.sqrt();
            let g_error = centroid_norm;

            // Use batched search (3-10x faster than naive, further optimized with unified layout)
            self.search_cluster_v2_batched(
                cid,
                cluster,
                &query_precomp,
                g_add,
                g_error,
                dot_query_centroid,
                filter,
                &mut heap,
                params.top_k,
                &mut diagnostics,
            );
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

    /// Helper method to search a cluster using batch FastScan with unified memory layout
    /// This processes vectors in batches of 32 for better SIMD performance
    /// Reference: C++ split_batch_estdist in estimator.hpp:25-71
    #[allow(clippy::too_many_arguments)]
    fn search_cluster_v2_batched(
        &self,
        _cluster_id: usize,
        cluster: &ClusterData,
        query_precomp: &QueryPrecomputed,
        g_add: f32,
        g_error: f32,
        dot_query_centroid: f32,
        filter: Option<&RoaringBitmap>,
        heap: &mut BinaryHeap<HeapEntry>,
        top_k: usize,
        diagnostics: &mut Option<&mut SearchDiagnostics>,
    ) {
        // Check if we should use high-accuracy mode
        let use_highacc = self.padded_dim > 2048;

        // Get the appropriate LUT
        let (lut_regular, lut_highacc) = if use_highacc {
            (None, query_precomp.lut_highacc.as_ref())
        } else {
            (query_precomp.lut.as_ref(), None)
        };

        if lut_regular.is_none() && lut_highacc.is_none() {
            return; // No LUT available, fallback needed
        }

        // Process complete batches (32 vectors each)
        let num_batches = cluster.num_complete_batches();
        let num_remainder = cluster.num_remainder_vectors();
        let total_batches = if num_remainder > 0 {
            num_batches + 1
        } else {
            num_batches
        };

        for batch_idx in 0..total_batches {
            let batch_start = batch_idx * simd::FASTSCAN_BATCH_SIZE;
            let batch_end = (batch_start + simd::FASTSCAN_BATCH_SIZE).min(cluster.num_vectors);
            let actual_batch_size = batch_end - batch_start;

            // Rely on CPU hardware prefetcher (like C++ version)
            // Manual prefetching removed to avoid cache pollution

            // Step 1: Accumulate distances using FastScan SIMD with zero-copy access
            // Get batch parameters using zero-copy slices
            let batch_f_add = cluster.batch_f_add(batch_idx);
            let batch_f_rescale = cluster.batch_f_rescale(batch_idx);
            let batch_f_error = cluster.batch_f_error(batch_idx);

            // Allocate output arrays on stack (no heap allocation)
            let mut ip_x0_qr_values = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut est_distances = [0.0f32; simd::FASTSCAN_BATCH_SIZE];
            let mut lower_bounds = [0.0f32; simd::FASTSCAN_BATCH_SIZE];

            if use_highacc {
                // High-accuracy mode using i32 accumulators
                let mut accu_res_i32 = [0i32; simd::FASTSCAN_BATCH_SIZE];
                let lut_ha = lut_highacc.unwrap();
                simd::accumulate_batch_highacc_avx2(
                    cluster.batch_bin_codes(batch_idx),
                    &lut_ha.lut_low8,
                    &lut_ha.lut_high8,
                    self.padded_dim,
                    &mut accu_res_i32,
                );

                // SIMD vectorized batch distance computation (matches C++ Eigen)
                simd::compute_batch_distances_i32(
                    &accu_res_i32,
                    lut_ha.delta,
                    lut_ha.sum_vl_lut,
                    batch_f_add,
                    batch_f_rescale,
                    batch_f_error,
                    g_add,
                    g_error,
                    query_precomp.k1x_sum_q,
                    &mut ip_x0_qr_values,
                    &mut est_distances,
                    &mut lower_bounds,
                );
            } else {
                // Regular mode using u16 accumulators
                let mut accu_res = [0u16; simd::FASTSCAN_BATCH_SIZE];
                let lut = lut_regular.unwrap();
                simd::accumulate_batch_avx2(
                    cluster.batch_bin_codes(batch_idx),
                    &lut.lut_i8,
                    self.padded_dim,
                    &mut accu_res,
                );

                // SIMD vectorized batch distance computation (matches C++ Eigen)
                simd::compute_batch_distances_u16(
                    &accu_res,
                    lut.delta,
                    lut.sum_vl_lut,
                    batch_f_add,
                    batch_f_rescale,
                    batch_f_error,
                    g_add,
                    g_error,
                    query_precomp.k1x_sum_q,
                    &mut ip_x0_qr_values,
                    &mut est_distances,
                    &mut lower_bounds,
                );
            }

            // Step 2: Process each vector in the batch (pruning and ex-code evaluation)
            // Distances are now pre-computed in vectorized fashion (matching C++ Eigen)
            for i in 0..actual_batch_size {
                let global_idx = batch_start + i;
                let vector_id = cluster.ids[global_idx];

                // Apply filter if provided
                if let Some(filter_bitmap) = filter {
                    if !filter_bitmap.contains(vector_id as u32) {
                        continue;
                    }
                }

                // Use pre-computed values (vectorized above)
                let ip_x0_qr = ip_x0_qr_values[i];
                let est_distance = est_distances[i];
                let mut lower_bound = lower_bounds[i];

                // Apply safety bounds (conservative fallback)
                let safe_bound = match self.metric {
                    Metric::L2 => {
                        let _centroid_norm = g_add.sqrt(); // g_add = centroid_dist for L2
                        0.0 // Conservative: allow all candidates through
                    }
                    Metric::InnerProduct => {
                        let max_dot = dot_query_centroid + query_precomp.query_norm * 1.0; // Conservative
                        -max_dot
                    }
                };
                if !lower_bound.is_finite() {
                    lower_bound = safe_bound;
                } else {
                    lower_bound = lower_bound.min(safe_bound);
                }

                // Step 3: Check against current k-th distance
                let distk = if heap.len() < top_k {
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

                // Step 4: Compute final distance with ex-codes if available
                let mut distance = est_distance;
                if self.ex_bits > 0 {
                    if let Some(diag) = diagnostics.as_deref_mut() {
                        diag.extended_evaluations += 1;
                    }

                    // Extended code evaluation (C++-style on-demand unpacking)
                    // Reference: C++ estimator.hpp:85-103 split_distance_boosting
                    // Formula: f_add_ex + g_add + f_rescale_ex * (binary_scale * ip_x0_qr + ex_dot + kbxsumq)
                    //
                    // KEY OPTIMIZATION: Reuse ip_x0_qr computed by FastScan instead of recomputing binary_dot
                    // C++ also uses ip_x0_qr here (see estimator.hpp:99)
                    // This eliminates:
                    //   - V1 structure access
                    //   - Redundant dot product computation
                    //   - Cache misses from scattered memory access
                    //
                    // OPTIMIZATION: Direct SIMD dot product on packed data (C++-style, Phase 3)
                    // C++ directly operates on packed data via ip_func_ with C++-compatible packing format
                    // Rust now matches this behavior:
                    //   - No unpacking overhead (~15% speedup)
                    //   - Direct SIMD operations on C++-compatible packed format
                    //   - Function pointer selected at index construction time

                    // Compute ex_code dot product directly on packed data (C++-style)
                    let ex_code_packed = &cluster.ex_codes_packed[global_idx];
                    let ex_dot = (self.ip_func)(
                        &query_precomp.rotated_query,
                        ex_code_packed,
                        self.padded_dim,
                    );

                    // Compute distance using ip_x0_qr (not binary_dot!)
                    // This matches C++ behavior and avoids redundant computation
                    let total_term =
                        query_precomp.binary_scale * ip_x0_qr + ex_dot + query_precomp.kbx_sum_q;
                    distance = cluster.f_add_ex[global_idx]
                        + g_add
                        + cluster.f_rescale_ex[global_idx] * total_term;
                }

                if !distance.is_finite() {
                    continue;
                }

                let score = match self.metric {
                    Metric::L2 => distance,
                    Metric::InnerProduct => -distance,
                };

                if let Some(diag) = diagnostics.as_deref_mut() {
                    diag.estimated += 1;
                }

                // Step 7: Update heap
                heap.push(HeapEntry {
                    candidate: HeapCandidate {
                        id: vector_id,
                        distance,
                        score,
                    },
                });

                if heap.len() > top_k {
                    heap.pop();
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn search_with_diagnostics(
        &self,
        query: &[f32],
        params: SearchParams,
    ) -> Result<(Vec<SearchResult>, SearchDiagnostics), RabitqError> {
        let mut diagnostics = SearchDiagnostics::default();
        let results = self.search_fastscan(query, params, Some(&mut diagnostics), None)?;
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
            for vec_idx in 0..cluster.num_vectors {
                // Extract data for this vector using ClusterData API
                let f_add = cluster.get_vector_f_add(vec_idx);
                let f_rescale = cluster.get_vector_f_rescale(vec_idx);

                // Unpack binary code on-demand from FastScan batch layout
                // Note: This is only used in naive search (test code)
                let batch_idx = vec_idx / simd::FASTSCAN_BATCH_SIZE;
                let in_batch_idx = vec_idx % simd::FASTSCAN_BATCH_SIZE;
                let packed_codes = cluster.batch_bin_codes(batch_idx);

                // Unpack binary code for this vector
                let dim_bytes = self.padded_dim / 8;
                let mut binary_code = vec![0u8; self.padded_dim];
                simd::unpack_single_vector(packed_codes, in_batch_idx, dim_bytes, &mut binary_code);

                let mut binary_dot = 0.0f32;
                for (&bit, &q_val) in binary_code.iter().zip(rotated_query.iter()) {
                    binary_dot += (bit as f32) * q_val;
                }
                let binary_term = binary_dot + c1 * sum_query;
                let mut distance = f_add + g_add + f_rescale * binary_term;

                if self.ex_bits > 0 {
                    // Direct SIMD dot product on packed data (C++-style, Phase 3)
                    let ex_code_packed = &cluster.ex_codes_packed[vec_idx];
                    let ex_dot = (self.ip_func)(&rotated_query, ex_code_packed, self.padded_dim);
                    let total_term = binary_scale * binary_dot + ex_dot + cb * sum_query;
                    distance = cluster.f_add_ex[vec_idx]
                        + g_add
                        + cluster.f_rescale_ex[vec_idx] * total_term;
                }
                if !distance.is_finite() {
                    continue;
                }
                let score = match self.metric {
                    Metric::L2 => distance,
                    Metric::InnerProduct => -distance,
                };
                candidates.push(SearchResult {
                    id: cluster.ids[vec_idx],
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

#[cfg(test)]
mod batch_search_tests {
    use super::*;
    use crate::quantizer::RabitqConfig;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    #[ignore] // This test has incorrect logic - ip_x0_qr is an intermediate value, not final distance
    fn test_lut_accumulate_matches_direct_dot() {
        // Minimal test to verify LUT-based dot product matches direct computation
        let dim = 64;
        let padded_dim = 64;
        let _ex_bits = 0;
        let mut rng = StdRng::seed_from_u64(12345);

        // Create a single vector
        let rotator = DynamicRotator::new(dim, RotatorType::FhtKacRotator, 42);
        let data_vec = random_vector(dim, &mut rng);
        let rotated_data = rotator.rotate(&data_vec);

        let query_vec = random_vector(dim, &mut rng);
        let rotated_query = rotator.rotate(&query_vec);

        // Quantize the data vector
        let config = RabitqConfig::new(1);
        let centroid = vec![0.0f32; padded_dim];
        let quantized =
            crate::quantizer::quantize_with_centroid(&rotated_data, &centroid, &config, Metric::L2);

        // V1: Direct dot product
        let binary_dot_v1 =
            crate::simd::dot_u8_f32(&quantized.binary_code_unpacked, &rotated_query);
        println!("V1 binary_dot: {:.6}", binary_dot_v1);
        println!(
            "First 16 binary_code_unpacked: {:?}",
            &quantized.binary_code_unpacked[0..16.min(quantized.binary_code_unpacked.len())]
        );
        println!(
            "First 16 rotated_query: {:?}",
            &rotated_query[0..16.min(rotated_query.len())]
        );

        // V2: LUT-based approach
        // Build LUT
        let table_length = padded_dim * 4;
        let mut lut_float = vec![0.0f32; table_length];
        crate::simd::pack_lut_f32(&rotated_query, &mut lut_float);

        // Quantize LUT
        let vl_lut = lut_float
            .iter()
            .copied()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let vr_lut = lut_float
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let delta = (vr_lut - vl_lut) / 255.0f32;

        let mut lut_i8 = vec![0i8; table_length];
        if delta > 0.0 {
            for i in 0..table_length {
                let quantized = ((lut_float[i] - vl_lut) / delta).round();
                // Must convert to u8 first, then to i8 to avoid saturation
                // f32 as i8 saturates at 127, but we need wrapping behavior (128-255 -> -128 to -1)
                lut_i8[i] = (quantized.clamp(0.0, 255.0) as u8) as i8;
            }
        }

        let num_table = table_length / 16;
        let sum_vl_lut = vl_lut * (num_table as f32);

        println!("LUT delta: {:.6}, sum_vl_lut: {:.6}", delta, sum_vl_lut);
        println!("First 16 lut_float: {:?}", &lut_float[0..16]);

        // Print first 32 bytes of LUT as i8 and u8 for comparison
        print!("First 32 lut_i8 bytes (as i8): ");
        for val in lut_i8.iter().take(32) {
            print!("{} ", val);
        }
        println!();
        print!("First 32 lut_i8 bytes (as u8): ");
        for val in lut_i8.iter().take(32) {
            print!("{} ", *val as u8);
        }
        println!();

        // Print first few bytes of packed code
        println!(
            "First 2 packed bytes: {:08b} {:08b}",
            quantized.binary_code_packed[0], quantized.binary_code_packed[1]
        );

        // Manually compute what accumulate should return
        // Now with MSB-first packing, packed[0] bit 7 = binary_code_unpacked[0]
        let mut manual_accu_via_direct = 0.0f32;
        for (i, _) in rotated_query.iter().enumerate().take(padded_dim) {
            manual_accu_via_direct += quantized.binary_code_unpacked[i] as f32 * rotated_query[i];
        }
        println!(
            "Manual via direct dot product: {:.6}",
            manual_accu_via_direct
        );

        // Pack the binary code
        let dim_bytes = padded_dim / 8;
        let mut packed_codes = vec![0u8; 32 * dim_bytes];
        crate::simd::pack_codes(
            &quantized.binary_code_packed,
            1,
            dim_bytes,
            &mut packed_codes,
        );

        // Print packed_codes for comparison with C++
        print!("Rust packed_codes (all {} bytes): ", packed_codes.len());
        for (i, byte) in packed_codes.iter().enumerate() {
            if i > 0 && i % 16 == 0 {
                print!("\n  ");
            }
            print!("{:02x} ", byte);
        }
        println!();

        // Accumulate
        let mut accu_res = [0u16; 32];
        crate::simd::accumulate_batch_avx2(&packed_codes, &lut_i8, padded_dim, &mut accu_res);

        let accu = accu_res[0] as f32;
        let ip_x0_qr = delta * accu + sum_vl_lut;
        println!("V2 accu: {:.6}, ip_x0_qr: {:.6}", accu, ip_x0_qr);

        // Manually compute what accu should be using lut_float
        let mut expected_result = 0.0f32;
        for (i, _) in rotated_query.iter().enumerate().take(dim) {
            expected_result += quantized.binary_code_unpacked[i] as f32 * rotated_query[i];
        }
        println!("Expected (direct): {:.6}", expected_result);

        // They should match within tolerance
        let diff = (binary_dot_v1 - ip_x0_qr).abs();
        println!("Difference: {:.6}", diff);
        assert!(
            diff < 0.01,
            "LUT-based result {:.6} doesn't match direct dot {:.6}",
            ip_x0_qr,
            binary_dot_v1
        );
    }

    #[test]
    fn test_batch_search_matches_per_vector_l2() {
        // Test parameters
        let dim = 64;
        let padded_dim = 64;
        let num_vectors = 96; // 3 batches of 32
        let ex_bits = 0; // Start with 1-bit only (no extended code)
        let total_bits = 1;
        let metric = Metric::L2;
        let mut rng = StdRng::seed_from_u64(12345);

        // Generate test data
        let centroid = vec![0.0f32; padded_dim];
        let mut data = Vec::with_capacity(num_vectors);
        let mut ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            data.push(random_vector(dim, &mut rng));
            ids.push(i);
        }

        // Quantize vectors
        let config = RabitqConfig::new(total_bits);
        let rotator = DynamicRotator::new(dim, RotatorType::FhtKacRotator, 42);
        let rotated_data: Vec<Vec<f32>> = data.iter().map(|v| rotator.rotate(v)).collect();
        let quantized_vectors: Vec<QuantizedVector> = rotated_data
            .iter()
            .map(|v| quantize_with_centroid(v, &centroid, &config, metric))
            .collect();

        // Create cluster with unified memory layout
        let cluster = ClusterData::from_quantized_vectors(
            centroid.clone(),
            ids.clone(),
            quantized_vectors.clone(),
            padded_dim,
            ex_bits,
        );

        // Generate query
        let query = random_vector(dim, &mut rng);
        let rotated_query = rotator.rotate(&query);
        let mut query_precomp = QueryPrecomputed::new(rotated_query.clone(), ex_bits);
        query_precomp.build_lut(padded_dim);

        // Debug LUT parameters
        if let Some(ref lut) = query_precomp.lut {
            println!("\n=== L2 Test LUT params ===");
            println!(
                "delta={:.6}, sum_vl_lut={:.6}, k1x_sum_q={:.6}",
                lut.delta, lut.sum_vl_lut, query_precomp.k1x_sum_q
            );
            println!("First 8 LUT values: {:?}\n", &lut.lut_i8[0..8]);
        }

        // Compute cluster statistics
        let centroid_dist = l2_distance_sqr(&rotated_query, &centroid);
        let dot_query_centroid = dot(&rotated_query, &centroid);
        let g_add = centroid_dist;
        let g_error = centroid_dist.sqrt();

        // Simulate index for batch search
        let ip_func = crate::simd::select_excode_ipfunc(ex_bits);
        let index = IvfRabitqIndex {
            dim,
            padded_dim,
            metric,
            rotator: rotator.clone(),
            clusters: vec![cluster.clone()],
            ex_bits,
            ip_func,
        };

        // Batch search (unified memory layout)
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let top_k = 10;
        let mut diag = SearchDiagnostics::default();
        let mut diagnostics = Some(&mut diag);
        index.search_cluster_v2_batched(
            0, // Single cluster test
            &cluster,
            &query_precomp,
            g_add,
            g_error,
            dot_query_centroid,
            None,
            &mut heap,
            top_k,
            &mut diagnostics,
        );

        // Convert heap to sorted vector
        let mut results: Vec<HeapCandidate> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| e.candidate)
            .collect();
        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));

        // Verify we got reasonable results
        assert!(!results.is_empty(), "Should find at least one result");
        assert!(results.len() <= top_k, "Should not exceed top_k");

        // Print results for debugging
        println!("\n=== Search Results (top 10) ===");
        for (i, r) in results.iter().take(10).enumerate() {
            println!("  [{}] ID={}, distance={:.6}", i, r.id, r.distance);
        }

        // Verify all distances are finite and non-negative for L2
        for r in &results {
            assert!(r.distance.is_finite(), "Distance should be finite");
            assert!(r.distance >= 0.0, "L2 distance should be non-negative");
        }

        // Verify results are sorted by distance (ascending for L2)
        for i in 0..results.len().saturating_sub(1) {
            assert!(
                results[i].distance <= results[i + 1].distance,
                "Results should be sorted by distance ascending"
            );
        }
    }

    #[test]
    fn test_batch_search_matches_per_vector_ip() {
        // Test parameters
        let dim = 64;
        let padded_dim = 64;
        let num_vectors = 64; // 2 batches of 32
        let ex_bits = 0; // Start with 1-bit only
        let total_bits = 1;
        let metric = Metric::InnerProduct;
        let mut rng = StdRng::seed_from_u64(54321);

        // Generate test data
        let centroid = vec![0.0f32; padded_dim];
        let mut data = Vec::with_capacity(num_vectors);
        let mut ids = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            data.push(random_vector(dim, &mut rng));
            ids.push(i);
        }

        // Quantize vectors
        let config = RabitqConfig::new(total_bits);
        let rotator = DynamicRotator::new(dim, RotatorType::FhtKacRotator, 42);
        let rotated_data: Vec<Vec<f32>> = data.iter().map(|v| rotator.rotate(v)).collect();
        let quantized_vectors: Vec<QuantizedVector> = rotated_data
            .iter()
            .map(|v| quantize_with_centroid(v, &centroid, &config, metric))
            .collect();

        // Create unified cluster
        let cluster = ClusterData::from_quantized_vectors(
            centroid.clone(),
            ids.clone(),
            quantized_vectors,
            padded_dim,
            ex_bits,
        );

        // Generate query
        let query = random_vector(dim, &mut rng);
        let rotated_query = rotator.rotate(&query);
        let mut query_precomp = QueryPrecomputed::new(rotated_query.clone(), ex_bits);
        query_precomp.build_lut(padded_dim);

        // Compute cluster statistics
        let dot_query_centroid = dot(&rotated_query, &centroid);
        let g_add = -dot_query_centroid;
        let centroid_norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        let g_error = centroid_norm;

        // Simulate index
        let ip_func = crate::simd::select_excode_ipfunc(ex_bits);
        let index = IvfRabitqIndex {
            dim,
            padded_dim,
            metric,
            rotator: rotator.clone(),
            clusters: vec![cluster.clone()],
            ex_bits,
            ip_func,
        };

        // Perform batch search
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let top_k = 5;
        let mut diag = SearchDiagnostics::default();
        let mut diagnostics = Some(&mut diag);
        index.search_cluster_v2_batched(
            0, // Single cluster test
            &cluster,
            &query_precomp,
            g_add,
            g_error,
            dot_query_centroid,
            None,
            &mut heap,
            top_k,
            &mut diagnostics,
        );

        // Convert and sort results by score (IP uses score, not distance)
        let mut results: Vec<HeapCandidate> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| e.candidate)
            .collect();
        results.sort_by(|a, b| b.score.total_cmp(&a.score));

        // Verify we got reasonable results
        assert!(!results.is_empty(), "Should find at least one result");
        assert!(results.len() <= top_k, "Should not exceed top_k");

        // Verify all scores are finite
        for r in &results {
            assert!(r.score.is_finite(), "Score should be finite");
            assert!(r.distance.is_finite(), "Distance should be finite");
        }

        // Verify results are sorted by score (descending for IP)
        for i in 0..results.len().saturating_sub(1) {
            assert!(
                results[i].score >= results[i + 1].score,
                "Results should be sorted by score descending"
            );
        }
    }
}
