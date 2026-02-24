use crate::fastscan::BatchData;
use crate::{quantizer, Metric, QuantizedVector, RabitqConfig};
use serde::{Deserialize, Serialize};

/// A posting list containing vectors assigned to a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingList {
    pub cluster_id: u32,
    pub centroid: Vec<f32>,
    pub size: u32,
    pub rabitq_config: RabitqConfig,

    /// Original IDs of vectors in this cluster
    pub ids: Vec<u64>,

    /// Source quantized vectors (dropped after build_batch_layout for memory efficiency)
    #[serde(skip)]
    pub vectors: Option<Vec<QuantizedVectorWithId>>,

    // FastScan batch layout
    pub batch_data: BatchData,
    pub ex_codes_packed: Vec<Vec<u8>>,
    pub f_add_ex: Vec<f32>,
    pub f_rescale_ex: Vec<f32>,
    pub padded_dim: usize,
}

/// Quantized vector with its original ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVectorWithId {
    pub vector_id: u64,
    pub quantized: QuantizedVector,
}

impl PostingList {
    /// Create a new empty posting list
    pub fn new(cluster_id: u32, centroid: Vec<f32>) -> Self {
        let padded_dim = centroid.len();
        Self {
            cluster_id,
            centroid,
            size: 0,
            rabitq_config: RabitqConfig::default(),
            ids: Vec::new(),
            vectors: Some(Vec::new()),
            batch_data: BatchData {
                data: Vec::new(),
                padded_dim,
            },
            ex_codes_packed: Vec::new(),
            f_add_ex: Vec::new(),
            f_rescale_ex: Vec::new(),
            padded_dim,
        }
    }

    /// Add a quantized vector to the posting list
    pub fn add_vector(&mut self, vector_id: u64, quantized: QuantizedVector) {
        if let Some(ref mut vectors) = self.vectors {
            vectors.push(QuantizedVectorWithId {
                vector_id,
                quantized,
            });
            self.size = vectors.len() as u32;
        }
    }

    /// Quantize and add vectors to this posting list
    pub fn quantize_vectors(
        &mut self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        rabitq_bits: usize,
        metric: Metric,
        faster_config: bool,
    ) -> Result<(), String> {
        if vectors.is_empty() {
            return Ok(());
        }

        assert_eq!(
            vectors.len(),
            vector_ids.len(),
            "vectors and vector_ids must have same length"
        );

        let dim = vectors[0].len();

        // Configure RaBitQ
        self.rabitq_config = if faster_config {
            RabitqConfig::faster(dim, rabitq_bits, 42)
        } else {
            RabitqConfig::new(rabitq_bits)
        };

        // Quantize each vector (as residual from centroid)
        for (i, vector) in vectors.iter().enumerate() {
            let quantized = quantizer::quantize_with_centroid(
                vector,
                &self.centroid,
                &self.rabitq_config,
                metric,
            );

            self.add_vector(vector_ids[i], quantized);
        }

        Ok(())
    }

    /// Get ex_bits for this posting list's RaBitQ config
    pub fn ex_bits(&self) -> u8 {
        self.rabitq_config.total_bits.saturating_sub(1) as u8
    }

    /// Estimate memory size in bytes
    pub fn memory_size(&self) -> usize {
        let vectors_heap: usize = self
            .vectors
            .as_ref()
            .map(|v| v.iter().map(|qv| qv.quantized.heap_size()).sum())
            .unwrap_or(0);
        let ex_codes_heap: usize = self.ex_codes_packed.iter().map(|v| v.capacity()).sum();

        std::mem::size_of::<Self>()
            + self.centroid.len() * std::mem::size_of::<f32>()
            + self.ids.capacity() * 8
            + self
                .vectors
                .as_ref()
                .map(|v| v.capacity() * std::mem::size_of::<QuantizedVectorWithId>())
                .unwrap_or(0)
            + vectors_heap
            + self.batch_data.data.capacity()
            + ex_codes_heap
            + self.f_add_ex.capacity() * 4
            + self.f_rescale_ex.capacity() * 4
    }

    /// Get the number of vectors in this posting list
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// Check if the posting list is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Build FastScan batch layout for efficient batch distance computation
    pub fn build_batch_layout(&mut self) {
        let vectors_opt = self.vectors.take();
        if let Some(vectors) = vectors_opt {
            if vectors.is_empty() {
                self.ids = Vec::new(); // Ensure ids is synced even if empty
                return;
            }

            // Sync IDs for search
            self.ids = vectors.iter().map(|v| v.vector_id).collect();

            // Extract quantized vectors
            let quantized_vecs: Vec<QuantizedVector> =
                vectors.into_iter().map(|v| v.quantized).collect();

            let ex_bits = self.rabitq_config.total_bits.saturating_sub(1);

            // Pack into batch data
            let (batch_data, ex_codes, f_add_ex_vals, f_rescale_ex_vals) =
                crate::fastscan::BatchData::pack_batch(&quantized_vecs, self.padded_dim, ex_bits);

            self.batch_data = batch_data;
            self.ex_codes_packed = ex_codes;
            self.f_add_ex = f_add_ex_vals;
            self.f_rescale_ex = f_rescale_ex_vals;

            // self.vectors is already None due to take()
        }
    }

    /// Get number of complete batches
    pub fn num_complete_batches(&self) -> usize {
        self.size as usize / crate::fastscan::BATCH_SIZE
    }

    /// Get number of remainder vectors
    pub fn num_remainder_vectors(&self) -> usize {
        self.size as usize % crate::fastscan::BATCH_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posting_list_creation() {
        let centroid = vec![1.0, 2.0, 3.0];
        let plist = PostingList::new(0, centroid.clone());

        assert_eq!(plist.cluster_id, 0);
        assert_eq!(plist.centroid, centroid);
        assert_eq!(plist.size, 0);
        assert!(plist.is_empty());
    }
}
