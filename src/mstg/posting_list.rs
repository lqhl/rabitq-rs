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
    pub vectors: Vec<QuantizedVectorWithId>,

    // FastScan batch layout (always built)
    #[serde(skip)]
    pub batch_data: BatchData,
    #[serde(skip)]
    pub ex_codes_packed: Vec<Vec<u8>>,
    #[serde(skip)]
    pub f_add_ex: Vec<f32>,
    #[serde(skip)]
    pub f_rescale_ex: Vec<f32>,
    #[serde(skip)]
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
            vectors: Vec::new(),
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
        self.vectors.push(QuantizedVectorWithId {
            vector_id,
            quantized,
        });
        self.size = self.vectors.len() as u32;
    }

    /// Quantize and add vectors to this posting list
    ///
    /// This trains RaBitQ on the residuals (vectors - centroid) and quantizes all vectors
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

    /// Quantize and add vectors by indices into the shared dataset
    ///
    /// This avoids cloning the full vectors during index build.
    pub fn quantize_vectors_by_ids(
        &mut self,
        data: &[Vec<f32>],
        vector_ids: &[usize],
        rabitq_bits: usize,
        metric: Metric,
        faster_config: bool,
    ) -> Result<(), String> {
        if vector_ids.is_empty() {
            return Ok(());
        }

        let dim = data[vector_ids[0]].len();

        // Configure RaBitQ
        self.rabitq_config = if faster_config {
            RabitqConfig::faster(dim, rabitq_bits, 42)
        } else {
            RabitqConfig::new(rabitq_bits)
        };

        // Quantize each vector (as residual from centroid)
        for &vec_id in vector_ids {
            let vector = &data[vec_id];
            let quantized = quantizer::quantize_with_centroid(
                vector,
                &self.centroid,
                &self.rabitq_config,
                metric,
            );

            self.add_vector(vec_id as u64, quantized);
        }

        Ok(())
    }

    /// Get ex_bits for this posting list's RaBitQ config
    ///
    /// Returns the number of extended bits used in quantization
    pub fn ex_bits(&self) -> u8 {
        self.rabitq_config.total_bits.saturating_sub(1) as u8
    }

    /// Estimate memory size in bytes
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.centroid.len() * std::mem::size_of::<f32>()
            + self.vectors.capacity() * std::mem::size_of::<QuantizedVectorWithId>()
    }

    /// Get the number of vectors in this posting list
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the posting list is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Build FastScan batch layout for efficient batch distance computation
    /// This must be called after quantization is complete, before search
    pub fn build_batch_layout(&mut self) {
        if self.vectors.is_empty() {
            return;
        }

        // Extract quantized vectors
        let quantized_vecs: Vec<QuantizedVector> =
            self.vectors.iter().map(|v| v.quantized.clone()).collect();

        let ex_bits = self.rabitq_config.total_bits.saturating_sub(1);

        // Pack into batch data
        let (batch_data, ex_codes, f_add_ex_vals, f_rescale_ex_vals) =
            BatchData::pack_batch(&quantized_vecs, self.padded_dim, ex_bits);

        self.batch_data = batch_data;
        self.ex_codes_packed = ex_codes;
        self.f_add_ex = f_add_ex_vals;
        self.f_rescale_ex = f_rescale_ex_vals;
    }

    /// Get number of complete batches
    pub fn num_complete_batches(&self) -> usize {
        self.vectors.len() / crate::fastscan::BATCH_SIZE
    }

    /// Get number of remainder vectors
    pub fn num_remainder_vectors(&self) -> usize {
        self.vectors.len() % crate::fastscan::BATCH_SIZE
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
