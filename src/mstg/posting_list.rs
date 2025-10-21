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
        Self {
            cluster_id,
            centroid,
            size: 0,
            rabitq_config: RabitqConfig::default(),
            vectors: Vec::new(),
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

    /// Estimate distance from query to a quantized vector in this posting list
    ///
    /// This computes the approximate L2 distance using the quantized representation
    pub fn estimate_distance(&self, query: &[f32], vec_idx: usize) -> Option<f32> {
        use crate::math::l2_distance_sqr;

        if vec_idx >= self.vectors.len() {
            return None;
        }

        let qvec = &self.vectors[vec_idx];

        // Compute centroid distance component (g_add)
        let g_add = l2_distance_sqr(query, &self.centroid);

        // Unpack binary code and compute centered version (bit - 0.5)
        let binary_code = qvec.quantized.unpack_binary_code();

        // Compute dot product with query using centered binary code
        let mut binary_dot = 0.0f32;
        for (&bit, &q_val) in binary_code.iter().zip(query.iter()) {
            binary_dot += (bit as f32) * q_val;
        }

        // Compute sum of query values for the offset term
        let sum_query: f32 = query.iter().sum();
        let c1 = -0.5f32;
        let k1x_sum_q = c1 * sum_query;

        let binary_term = binary_dot + k1x_sum_q;

        // Basic estimate: f_add + g_add + f_rescale * binary_term
        let est_distance = qvec.quantized.f_add + g_add + qvec.quantized.f_rescale * binary_term;

        // Use extended code if available for better accuracy
        let distance = if qvec.quantized.ex_bits > 0 {
            let ex_code = qvec.quantized.unpack_ex_code();
            let mut ex_dot = 0.0f32;
            for (&code, &q_val) in ex_code.iter().zip(query.iter()) {
                ex_dot += (code as f32) * q_val;
            }

            let binary_scale = (1 << qvec.quantized.ex_bits) as f32;
            let cb = -((1 << qvec.quantized.ex_bits) as f32 - 0.5);
            let kbx_sum_q = cb * sum_query;
            let total_term = binary_scale * binary_dot + ex_dot + kbx_sum_q;

            qvec.quantized.f_add_ex + g_add + qvec.quantized.f_rescale_ex * total_term
        } else {
            est_distance
        };

        Some(distance)
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
