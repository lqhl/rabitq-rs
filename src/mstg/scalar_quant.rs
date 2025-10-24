use serde::{Deserialize, Serialize};

/// Trait for scalar-quantized vectors
pub trait QuantizedVector: Clone + Send + Sync + Sized {
    fn quantize(vector: &[f32]) -> Self;
    fn dequantize(&self) -> Vec<f32>;
    fn distance_to(&self, other: &Self) -> f32;
    fn memory_size(&self) -> usize;
}

/// BF16 (Brain Float16) quantized vector
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BF16Vector {
    data: Vec<u16>,
}

impl QuantizedVector for BF16Vector {
    fn quantize(vector: &[f32]) -> Self {
        let data = vector.iter().map(|&x| fp32_to_bf16(x)).collect();
        Self { data }
    }

    fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&x| bf16_to_fp32(x)).collect()
    }

    fn distance_to(&self, other: &Self) -> f32 {
        let mut sum = 0.0f32;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = bf16_to_fp32(*a) - bf16_to_fp32(*b);
            sum += diff * diff;
        }
        sum
    }

    fn memory_size(&self) -> usize {
        self.data.len() * 2
    }
}

/// FP32 (no quantization)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FP32Vector {
    data: Vec<f32>,
}

impl QuantizedVector for FP32Vector {
    fn quantize(vector: &[f32]) -> Self {
        Self {
            data: vector.to_vec(),
        }
    }

    fn dequantize(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn distance_to(&self, other: &Self) -> f32 {
        crate::math::l2_distance_sqr(&self.data, &other.data)
    }

    fn memory_size(&self) -> usize {
        self.data.len() * 4
    }
}

/// Convert FP32 to BF16 with rounding
#[inline]
pub fn fp32_to_bf16(x: f32) -> u16 {
    // BF16: sign(1) + exponent(8) + mantissa(7)
    // Truncate fp32 mantissa from 23 to 7 bits with rounding
    let bits = x.to_bits();

    // Round to nearest even (add rounding bias before truncating)
    let rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);

    (rounded >> 16) as u16
}

/// Convert BF16 to FP32
#[inline]
pub fn bf16_to_fp32(x: u16) -> f32 {
    // Expand bf16 back to fp32 by shifting and zero-padding mantissa
    f32::from_bits((x as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        let values = vec![1.0, -1.0, 0.5, 100.0, 0.001, std::f32::consts::PI];

        for &val in &values {
            let bf16 = fp32_to_bf16(val);
            let recovered = bf16_to_fp32(bf16);
            let error = (val - recovered).abs() / val.abs().max(1e-6);
            println!("{} -> bf16 -> {} (error: {:.6})", val, recovered, error);
            assert!(error < 0.01); // <1% error
        }
    }

    #[test]
    fn test_bf16_vector_distance() {
        // Use larger vectors with more substantial distances for better testing
        let v1 = vec![1.0; 128];
        let v2 = vec![2.0; 128];

        let qv1 = BF16Vector::quantize(&v1);
        let qv2 = BF16Vector::quantize(&v2);

        let exact_dist = crate::math::l2_distance_sqr(&v1, &v2);
        let approx_dist = qv1.distance_to(&qv2);

        let error = (exact_dist - approx_dist).abs() / exact_dist;
        println!(
            "Exact: {}, Approx: {}, Error: {:.2}%",
            exact_dist,
            approx_dist,
            error * 100.0
        );
        // BF16 maintains high accuracy for larger distances
        // For GIST-scale vectors, error should be <1%
        assert!(error < 0.01);
    }

    #[test]
    fn test_bf16_memory_savings() {
        let vector = vec![0.5; 960]; // GIST dimensionality

        let bf16 = BF16Vector::quantize(&vector);
        let fp32 = FP32Vector::quantize(&vector);

        assert_eq!(bf16.memory_size(), 960 * 2);
        assert_eq!(fp32.memory_size(), 960 * 4);
        assert_eq!(bf16.memory_size(), fp32.memory_size() / 2);
    }
}
