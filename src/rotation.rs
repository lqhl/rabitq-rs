use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use crate::math::{dot, normalize};

/// Random orthonormal rotator implemented via Gram-Schmidt orthogonalisation.
#[derive(Debug, Clone)]
pub struct RandomRotator {
    dim: usize,
    matrix: Vec<f32>, // Row-major storage
}

impl RandomRotator {
    /// Create a new random rotator with the provided seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_rng(dim, &mut rng)
    }

    fn with_rng(dim: usize, rng: &mut StdRng) -> Self {
        let normal = Normal::new(0.0, 1.0).expect("failed to create normal distribution");
        let mut basis: Vec<Vec<f32>> = Vec::with_capacity(dim);

        for _ in 0..dim {
            let mut vec: Vec<f32> = (0..dim).map(|_| normal.sample(rng) as f32).collect();

            // Orthogonalise against previous basis vectors.
            for prev in &basis {
                let proj = dot(&vec, prev);
                for (v, p) in vec.iter_mut().zip(prev.iter()) {
                    *v -= proj * *p;
                }
            }

            let mut attempts = 0;
            loop {
                let norm = normalize(&mut vec);
                if norm > f32::EPSILON {
                    break;
                }
                attempts += 1;
                if attempts > 8 {
                    // Fallback to a canonical basis vector.
                    vec.fill(0.0);
                    vec[attempts % dim] = 1.0;
                    break;
                }
                for value in vec.iter_mut() {
                    *value = normal.sample(rng) as f32;
                }
                for prev in &basis {
                    let proj = dot(&vec, prev);
                    for (v, p) in vec.iter_mut().zip(prev.iter()) {
                        *v -= proj * *p;
                    }
                }
            }

            basis.push(vec);
        }

        let mut matrix = Vec::with_capacity(dim * dim);
        for row in basis {
            matrix.extend_from_slice(&row);
        }

        Self { dim, matrix }
    }

    /// Apply the rotation to a vector, returning the rotated output.
    pub fn rotate(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.dim);
        let mut output = vec![0.0f32; self.dim];
        self.rotate_into(input, &mut output);
        output
    }

    /// Apply the rotation into an existing buffer.
    pub fn rotate_into(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.dim);
        assert_eq!(output.len(), self.dim);

        for (row_idx, chunk) in self.matrix.chunks(self.dim).enumerate() {
            let mut acc = 0.0f32;
            for (value, &weight) in input.iter().zip(chunk.iter()) {
                acc += value * weight;
            }
            output[row_idx] = acc;
        }
    }
}
