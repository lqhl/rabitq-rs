/// Compute the dot product between two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the squared L2 norm of a vector.
pub fn l2_norm_sqr(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

/// Compute the squared Euclidean distance between two vectors.
pub fn l2_distance_sqr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Normalize a vector in-place. Returns the original norm.
pub fn normalize(v: &mut [f32]) -> f32 {
    let norm = l2_norm_sqr(v).sqrt();
    if norm <= f32::EPSILON {
        return 0.0;
    }
    for value in v.iter_mut() {
        *value /= norm;
    }
    norm
}

/// Compute `a - b` element-wise.
pub fn subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
