/// Compute the dot product between two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Compute the squared L2 norm of a vector.
pub fn l2_norm_sqr(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for value in v.iter() {
        sum += value * value;
    }
    sum
}

/// Compute the squared Euclidean distance between two vectors.
pub fn l2_distance_sqr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut i = 0usize;
    let len = a.len();
    while i + 4 <= len {
        let dx0 = a[i] - b[i];
        let dx1 = a[i + 1] - b[i + 1];
        let dx2 = a[i + 2] - b[i + 2];
        let dx3 = a[i + 3] - b[i + 3];
        sum0 += dx0 * dx0 + dx1 * dx1;
        sum1 += dx2 * dx2 + dx3 * dx3;
        i += 4;
    }
    while i < len {
        let diff = a[i] - b[i];
        sum0 += diff * diff;
        i += 1;
    }
    sum0 + sum1
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
