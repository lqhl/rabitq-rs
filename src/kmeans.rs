use kmeans::{AbortStrategy, EuclideanDistance, KMeans as ThirdPartyKMeans, KMeansConfig};
use rand::prelude::*;
use rand::RngCore;
use std::env;

const SUPPORTED_LANES: [usize; 4] = [16, 8, 4, 1];

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

/// Run k-means clustering via the third-party `kmeans` crate using k-means++ initialisation.
pub fn run_kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, rng: &mut StdRng) -> KMeansResult {
    let dim = validate_inputs(data, k, max_iter);
    let lane_count = detect_simd_lane_count();

    match lane_count {
        16 => run_kmeans_lane_16(data, k, max_iter, dim, rng),
        8 => run_kmeans_lane_8(data, k, max_iter, dim, rng),
        4 => run_kmeans_lane_4(data, k, max_iter, dim, rng),
        _ => run_kmeans_lane_1(data, k, max_iter, dim, rng),
    }
}

macro_rules! define_lane_runner {
    ($name:ident, $lanes:expr) => {
        fn $name(
            data: &[Vec<f32>],
            k: usize,
            max_iter: usize,
            dim: usize,
            rng: &mut StdRng,
        ) -> KMeansResult {
            let sample_cnt = data.len();
            let flattened = flatten_samples(data, sample_cnt * dim);

            let mut seed_bytes = [0u8; 32];
            rng.fill_bytes(&mut seed_bytes);
            let init_rng = StdRng::from_seed(seed_bytes);

            let config = KMeansConfig::build()
                .random_generator(init_rng)
                .abort_strategy(AbortStrategy::NoImprovement { threshold: 0.0 })
                .build();

            let kmeans = ThirdPartyKMeans::<f32, $lanes, _>::new(
                &flattened,
                sample_cnt,
                dim,
                EuclideanDistance,
            );

            let state =
                kmeans.kmeans_lloyd(k, max_iter, ThirdPartyKMeans::init_kmeanplusplus, &config);

            let centroid_values = state.centroids.to_vec();
            let assignments = state.assignments;

            let centroids = centroid_values
                .chunks_exact(dim)
                .map(|chunk| chunk.to_vec())
                .collect();

            KMeansResult {
                centroids,
                assignments,
            }
        }
    };
}

define_lane_runner!(run_kmeans_lane_16, 16);
define_lane_runner!(run_kmeans_lane_8, 8);
define_lane_runner!(run_kmeans_lane_4, 4);
define_lane_runner!(run_kmeans_lane_1, 1);

fn validate_inputs(data: &[Vec<f32>], k: usize, max_iter: usize) -> usize {
    assert!(!data.is_empty(), "k-means requires non-empty data");
    assert!(k > 0, "k must be positive");
    assert!(max_iter > 0, "max_iter must be positive");
    assert!(k <= data.len(), "k cannot exceed number of samples");

    let dim = data[0].len();
    assert!(
        data.iter().all(|v| v.len() == dim),
        "all vectors must share the same dimension",
    );
    dim
}

fn flatten_samples(data: &[Vec<f32>], capacity: usize) -> Vec<f32> {
    let mut flattened = Vec::with_capacity(capacity);
    for vector in data {
        flattened.extend_from_slice(vector);
    }
    flattened
}

fn detect_simd_lane_count() -> usize {
    if let Some(override_lanes) = lane_override_from_env() {
        return override_lanes;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return 16;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return 8;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return 4;
        }
        1
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return 4;
        }
        1
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        1
    }
}

fn lane_override_from_env() -> Option<usize> {
    match env::var("RABITQ_KMEANS_SIMD_LANES") {
        Ok(value) => parse_lane_override(value.trim()),
        Err(_) => None,
    }
}

fn parse_lane_override(value: &str) -> Option<usize> {
    let lanes = value.parse::<usize>().ok()?;
    if SUPPORTED_LANES.contains(&lanes) {
        Some(lanes)
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) fn run_kmeans_with_fixed_lanes<const LANES: usize>(
    data: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    rng: &mut StdRng,
) -> KMeansResult {
    let dim = validate_inputs(data, k, max_iter);
    match LANES {
        16 => run_kmeans_lane_16(data, k, max_iter, dim, rng),
        8 => run_kmeans_lane_8(data, k, max_iter, dim, rng),
        4 => run_kmeans_lane_4(data, k, max_iter, dim, rng),
        1 => run_kmeans_lane_1(data, k, max_iter, dim, rng),
        _ => panic!("unsupported SIMD lane count: {LANES}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataset() -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        for _ in 0..32 {
            data.push(vec![0.0, 0.0]);
            data.push(vec![10.0, 10.0]);
        }
        data
    }

    fn assert_centroids_close(expected: &[Vec<f32>], actual: &[Vec<f32>]) {
        assert_eq!(expected.len(), actual.len());
        for (lhs, rhs) in expected.iter().zip(actual.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (a, b) in lhs.iter().zip(rhs.iter()) {
                assert!((a - b).abs() < 1e-5, "centroids diverged: {a} vs {b}");
            }
        }
    }

    #[test]
    fn flattening_preserves_layout() {
        let data = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let flattened = flatten_samples(&data, 6);
        assert_eq!(flattened, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn env_override_applies_lane_choice() {
        env::set_var("RABITQ_KMEANS_SIMD_LANES", "1");
        let lanes = detect_simd_lane_count();
        env::remove_var("RABITQ_KMEANS_SIMD_LANES");
        assert_eq!(lanes, 1);
    }

    #[test]
    fn parse_rejects_invalid_lane_values() {
        assert_eq!(parse_lane_override("abc"), None);
        assert_eq!(parse_lane_override("3"), None);
        assert_eq!(parse_lane_override("8"), Some(8));
    }

    #[test]
    fn fixed_lane_variants_match_reference() {
        let data = sample_dataset();
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let reference = run_kmeans_with_fixed_lanes::<1>(&data, 2, 15, &mut rng);

        for &lanes in SUPPORTED_LANES.iter().filter(|&&lanes| lanes != 1) {
            let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
            let result = match lanes {
                4 => run_kmeans_with_fixed_lanes::<4>(&data, 2, 15, &mut rng),
                8 => run_kmeans_with_fixed_lanes::<8>(&data, 2, 15, &mut rng),
                16 => run_kmeans_with_fixed_lanes::<16>(&data, 2, 15, &mut rng),
                _ => unreachable!(),
            };
            assert_eq!(reference.assignments, result.assignments);
            assert_centroids_close(&reference.centroids, &result.centroids);
        }
    }

    #[test]
    fn override_aligns_dispatch_with_reference() {
        let data = sample_dataset();
        env::set_var("RABITQ_KMEANS_SIMD_LANES", "4");
        let mut rng = StdRng::seed_from_u64(0xFEEDFACE);
        let dispatched = run_kmeans(&data, 2, 15, &mut rng);
        env::remove_var("RABITQ_KMEANS_SIMD_LANES");

        let mut rng = StdRng::seed_from_u64(0xFEEDFACE);
        let expected = run_kmeans_with_fixed_lanes::<4>(&data, 2, 15, &mut rng);

        assert_eq!(expected.assignments, dispatched.assignments);
        assert_centroids_close(&expected.centroids, &dispatched.centroids);
    }
}
