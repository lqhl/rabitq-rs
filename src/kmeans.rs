use kmeans::{AbortStrategy, EuclideanDistance, KMeans as ThirdPartyKMeans, KMeansConfig};
use rand::prelude::*;
use rand::RngCore;

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

const SIMD_LANES: usize = 16;

/// Run k-means clustering via the third-party `kmeans` crate using k-means++ initialisation.
pub fn run_kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, rng: &mut StdRng) -> KMeansResult {
    assert!(!data.is_empty(), "k-means requires non-empty data");
    assert!(k > 0, "k must be positive");
    assert!(max_iter > 0, "max_iter must be positive");
    assert!(k <= data.len(), "k cannot exceed number of samples");

    let dim = data[0].len();
    assert!(
        data.iter().all(|v| v.len() == dim),
        "all vectors must share the same dimension"
    );

    let sample_cnt = data.len();

    let mut flattened = Vec::with_capacity(sample_cnt * dim);
    for vector in data {
        flattened.extend_from_slice(vector);
    }

    let mut seed_bytes = [0u8; 32];
    rng.fill_bytes(&mut seed_bytes);
    let init_rng = StdRng::from_seed(seed_bytes);

    let config = KMeansConfig::build()
        .random_generator(init_rng)
        .abort_strategy(AbortStrategy::NoImprovement { threshold: 0.0 })
        .build();

    let kmeans =
        ThirdPartyKMeans::<f32, SIMD_LANES, _>::new(&flattened, sample_cnt, dim, EuclideanDistance);

    let state = kmeans.kmeans_lloyd(k, max_iter, ThirdPartyKMeans::init_kmeanplusplus, &config);

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
