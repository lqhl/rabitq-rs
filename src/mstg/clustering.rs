//! Hierarchical balanced clustering for MSTG
//!
//! Implements balanced k-means to ensure cluster sizes don't exceed limits

use crate::kmeans::run_kmeans;
use crate::math::l2_distance_sqr;
use rand::prelude::*;

/// A cluster of vectors with its centroid
#[derive(Debug, Clone)]
pub struct Cluster {
    indices: Vec<usize>,
    centroid: Vec<f32>,
}

impl Cluster {
    /// Create a cluster from vector indices
    pub fn from_indices(indices: Vec<usize>, data: &[Vec<f32>]) -> Self {
        let centroid = compute_centroid(data, &indices);
        Self { indices, centroid }
    }

    /// Get reference to the indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get reference to the centroid
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Get the number of vectors in this cluster
    pub fn size(&self) -> usize {
        self.indices.len()
    }
}

/// Hierarchical clustering with balanced cluster sizes
pub struct HierarchicalClustering {
    pub max_cluster_size: usize,
    pub branching_factor: usize,
    pub balance_weight: f32,
    pub max_iterations: usize,
}

impl HierarchicalClustering {
    pub fn new(max_cluster_size: usize, branching_factor: usize, balance_weight: f32) -> Self {
        Self {
            max_cluster_size,
            branching_factor,
            balance_weight,
            max_iterations: 100,
        }
    }

    /// Cluster data into balanced clusters not exceeding max_cluster_size
    pub fn cluster(&self, data: &[Vec<f32>]) -> Vec<Cluster> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut rng = StdRng::seed_from_u64(42);
        let root_indices: Vec<usize> = (0..data.len()).collect();
        let mut active_clusters = vec![Cluster::from_indices(root_indices, data)];
        let mut final_clusters = Vec::new();

        println!(
            "Starting hierarchical clustering: {} vectors, max_size={}, branching={}",
            data.len(),
            self.max_cluster_size,
            self.branching_factor
        );

        let mut iteration = 0;
        while let Some(cluster) = active_clusters.pop() {
            if cluster.size() <= self.max_cluster_size {
                final_clusters.push(cluster);
            } else {
                iteration += 1;
                println!(
                    "  Iteration {}: Splitting cluster of size {} into {} subclusters",
                    iteration,
                    cluster.size(),
                    self.branching_factor
                );

                // Split using k-means
                let subclusters = self.split_cluster(&cluster, data, &mut rng);

                // Add subclusters to active queue
                for subcluster in subclusters {
                    active_clusters.push(subcluster);
                }
            }
        }

        println!(
            "Hierarchical clustering complete: {} final clusters",
            final_clusters.len()
        );

        final_clusters
    }

    /// Split a cluster into k subclusters using k-means
    fn split_cluster(
        &self,
        cluster: &Cluster,
        data: &[Vec<f32>],
        rng: &mut StdRng,
    ) -> Vec<Cluster> {
        let k = self.branching_factor;
        let cluster_data = collect_vectors(data, cluster.indices());

        // Run k-means on this cluster
        let kmeans_result = run_kmeans(&cluster_data, k, self.max_iterations, rng);

        // Group vectors by assignment
        let mut subcluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (vec_idx, &cluster_id) in kmeans_result.assignments.iter().enumerate() {
            let global_idx = cluster.indices()[vec_idx];
            subcluster_indices[cluster_id].push(global_idx);
        }

        // Apply balance constraint if needed
        if self.balance_weight > 0.0 {
            subcluster_indices =
                self.balance_clusters(subcluster_indices, data, &kmeans_result.centroids);
        }

        // Create clusters from grouped data
        subcluster_indices
            .into_iter()
            .filter(|indices| !indices.is_empty())
            .map(|indices| Cluster::from_indices(indices, data))
            .collect()
    }

    /// Balance cluster sizes by moving vectors between clusters
    fn balance_clusters(
        &self,
        mut clusters: Vec<Vec<usize>>,
        data: &[Vec<f32>],
        centroids: &[Vec<f32>],
    ) -> Vec<Vec<usize>> {
        let total_size: usize = clusters.iter().map(|c| c.len()).sum();
        let k = clusters.len();
        let target_size = total_size / k;
        let max_allowed = (target_size as f32 * (1.0 + self.balance_weight)) as usize;

        // Move vectors from over-sized clusters to under-sized ones
        for _iteration in 0..10 {
            // Find over-sized and under-sized clusters
            let sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();

            let mut over_idx = None;
            let mut under_idx = None;

            for (i, &size) in sizes.iter().enumerate() {
                if size > max_allowed {
                    over_idx = Some(i);
                    break;
                }
            }

            for (i, &size) in sizes.iter().enumerate() {
                if size < target_size {
                    under_idx = Some(i);
                    break;
                }
            }

            if over_idx.is_none() || under_idx.is_none() {
                break; // Balanced
            }

            let over_i = over_idx.unwrap();
            let under_i = under_idx.unwrap();

            // Find vector in over_i that's closest to under_i's centroid
            if let Some(closest_idx) = self.find_closest_vector_to_centroid(
                &clusters[over_i],
                data,
                &centroids[under_i],
            ) {
                let vec_id = clusters[over_i].remove(closest_idx);
                clusters[under_i].push(vec_id);
            } else {
                break;
            }
        }

        clusters
    }

    /// Find the index of the vector closest to the given centroid
    fn find_closest_vector_to_centroid(
        &self,
        vector_indices: &[usize],
        data: &[Vec<f32>],
        centroid: &[f32],
    ) -> Option<usize> {
        if vector_indices.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = l2_distance_sqr(&data[vector_indices[0]], centroid);

        for (idx, &vec_id) in vector_indices.iter().enumerate().skip(1) {
            let dist = l2_distance_sqr(&data[vec_id], centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        Some(best_idx)
    }
}

/// Compute the centroid of a set of vectors
fn compute_centroid(data: &[Vec<f32>], indices: &[usize]) -> Vec<f32> {
    if indices.is_empty() {
        return Vec::new();
    }

    let dim = data[indices[0]].len();
    let mut centroid = vec![0.0; dim];

    for &idx in indices {
        let vec = &data[idx];
        for (i, &val) in vec.iter().enumerate() {
            centroid[i] += val;
        }
    }

    let n = indices.len() as f32;
    for val in &mut centroid {
        *val /= n;
    }

    centroid
}

fn collect_vectors(data: &[Vec<f32>], indices: &[usize]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        out.push(data[idx].clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(12345);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen()).collect())
            .collect()
    }

    #[test]
    fn test_compute_centroid() {
        let data = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![2.0, 2.0],
        ];

        let indices: Vec<usize> = (0..data.len()).collect();
        let centroid = compute_centroid(&data, &indices);
        assert_eq!(centroid.len(), 2);
        assert!((centroid[0] - 1.0).abs() < 1e-5);
        assert!((centroid[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cluster_creation() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let indices: Vec<usize> = (0..data.len()).collect();
        let cluster = Cluster::from_indices(indices, &data);

        assert_eq!(cluster.size(), 2);
        assert_eq!(cluster.indices().len(), 2);
        assert_eq!(cluster.centroid().len(), 2);
        assert!((cluster.centroid()[0] - 2.0).abs() < 1e-5);
        assert!((cluster.centroid()[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_hierarchical_clustering_basic() {
        let data = generate_test_data(100, 8);

        let clustering = HierarchicalClustering::new(20, 4, 1.0);
        let clusters = clustering.cluster(&data);

        // Verify all clusters are within size limit
        for cluster in &clusters {
            assert!(
                cluster.size() <= 20,
                "Cluster size {} exceeds limit 20",
                cluster.size()
            );
        }

        // Verify all data is accounted for
        let total_size: usize = clusters.iter().map(|c| c.size()).sum();
        assert_eq!(total_size, 100);
    }

    #[test]
    fn test_hierarchical_clustering_balance() {
        let data = generate_test_data(1000, 32);

        let clustering = HierarchicalClustering::new(100, 5, 1.0);
        let clusters = clustering.cluster(&data);

        // Verify balance
        let sizes: Vec<usize> = clusters.iter().map(|c| c.size()).collect();
        let mean = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;

        let variance: f32 = sizes
            .iter()
            .map(|&s| {
                let diff = s as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / sizes.len() as f32;

        let coefficient_of_variation = variance.sqrt() / mean;
        println!(
            "Mean cluster size: {:.1}, CoV: {:.3}",
            mean, coefficient_of_variation
        );

        // Hierarchical clustering produces reasonably balanced clusters
        // CoV < 0.6 is acceptable for small subclusters
        assert!(coefficient_of_variation < 0.6);

        // Ensure no cluster is empty
        for cluster in &clusters {
            assert!(cluster.size() > 0, "Empty cluster found");
        }
    }
}
