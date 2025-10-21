//! Main MSTG index structure

use super::*;
use rayon::prelude::*;

/// Main MSTG (Multi-Scale Tree Graph) index
pub struct MstgIndex {
    pub config: MstgConfig,
    pub centroid_index: CentroidIndex,
    pub posting_lists: Vec<PostingList>,
    pub directory: PostingListDirectory,
}

impl MstgIndex {
    /// Build an MSTG index from data
    pub fn build(data: &[Vec<f32>], config: MstgConfig) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot build index from empty data".to_string());
        }

        println!("Building MSTG index for {} vectors", data.len());

        // Step 1: Hierarchical balanced clustering
        println!("Step 1: Hierarchical balanced clustering...");
        let clustering = HierarchicalClustering::new(
            config.max_posting_size,
            config.branching_factor,
            config.balance_weight,
        );
        let clusters = clustering.cluster(data);
        println!("  Created {} clusters", clusters.len());

        // Step 2: Closure assignment
        println!("Step 2: Closure assignment with RNG rule...");
        let centroids: Vec<Vec<f32>> = clusters.iter().map(|c| c.centroid().to_vec()).collect();

        let assigner = ClosureAssigner::new(config.closure_epsilon, config.max_replicas);

        let cluster_assignments: Vec<Vec<usize>> = data
            .par_iter()
            .map(|v| assigner.assign(v, &centroids))
            .collect();

        // Count total assignments for statistics
        let total_assignments: usize = cluster_assignments.iter().map(|a| a.len()).sum();
        let replication_factor = total_assignments as f32 / data.len() as f32;
        println!("  Average replication factor: {:.2}", replication_factor);

        // Step 3: Create posting lists with RaBitQ quantization
        println!("Step 3: Creating and quantizing posting lists...");
        let posting_lists: Vec<PostingList> = clusters
            .par_iter()
            .enumerate()
            .map(|(cluster_id, cluster)| {
                let mut plist = PostingList::new(cluster_id as u32, cluster.centroid().to_vec());

                // Collect vectors assigned to this cluster
                let mut assigned_vectors = Vec::new();
                let mut assigned_ids = Vec::new();

                for (vec_id, assignments) in cluster_assignments.iter().enumerate() {
                    if assignments.contains(&cluster_id) {
                        assigned_vectors.push(data[vec_id].clone());
                        assigned_ids.push(vec_id as u64);
                    }
                }

                // Quantize if there are vectors
                if !assigned_vectors.is_empty() {
                    plist
                        .quantize_vectors(
                            &assigned_vectors,
                            &assigned_ids,
                            config.rabitq_bits,
                            config.metric,
                            config.faster_config,
                        )
                        .unwrap_or_else(|e| {
                            eprintln!(
                                "Warning: Quantization failed for cluster {}: {}",
                                cluster_id, e
                            );
                        });
                }

                plist
            })
            .collect();

        println!(
            "  Created {} posting lists with {} total vectors",
            posting_lists.len(),
            posting_lists.iter().map(|p| p.len()).sum::<usize>()
        );

        // Step 4: Build centroid index
        println!("Step 4: Building centroid index...");
        let centroid_reps: Vec<Vec<f32>> =
            posting_lists.iter().map(|p| p.centroid.clone()).collect();
        let centroid_ids: Vec<u32> = (0..posting_lists.len() as u32).collect();

        let centroid_index =
            CentroidIndex::build(centroid_reps, centroid_ids, config.centroid_precision);

        println!(
            "  Built centroid index with {} centroids ({} precision)",
            centroid_index.len(),
            match config.centroid_precision {
                ScalarPrecision::FP32 => "FP32",
                ScalarPrecision::BF16 => "BF16",
                _ => "Other",
            }
        );

        // Step 5: Create metadata directory
        let directory = PostingListDirectory::new();

        println!("MSTG index build complete");
        println!(
            "  Memory usage estimate: ~{:.2} MB",
            Self::estimate_memory_mb(&centroid_index, &posting_lists)
        );

        Ok(Self {
            config,
            centroid_index,
            posting_lists,
            directory,
        })
    }

    /// Estimate total memory usage in MB
    fn estimate_memory_mb(centroid_index: &CentroidIndex, posting_lists: &[PostingList]) -> f32 {
        let centroid_mem = centroid_index.memory_usage();
        let posting_mem: usize = posting_lists.iter().map(|p| p.memory_size()).sum();
        ((centroid_mem + posting_mem) as f32) / (1024.0 * 1024.0)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], params: &SearchParams) -> Vec<SearchResult> {
        // Step 1: Find candidate centroids
        let centroid_candidates = self.centroid_index.search(query, params.ef_search);

        // Step 2: Dynamic pruning
        let selected_centroids = self.dynamic_prune(&centroid_candidates, params.pruning_epsilon);

        // Step 3: Search posting lists
        let mut all_candidates: Vec<(u64, f32)> = selected_centroids
            .par_iter()
            .flat_map(|&cid| {
                let plist = &self.posting_lists[cid as usize];
                plist
                    .vectors
                    .iter()
                    .enumerate()
                    .filter_map(|(vec_idx, qvec)| {
                        plist
                            .estimate_distance(query, vec_idx)
                            .map(|dist| (qvec.vector_id, dist))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Step 4: Sort and return top-k
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_candidates.truncate(params.top_k);

        all_candidates
            .into_iter()
            .map(|(id, dist)| SearchResult {
                vector_id: id as usize,
                distance: dist,
            })
            .collect()
    }

    /// Apply dynamic pruning to centroid candidates
    fn dynamic_prune(&self, candidates: &[(u32, f32)], epsilon: f32) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let closest_dist = candidates[0].1;
        let threshold = closest_dist * (1.0 + epsilon);

        candidates
            .iter()
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub vector_id: usize,
    pub distance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(42);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen()).collect())
            .collect()
    }

    #[test]
    fn test_mstg_index_build_small() {
        let data = generate_test_data(100, 16);
        let config = MstgConfig {
            max_posting_size: 20,
            branching_factor: 4,
            ..Default::default()
        };

        let index = MstgIndex::build(&data, config).unwrap();

        assert!(!index.posting_lists.is_empty());
        assert!(index.centroid_index.len() > 0);
    }

    #[test]
    fn test_mstg_search() {
        let data = generate_test_data(100, 16);
        let config = MstgConfig {
            max_posting_size: 25,
            branching_factor: 4,
            ..Default::default()
        };

        let index = MstgIndex::build(&data, config).unwrap();

        let query = &data[0];
        let params = SearchParams::balanced(10);
        let results = index.search(query, &params);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // First result should be the query itself (or very close)
        assert!(results[0].distance < 0.1);
    }
}
