//! Index building pipeline
//!
//! Orchestrates hierarchical clustering, closure assignment, and index serialization.

use super::*;
use rayon::prelude::*;

pub struct MstgBuilder<'a> {
    config: MstgConfig,
    data: &'a [Vec<f32>],
}

impl<'a> MstgBuilder<'a> {
    pub fn new(data: &'a [Vec<f32>], config: MstgConfig) -> Self {
        Self { config, data }
    }

    pub fn build(self) -> Result<MstgIndex, String> {
        if self.data.is_empty() {
            return Err("Cannot build index from empty data".to_string());
        }

        println!("Building MSTG index for {} vectors", self.data.len());

        // Step 1: Hierarchical balanced clustering
        println!("Step 1: Hierarchical balanced clustering...");
        let clustering = HierarchicalClustering::new(
            self.config.max_posting_size,
            self.config.branching_factor,
            self.config.balance_weight,
        );
        let clusters = clustering.cluster(self.data);
        println!("  Created {} clusters", clusters.len());

        // Step 2: Closure assignment
        println!("Step 2: Closure assignment with RNG rule...");
        let centroids: Vec<Vec<f32>> = clusters.iter().map(|c| c.centroid().to_vec()).collect();

        let assigner = ClosureAssigner::new(self.config.closure_epsilon, self.config.max_replicas);

        let cluster_assignments: Vec<Vec<usize>> = self
            .data
            .par_iter()
            .map(|v| assigner.assign(v, &centroids))
            .collect();

        // Count total assignments for statistics
        let total_assignments: usize = cluster_assignments.iter().map(|a| a.len()).sum();
        let replication_factor = total_assignments as f32 / self.data.len() as f32;
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
                        assigned_vectors.push(self.data[vec_id].clone());
                        assigned_ids.push(vec_id as u64);
                    }
                }

                // Quantize if there are vectors
                if !assigned_vectors.is_empty() {
                    plist
                        .quantize_vectors(
                            &assigned_vectors,
                            &assigned_ids,
                            self.config.rabitq_bits,
                            self.config.metric,
                            self.config.faster_config,
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

        // Build FastScan batch layout for each posting list
        println!("Step 3.5: Building FastScan batch layouts...");
        let posting_lists: Vec<PostingList> = posting_lists
            .into_par_iter()
            .map(|mut plist| {
                plist.build_batch_layout();
                plist
            })
            .collect();

        // Step 4: Build centroid index
        println!("Step 4: Building centroid index...");
        let centroid_reps: Vec<Vec<f32>> =
            posting_lists.iter().map(|p| p.centroid.clone()).collect();
        let centroid_ids: Vec<u32> = (0..posting_lists.len() as u32).collect();

        let centroid_index =
            CentroidIndex::build(centroid_reps, centroid_ids, self.config.centroid_precision);

        println!(
            "  Built centroid index with {} centroids ({} precision)",
            centroid_index.len(),
            match self.config.centroid_precision {
                ScalarPrecision::FP32 => "FP32",
                ScalarPrecision::BF16 => "BF16",
                ScalarPrecision::FP16 => "FP16",
                ScalarPrecision::INT8 => "INT8",
            }
        );

        // Step 5: Create metadata directory
        let mut directory = PostingListDirectory::new();
        let mut offset = 0;

        for plist in &posting_lists {
            let size_bytes = bincode::serialized_size(plist).unwrap_or(0) as u32;
            let avg_norm = plist.len() as f32;

            directory.add_entry(PostingListEntry {
                cluster_id: plist.cluster_id,
                centroid_id: plist.cluster_id, // Usually the same
                disk_offset: offset,
                size_bytes,
                num_vectors: plist.len() as u32,
                avg_norm,
            });
            offset += size_bytes as u64 + 8; // Account for the 8-byte u64 length prefix
        }

        let posting_lists_data = PostingDataSource::InMemory(posting_lists);

        println!("MSTG index build complete");
        println!(
            "  Memory usage estimate: ~{:.2} MB",
            MstgIndex::estimate_memory_mb(&centroid_index, &posting_lists_data)
        );

        Ok(MstgIndex {
            config: self.config,
            centroid_index,
            posting_lists: posting_lists_data,
            directory,
        })
    }
}
