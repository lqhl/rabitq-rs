//! Main MSTG index structure

use super::*;

/// Source of posting lists (memory or mmapped disk)
pub enum PostingDataSource {
    InMemory(Vec<PostingList>),
    Mmap(memmap2::Mmap, u64),
}

impl PostingDataSource {
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(v) => v.len(),
            Self::Mmap(_, _) => usize::MAX, // Can't know without directory, assume not empty
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PostingList> {
        match self {
            Self::InMemory(v) => v.iter(),
            Self::Mmap(_, _) => [].iter(), // Placeholder for Mmap iteration
        }
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, PostingList> {
        match self {
            Self::InMemory(v) => v.iter_mut(),
            Self::Mmap(_, _) => [].iter_mut(), // Placeholder for Mmap iteration
        }
    }

    pub fn with_posting_list<F, R>(
        &self,
        cid: u32,
        directory: &PostingListDirectory,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&PostingList) -> R,
    {
        match self {
            Self::InMemory(v) => v.get(cid as usize).map(f),
            Self::Mmap(mmap, base_offset) => {
                if let Some(entry) = directory.entries.get(cid as usize) {
                    let prefix_start = *base_offset as usize + entry.disk_offset as usize;
                    if prefix_start + 8 <= mmap.len() {
                        let mut len_bytes = [0u8; 8];
                        len_bytes.copy_from_slice(&mmap[prefix_start..prefix_start + 8]);
                        let actual_len = u64::from_le_bytes(len_bytes) as usize;

                        let start = prefix_start + 8;
                        let end = start + actual_len;

                        if end <= mmap.len() {
                            if let Ok(mut plist) =
                                bincode::deserialize::<PostingList>(&mmap[start..end])
                            {
                                plist.padded_dim = plist.centroid.len();
                                plist.build_batch_layout();
                                return Some(f(&plist));
                            }
                        }
                    }
                }
                None
            }
        }
    }
}

/// Main MSTG (Multi-Scale Tree Graph) index
pub struct MstgIndex {
    pub config: MstgConfig,
    pub centroid_index: CentroidIndex,
    pub posting_lists: PostingDataSource,
    pub directory: PostingListDirectory,
}

impl MstgIndex {
    /// Build an MSTG index from data
    pub fn build(data: &[Vec<f32>], config: MstgConfig) -> Result<Self, String> {
        MstgBuilder::new(data, config).build()
    }

    /// Estimate total memory usage in MB
    pub fn estimate_memory_mb(
        centroid_index: &CentroidIndex,
        posting_lists: &PostingDataSource,
    ) -> f32 {
        let centroid_mem = centroid_index.memory_usage();
        let posting_mem: usize = match posting_lists {
            PostingDataSource::InMemory(v) => v.iter().map(|p| p.memory_size()).sum(),
            PostingDataSource::Mmap(_, _) => 0,
        };
        ((centroid_mem + posting_mem) as f32) / (1024.0 * 1024.0)
    }
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
        assert!(!index.centroid_index.is_empty());
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

        // The query vector (id 0) should be retrieved in top-k.
        // Distance can vary across platforms/configs due to approximate search.
        assert!(
            results.iter().any(|r| r.vector_id == 0),
            "query vector id 0 not found in top results: {:?}",
            results
                .iter()
                .map(|r| (r.vector_id, r.distance))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_memory_differences_scalar_quantization() {
        use crate::mstg::config::ScalarPrecision;

        // Generate larger dummy data to show memory differences
        let data = generate_test_data(500, 128);

        // Build with FP32
        let config_fp32 = MstgConfig {
            centroid_precision: ScalarPrecision::FP32,
            ..Default::default()
        };
        let index_fp32 = MstgIndex::build(&data, config_fp32).unwrap();
        let mem_fp32 =
            MstgIndex::estimate_memory_mb(&index_fp32.centroid_index, &index_fp32.posting_lists);

        // Build with BF16
        let config_bf16 = MstgConfig {
            centroid_precision: ScalarPrecision::BF16,
            ..Default::default()
        };
        let index_bf16 = MstgIndex::build(&data, config_bf16).unwrap();
        let mem_bf16 =
            MstgIndex::estimate_memory_mb(&index_bf16.centroid_index, &index_bf16.posting_lists);

        // Build with INT8
        let config_int8 = MstgConfig {
            centroid_precision: ScalarPrecision::INT8,
            ..Default::default()
        };
        let index_int8 = MstgIndex::build(&data, config_int8).unwrap();
        let mem_int8 =
            MstgIndex::estimate_memory_mb(&index_int8.centroid_index, &index_int8.posting_lists);

        // Compare HNSW memory
        println!("FP32 Memory: {} MB", mem_fp32);
        println!("BF16 Memory: {} MB", mem_bf16);
        println!("INT8 Memory: {} MB", mem_int8);

        assert!(mem_int8 < mem_bf16);
        assert!(mem_bf16 < mem_fp32);
    }
}
