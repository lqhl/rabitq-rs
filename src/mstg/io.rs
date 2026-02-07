//! MSTG Index Persistence (Save/Load)
//!
//! This module handles serialization and deserialization of MSTG indexes,
//! including the HNSW graph structure without rebuilding.

use super::*;
use crate::RabitqError;
use crc32fast::Hasher;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Magic bytes to identify MSTG index files
const MAGIC_BYTES: &[u8; 4] = b"MSTG";

/// Current persistence version
const PERSISTENCE_VERSION: u32 = 1;

/// Write u32 with optional CRC update
fn write_u32<W: Write>(
    writer: &mut W,
    value: u32,
    hasher: Option<&mut Hasher>,
) -> std::io::Result<()> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    Ok(())
}

/// Read u32 with optional CRC update
fn read_u32<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    Ok(u32::from_le_bytes(bytes))
}

/// Write u64 with optional CRC update
fn write_u64<W: Write>(
    writer: &mut W,
    value: u64,
    hasher: Option<&mut Hasher>,
) -> std::io::Result<()> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    Ok(())
}

/// Read u64 with optional CRC update
fn read_u64<R: Read>(reader: &mut R, hasher: Option<&mut Hasher>) -> std::io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    if let Some(h) = hasher {
        h.update(&bytes);
    }
    Ok(u64::from_le_bytes(bytes))
}

impl MstgIndex {
    /// Save the index to a file
    ///
    /// This creates multiple files:
    /// - `{path}.mstg` - Main index file (config + posting lists)
    /// - `{path}.hnsw.graph` - HNSW graph structure
    /// - `{path}.hnsw.data` - HNSW data points
    ///
    /// # Example
    /// ```no_run
    /// # use rabitq_rs::mstg::*;
    /// # let index: MstgIndex = todo!();
    /// index.save_to_path("my_index").unwrap();
    /// // Creates: my_index.mstg, my_index.hnsw.graph, my_index.hnsw.data
    /// ```
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), RabitqError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Save main index file
        let index_path = format!("{}.mstg", path_str);
        self.save_main_index(&index_path)?;

        // Save HNSW graph (using hnsw_rs API)
        self.save_hnsw(&path_str)?;

        Ok(())
    }

    /// Load the index from a file
    ///
    /// This loads from the files created by `save_to_path`.
    ///
    /// # Example
    /// ```no_run
    /// # use rabitq_rs::mstg::*;
    /// let index = MstgIndex::load_from_path("my_index").unwrap();
    /// ```
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Load main index
        let index_path = format!("{}.mstg", path_str);
        let mut index = Self::load_main_index(&index_path)?;

        // Load HNSW graph
        index.load_hnsw(&path_str)?;

        // Fix padded_dim (it's #[serde(skip)] so it's 0 after deserialization)
        for plist in &mut index.posting_lists {
            plist.padded_dim = plist.centroid.len();
        }

        // Rebuild batch layouts for FastScan (batch_data is not serialized)
        println!("Rebuilding FastScan batch layouts...");
        for plist in &mut index.posting_lists {
            plist.build_batch_layout();
        }

        Ok(index)
    }

    /// Save main index (config + posting lists) to a file
    fn save_main_index(&self, path: &str) -> Result<(), RabitqError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut hasher = Hasher::new();

        // 1. Write magic and version
        writer.write_all(MAGIC_BYTES)?;
        write_u32(&mut writer, PERSISTENCE_VERSION, None)?;

        // 2. Serialize config using bincode
        let config_bytes = bincode::serialize(&self.config)
            .map_err(|_e| RabitqError::InvalidPersistence("failed to serialize config"))?;
        write_u64(&mut writer, config_bytes.len() as u64, Some(&mut hasher))?;
        writer.write_all(&config_bytes)?;
        hasher.update(&config_bytes);

        // 3. Save centroid IDs (needed for HNSW reconstruction)
        let centroid_ids = self.centroid_index.get_all_ids();
        write_u64(&mut writer, centroid_ids.len() as u64, Some(&mut hasher))?;
        for &id in &centroid_ids {
            write_u32(&mut writer, id, Some(&mut hasher))?;
        }

        // 4. Save posting lists
        write_u64(
            &mut writer,
            self.posting_lists.len() as u64,
            Some(&mut hasher),
        )?;
        for plist in &self.posting_lists {
            let plist_bytes = bincode::serialize(plist)
                .map_err(|_| RabitqError::InvalidPersistence("failed to serialize posting list"))?;
            write_u64(&mut writer, plist_bytes.len() as u64, Some(&mut hasher))?;
            writer.write_all(&plist_bytes)?;
            hasher.update(&plist_bytes);
        }

        // 5. Write CRC32 checksum
        let checksum = hasher.finalize();
        write_u32(&mut writer, checksum, None)?;

        writer.flush()?;
        Ok(())
    }

    /// Load main index from a file
    fn load_main_index(path: &str) -> Result<Self, RabitqError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Hasher::new();

        // 1. Verify magic and version
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC_BYTES {
            return Err(RabitqError::InvalidPersistence("invalid magic bytes"));
        }

        let version = read_u32(&mut reader, None)?;
        if version != PERSISTENCE_VERSION {
            return Err(RabitqError::InvalidPersistence("unsupported version"));
        }

        // 2. Load config
        let config_len = read_u64(&mut reader, Some(&mut hasher))? as usize;
        let mut config_bytes = vec![0u8; config_len];
        reader.read_exact(&mut config_bytes)?;
        hasher.update(&config_bytes);

        let config: MstgConfig = bincode::deserialize(&config_bytes)
            .map_err(|_| RabitqError::InvalidPersistence("failed to deserialize config"))?;

        // 3. Load centroid IDs
        let num_centroids = read_u64(&mut reader, Some(&mut hasher))? as usize;
        let mut centroid_ids = Vec::with_capacity(num_centroids);
        for _ in 0..num_centroids {
            centroid_ids.push(read_u32(&mut reader, Some(&mut hasher))?);
        }

        // 4. Load posting lists
        let num_plists = read_u64(&mut reader, Some(&mut hasher))? as usize;
        let mut posting_lists = Vec::with_capacity(num_plists);
        for _ in 0..num_plists {
            let plist_len = read_u64(&mut reader, Some(&mut hasher))? as usize;
            let mut plist_bytes = vec![0u8; plist_len];
            reader.read_exact(&mut plist_bytes)?;
            hasher.update(&plist_bytes);

            let plist: PostingList = bincode::deserialize(&plist_bytes).map_err(|_| {
                RabitqError::InvalidPersistence("failed to deserialize posting list")
            })?;
            posting_lists.push(plist);
        }

        // 5. Verify checksum
        let stored_checksum = read_u32(&mut reader, None)?;
        let computed_checksum = hasher.finalize();
        if stored_checksum != computed_checksum {
            return Err(RabitqError::InvalidPersistence("checksum mismatch"));
        }

        // Reconstruct centroid vectors from posting lists
        let centroid_vecs: Vec<Vec<f32>> =
            posting_lists.iter().map(|p| p.centroid.clone()).collect();

        let centroid_index = CentroidIndex::build(
            centroid_vecs,
            centroid_ids,
            config.centroid_precision,
            config.hnsw_m,
            config.hnsw_ef_construction,
            config.metric,
        );

        let directory = PostingListDirectory::new();

        Ok(Self {
            config,
            centroid_index,
            posting_lists,
            directory,
        })
    }

    /// Save HNSW graph using hnsw_rs dump API
    fn save_hnsw(&self, base_path: &str) -> Result<(), RabitqError> {
        // Use hnsw_rs file_dump API
        // This creates {base_path}.hnsw.graph and {base_path}.hnsw.data
        self.centroid_index.save_hnsw(base_path).map_err(|e| {
            RabitqError::Io(std::io::Error::other(format!("HNSW save failed: {}", e)))
        })?;

        Ok(())
    }

    /// Load HNSW graph using hnsw_rs reload API
    fn load_hnsw(&mut self, base_path: &str) -> Result<(), RabitqError> {
        self.centroid_index.load_hnsw(base_path).map_err(|e| {
            RabitqError::Io(std::io::Error::other(format!("HNSW load failed: {}", e)))
        })?;

        Ok(())
    }
}

impl CentroidIndex {
    /// Save HNSW to disk using hnsw_rs API
    pub(crate) fn save_hnsw(&self, base_path: &str) -> Result<(), String> {
        use hnsw_rs::api::AnnT; // Required for file_dump method

        // Ensure HNSW is built
        self.ensure_hnsw_built();

        let cache = self.hnsw_cache.read();
        let hnsw = cache.as_ref().ok_or_else(|| "HNSW not built".to_string())?;

        // Use hnsw_rs file_dump API
        let path_string = base_path.to_string();
        match hnsw {
            super::hnsw::HnswCache::L2(inner) => inner
                .file_dump(&path_string)
                .map_err(|e| format!("HNSW dump failed: {}", e))?,
            super::hnsw::HnswCache::Dot(inner) => inner
                .file_dump(&path_string)
                .map_err(|e| format!("HNSW dump failed: {}", e))?,
        };

        Ok(())
    }

    /// Load HNSW from disk using hnsw_rs API
    pub(crate) fn load_hnsw(&mut self, base_path: &str) -> Result<(), String> {
        use hnsw_rs::hnswio::*;
        use hnsw_rs::prelude::*;

        // Prepare directory and basename for hnsw_rs loader
        let path = PathBuf::from(base_path);
        let dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let basename = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "invalid path".to_string())?
            .to_string();

        // Load HNSW using hnsw_rs API
        let hnswio = HnswIo::new(dir, basename);

        let hnsw_cache = match self.metric() {
            crate::Metric::L2 => {
                let hnsw: Hnsw<f32, DistL2> = hnswio
                    .load_hnsw_with_dist(DistL2 {})
                    .map_err(|e| format!("HNSW load failed: {}", e))?;
                let hnsw_static: Hnsw<'static, f32, DistL2> = unsafe { std::mem::transmute(hnsw) };
                super::hnsw::HnswCache::L2(hnsw_static)
            }
            crate::Metric::InnerProduct => {
                let hnsw: Hnsw<f32, DistDot> = hnswio
                    .load_hnsw_with_dist(DistDot {})
                    .map_err(|e| format!("HNSW load failed: {}", e))?;
                let hnsw_static: Hnsw<'static, f32, DistDot> = unsafe { std::mem::transmute(hnsw) };
                super::hnsw::HnswCache::Dot(hnsw_static)
            }
        };

        // Update cache
        let mut cache = self.hnsw_cache.write();
        *cache = Some(hnsw_cache);

        Ok(())
    }

    /// Get all centroid IDs
    pub(crate) fn get_all_ids(&self) -> Vec<u32> {
        self.centroid_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mstg::SearchParams;
    use rand::prelude::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(42);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen()).collect())
            .collect()
    }

    #[test]
    fn test_save_load_mstg_index() {
        // Use larger dataset to avoid hnsw_rs serialization issues with small datasets
        // Need enough centroids for HNSW to build proper layers (aim for 500+ centroids)
        let data = generate_test_data(10000, 128);
        let config = MstgConfig {
            max_posting_size: 20, // Smaller posting size to generate more centroids
            branching_factor: 8,  // Higher branching to create more leaf nodes
            ..Default::default()
        };

        // Build index
        let index = MstgIndex::build(&data, config.clone()).unwrap();

        // Save to temp file (using timestamp to avoid conflicts)
        let path = format!(
            "/tmp/test_mstg_save_load_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        index.save_to_path(&path).unwrap();

        // Load index
        let loaded_index = MstgIndex::load_from_path(&path).unwrap();

        // Verify basic properties
        assert_eq!(index.posting_lists.len(), loaded_index.posting_lists.len());
        assert_eq!(
            index.centroid_index.len(),
            loaded_index.centroid_index.len()
        );

        // Test search on both
        let query = &data[0];
        let params = SearchParams::balanced(10);

        let results1 = index.search(query, &params);
        let results2 = loaded_index.search(query, &params);

        // Results should be very similar (may have minor differences due to HNSW randomness)
        assert!(!results1.is_empty());
        assert!(!results2.is_empty());
        assert_eq!(results1[0].vector_id, results2[0].vector_id);

        // Clean up temp files
        let _ = std::fs::remove_file(format!("{}.mstg", path));
        let _ = std::fs::remove_file(format!("{}.hnsw.graph", path));
        let _ = std::fs::remove_file(format!("{}.hnsw.data", path));
    }
}
