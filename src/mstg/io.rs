//! MSTG Index Persistence (Save/Load)
//!
//! This module handles serialization and deserialization of MSTG indexes,
//! including the HNSW graph structure without rebuilding.

use super::*;
use crate::mstg::hnsw::{CentroidData, DistBF16, DistFP16, DistINT8, HnswIndex};
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

    /// Load the index from files into memory (InMemory mode)
    ///
    /// All posting lists are fully deserialized into RAM. This gives the fastest
    /// search performance but uses more memory.
    ///
    /// # Example
    /// ```no_run
    /// # use rabitq_rs::mstg::*;
    /// let index = MstgIndex::load_from_path("my_index").unwrap();
    /// ```
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Load main index using mmap internally, then convert to InMemory
        let index_path = format!("{}.mstg", path_str);
        let mut index = Self::load_main_index_inmemory(&index_path)?;

        // Load HNSW graph
        index.load_hnsw(&path_str)?;

        Ok(index)
    }

    /// Load the index from files using mmap (Disk mode)
    ///
    /// Only the centroid index (HNSW) and metadata directory are loaded into
    /// memory. Posting lists stay on disk and are deserialized on-the-fly during
    /// search via mmap. This uses much less memory but incurs per-query
    /// deserialization + `build_batch_layout()` overhead.
    ///
    /// # Example
    /// ```no_run
    /// # use rabitq_rs::mstg::*;
    /// let index = MstgIndex::load_from_path_mmap("my_index").unwrap();
    /// ```
    pub fn load_from_path_mmap<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Load main index with mmap (posting lists stay on disk)
        let index_path = format!("{}.mstg", path_str);
        let mut index = Self::load_main_index_mmap(&index_path)?;

        // Load HNSW graph
        index.load_hnsw(&path_str)?;

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

        // 3.5. Save directory
        let dir_bytes = bincode::serialize(&self.directory)
            .map_err(|_e| RabitqError::InvalidPersistence("failed to serialize directory"))?;
        write_u64(&mut writer, dir_bytes.len() as u64, Some(&mut hasher))?;
        writer.write_all(&dir_bytes)?;
        hasher.update(&dir_bytes);

        // 4. Save posting lists
        write_u64(
            &mut writer,
            self.posting_lists.len() as u64,
            Some(&mut hasher),
        )?;
        for plist in self.posting_lists.iter() {
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

    /// Load header (magic, version, config, centroid IDs, directory) from file.
    /// Returns (config, centroid_ids, directory, hasher, reader) positioned
    /// right at the start of the posting list section.
    #[allow(clippy::type_complexity)]
    fn load_header(
        path: &str,
    ) -> Result<
        (
            MstgConfig,
            Vec<u32>,
            PostingListDirectory,
            Hasher,
            BufReader<File>,
        ),
        RabitqError,
    > {
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

        // 3.5. Load directory
        let dir_len = read_u64(&mut reader, Some(&mut hasher))? as usize;
        let mut dir_bytes = vec![0u8; dir_len];
        reader.read_exact(&mut dir_bytes)?;
        hasher.update(&dir_bytes);

        let directory: PostingListDirectory = bincode::deserialize(&dir_bytes)
            .map_err(|_| RabitqError::InvalidPersistence("failed to deserialize directory"))?;

        Ok((config, centroid_ids, directory, hasher, reader))
    }

    /// Verify CRC32 checksum over the mmap'd remainder of the file.
    fn verify_checksum(
        mmap: &memmap2::Mmap,
        base_offset: usize,
        file_len: usize,
        mut hasher: Hasher,
    ) -> Result<(), RabitqError> {
        hasher.update(&mmap[base_offset..file_len - 4]);

        let mut stored_checksum_bytes = [0u8; 4];
        stored_checksum_bytes.copy_from_slice(&mmap[file_len - 4..file_len]);
        let stored_checksum = u32::from_le_bytes(stored_checksum_bytes);

        let computed_checksum = hasher.finalize();
        if stored_checksum != computed_checksum {
            return Err(RabitqError::InvalidPersistence("checksum mismatch"));
        }
        Ok(())
    }

    /// Extract centroid vectors from the mmap'd posting list data.
    fn extract_centroids(
        mmap: &memmap2::Mmap,
        base_offset: usize,
        num_centroids: usize,
    ) -> Result<Vec<Vec<f32>>, RabitqError> {
        let mut centroid_vecs: Vec<Vec<f32>> = Vec::with_capacity(num_centroids);
        let mut offset = base_offset + 8; // skip num_plists u64

        for _ in 0..num_centroids {
            let plist_len =
                u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            let plist: PostingList = bincode::deserialize(&mmap[offset..offset + plist_len])
                .map_err(|_| {
                    RabitqError::InvalidPersistence("failed to deserialize posting list")
                })?;
            centroid_vecs.push(plist.centroid);
            offset += plist_len;
        }
        Ok(centroid_vecs)
    }

    /// Load main index into InMemory mode: all posting lists are deserialized
    /// into RAM with batch layouts pre-built.
    fn load_main_index_inmemory(path: &str) -> Result<Self, RabitqError> {
        let (config, centroid_ids, directory, hasher, mut reader) = Self::load_header(path)?;
        let num_centroids = centroid_ids.len();

        // Mmap for checksum verification and posting list reading
        let base_offset = std::io::Seek::stream_position(&mut reader)? as usize;
        let file = reader.into_inner();
        let file_len = file.metadata()?.len() as usize;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        Self::verify_checksum(&mmap, base_offset, file_len, hasher)?;

        // Deserialize all posting lists into memory
        println!(
            "Deserializing {} posting lists into memory...",
            num_centroids
        );
        let mut posting_lists: Vec<PostingList> = Vec::with_capacity(num_centroids);
        let mut centroid_vecs: Vec<Vec<f32>> = Vec::with_capacity(num_centroids);
        let mut offset = base_offset + 8; // skip num_plists u64

        for _ in 0..num_centroids {
            let plist_len =
                u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            let mut plist: PostingList = bincode::deserialize(&mmap[offset..offset + plist_len])
                .map_err(|_| {
                    RabitqError::InvalidPersistence("failed to deserialize posting list")
                })?;
            plist.padded_dim = plist.centroid.len();
            centroid_vecs.push(plist.centroid.clone());
            posting_lists.push(plist);
            offset += plist_len;
        }

        // Rebuild FastScan batch layouts (not serialized)
        println!("Rebuilding FastScan batch layouts...");
        for plist in posting_lists.iter_mut() {
            plist.build_batch_layout();
        }

        let centroid_index =
            CentroidIndex::build(centroid_vecs, centroid_ids, config.centroid_precision);

        Ok(Self {
            config,
            centroid_index,
            posting_lists: PostingDataSource::InMemory(posting_lists),
            directory,
        })
    }

    /// Load main index in Mmap (disk) mode: posting lists stay on disk and are
    /// deserialized on demand during search.
    fn load_main_index_mmap(path: &str) -> Result<Self, RabitqError> {
        let (config, centroid_ids, directory, hasher, mut reader) = Self::load_header(path)?;
        let num_centroids = centroid_ids.len();

        // Mmap the file
        let base_offset = std::io::Seek::stream_position(&mut reader)? as usize;
        let file = reader.into_inner();
        let file_len = file.metadata()?.len() as usize;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        Self::verify_checksum(&mmap, base_offset, file_len, hasher)?;

        // Extract centroid vectors (needed for CentroidIndex)
        let centroid_vecs = Self::extract_centroids(&mmap, base_offset, num_centroids)?;

        let centroid_index =
            CentroidIndex::build(centroid_vecs, centroid_ids, config.centroid_precision);

        // +8 to skip the num_plists u64 prefix
        let posting_lists_data = PostingDataSource::Mmap(mmap, base_offset as u64 + 8);

        println!(
            "Loaded MSTG in disk mode: {} centroids, posting lists on mmap",
            num_centroids
        );

        Ok(Self {
            config,
            centroid_index,
            posting_lists: posting_lists_data,
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
        let hnsw_opt = cache.as_ref().ok_or_else(|| "HNSW not built".to_string())?;

        let path_string = base_path.to_string();

        match hnsw_opt {
            HnswIndex::FP32(h) => h.file_dump(&path_string),
            HnswIndex::BF16(h) => h.file_dump(&path_string),
            HnswIndex::FP16(h) => h.file_dump(&path_string),
            HnswIndex::INT8(h) => h.file_dump(&path_string),
        }
        .map_err(|e| format!("HNSW dump failed: {}", e))?;

        Ok(())
    }

    /// Load HNSW from disk using hnsw_rs API
    pub(crate) fn load_hnsw(&mut self, base_path: &str) -> Result<(), String> {
        use hnsw_rs::hnswio::*;
        use hnsw_rs::prelude::*;

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

        let hnswio = HnswIo::new(dir, basename);

        let hnsw_index = match &self.centroids {
            CentroidData::FP32(_) => {
                let hnsw: Hnsw<f32, DistL2> = hnswio
                    .load_hnsw_with_dist(DistL2 {})
                    .map_err(|e| format!("HNSW load failed: {}", e))?;
                let hnsw_static: Hnsw<'static, f32, DistL2> = unsafe { std::mem::transmute(hnsw) };
                HnswIndex::FP32(hnsw_static)
            }
            CentroidData::BF16(_) => {
                let hnsw: Hnsw<u16, DistBF16> = hnswio
                    .load_hnsw_with_dist(DistBF16 {})
                    .map_err(|e| format!("HNSW load failed: {}", e))?;
                let hnsw_static: Hnsw<'static, u16, DistBF16> =
                    unsafe { std::mem::transmute(hnsw) };
                HnswIndex::BF16(hnsw_static)
            }
            CentroidData::FP16(_) => {
                let hnsw: Hnsw<half::f16, DistFP16> = hnswio
                    .load_hnsw_with_dist(DistFP16 {})
                    .map_err(|e| format!("HNSW load failed: {}", e))?;
                let hnsw_static: Hnsw<'static, half::f16, DistFP16> =
                    unsafe { std::mem::transmute(hnsw) };
                HnswIndex::FP16(hnsw_static)
            }
            CentroidData::INT8 { scale, .. } => {
                let hnsw: Hnsw<i8, DistINT8> = hnswio
                    .load_hnsw_with_dist(DistINT8 { scale: *scale })
                    .map_err(|e| format!("HNSW load failed: {}", e))?;
                let hnsw_static: Hnsw<'static, i8, DistINT8> = unsafe { std::mem::transmute(hnsw) };
                HnswIndex::INT8(hnsw_static)
            }
        };

        let mut cache = self.hnsw_cache.write();
        *cache = Some(hnsw_index);
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
        assert_eq!(index.directory.len(), loaded_index.directory.len());
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

    #[test]
    fn test_save_load_mstg_mmap() {
        let data = generate_test_data(10000, 128);
        let config = MstgConfig {
            max_posting_size: 20,
            branching_factor: 8,
            ..Default::default()
        };

        let index = MstgIndex::build(&data, config.clone()).unwrap();

        let path = format!(
            "/tmp/test_mstg_mmap_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        index.save_to_path(&path).unwrap();

        // Load as mmap (disk mode)
        let mmap_index = MstgIndex::load_from_path_mmap(&path).unwrap();

        // Verify properties
        assert_eq!(index.directory.len(), mmap_index.directory.len());
        assert_eq!(index.centroid_index.len(), mmap_index.centroid_index.len());

        // Posting lists should be Mmap variant
        assert!(matches!(
            mmap_index.posting_lists,
            PostingDataSource::Mmap(_, _)
        ));

        // Search should work in mmap mode
        let query = &data[0];
        let params = SearchParams::balanced(10);

        let results_mem = index.search(query, &params);
        let results_mmap = mmap_index.search(query, &params);

        assert!(!results_mem.is_empty());
        assert!(!results_mmap.is_empty());
        // Top result should match
        assert_eq!(results_mem[0].vector_id, results_mmap[0].vector_id);

        // Clean up
        let _ = std::fs::remove_file(format!("{}.mstg", path));
        let _ = std::fs::remove_file(format!("{}.hnsw.graph", path));
        let _ = std::fs::remove_file(format!("{}.hnsw.data", path));
    }
}
