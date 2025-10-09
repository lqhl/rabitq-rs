use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::brute_force::BruteForceRabitqIndex;
use crate::ivf::IvfRabitqIndex;
use crate::RabitqError;

/// Unified index type that can be either IVF or BruteForce.
///
/// This enum allows loading an index without knowing its type in advance.
/// The smart loader (`RabitqIndex::load_from_path` or `load_from_reader`)
/// automatically detects the index type based on the file magic header.
///
/// # Example
///
/// ```no_run
/// use rabitq_rs::RabitqIndex;
///
/// // Load index without knowing its type
/// let index = RabitqIndex::load_from_path("index.bin").expect("Failed to load");
///
/// // Use pattern matching to handle different index types
/// match index {
///     RabitqIndex::Ivf(ivf_index) => {
///         // Use IVF-specific search parameters
///         println!("Loaded IVF index with {} clusters", ivf_index.cluster_count());
///     }
///     RabitqIndex::BruteForce(bf_index) => {
///         // Use BruteForce-specific search parameters
///         println!("Loaded BruteForce index with {} vectors", bf_index.len());
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub enum RabitqIndex {
    /// IVF + RaBitQ index with clustering
    Ivf(IvfRabitqIndex),
    /// Brute-force RaBitQ index without clustering
    BruteForce(BruteForceRabitqIndex),
}

impl RabitqIndex {
    /// Load an index from a file path, automatically detecting its type.
    ///
    /// This method reads the magic header from the file and dispatches to
    /// the appropriate loader (`IvfRabitqIndex` or `BruteForceRabitqIndex`).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the serialized index file
    ///
    /// # Returns
    ///
    /// * `Ok(RabitqIndex::Ivf)` if the file contains an IVF index (magic: `b"RBQ1"`)
    /// * `Ok(RabitqIndex::BruteForce)` if the file contains a BruteForce index (magic: `b"RBF1"`)
    /// * `Err(RabitqError::InvalidPersistence)` if the magic header is unrecognized
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rabitq_rs::RabitqIndex;
    ///
    /// let index = RabitqIndex::load_from_path("my_index.bin")?;
    /// match index {
    ///     RabitqIndex::Ivf(ivf) => println!("Loaded IVF index"),
    ///     RabitqIndex::BruteForce(bf) => println!("Loaded BruteForce index"),
    /// }
    /// # Ok::<(), rabitq_rs::RabitqError>(())
    /// ```
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, RabitqError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader)
    }

    /// Load an index from a reader, automatically detecting its type.
    ///
    /// This method reads the magic header and dispatches to the appropriate loader.
    /// The reader must support seeking (for rewinding after reading the magic header).
    ///
    /// # Arguments
    ///
    /// * `reader` - A readable and seekable source containing the serialized index
    ///
    /// # Returns
    ///
    /// * `Ok(RabitqIndex::Ivf)` if magic is `b"RBQ1"`
    /// * `Ok(RabitqIndex::BruteForce)` if magic is `b"RBF1"`
    /// * `Err(RabitqError::InvalidPersistence)` if magic is unrecognized
    pub fn load_from_reader<R: Read + Seek>(mut reader: R) -> Result<Self, RabitqError> {
        // Read magic header (4 bytes)
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        // Rewind to the beginning
        reader.seek(SeekFrom::Start(0))?;

        // Dispatch based on magic
        match &magic {
            b"RBQ1" => {
                // IVF index
                let ivf_index = IvfRabitqIndex::load_from_reader(reader)?;
                Ok(RabitqIndex::Ivf(ivf_index))
            }
            b"RBF1" => {
                // BruteForce index
                let bf_index = BruteForceRabitqIndex::load_from_reader(reader)?;
                Ok(RabitqIndex::BruteForce(bf_index))
            }
            _ => Err(RabitqError::InvalidPersistence(
                "unrecognized index magic header (expected RBQ1 or RBF1)",
            )),
        }
    }

    /// Returns the number of vectors stored in the index.
    pub fn len(&self) -> usize {
        match self {
            RabitqIndex::Ivf(idx) => idx.len(),
            RabitqIndex::BruteForce(idx) => idx.len(),
        }
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        match self {
            RabitqIndex::Ivf(idx) => idx.is_empty(),
            RabitqIndex::BruteForce(idx) => idx.is_empty(),
        }
    }

    /// Returns true if this is an IVF index.
    pub fn is_ivf(&self) -> bool {
        matches!(self, RabitqIndex::Ivf(_))
    }

    /// Returns true if this is a BruteForce index.
    pub fn is_brute_force(&self) -> bool {
        matches!(self, RabitqIndex::BruteForce(_))
    }

    /// Unwraps the IVF index, panicking if it's not an IVF index.
    ///
    /// # Panics
    ///
    /// Panics if the index is not `RabitqIndex::Ivf`.
    pub fn unwrap_ivf(self) -> IvfRabitqIndex {
        match self {
            RabitqIndex::Ivf(idx) => idx,
            RabitqIndex::BruteForce(_) => panic!("called unwrap_ivf on BruteForce index"),
        }
    }

    /// Unwraps the BruteForce index, panicking if it's not a BruteForce index.
    ///
    /// # Panics
    ///
    /// Panics if the index is not `RabitqIndex::BruteForce`.
    pub fn unwrap_brute_force(self) -> BruteForceRabitqIndex {
        match self {
            RabitqIndex::BruteForce(idx) => idx,
            RabitqIndex::Ivf(_) => panic!("called unwrap_brute_force on IVF index"),
        }
    }

    /// Returns a reference to the IVF index if it is one.
    pub fn as_ivf(&self) -> Option<&IvfRabitqIndex> {
        match self {
            RabitqIndex::Ivf(idx) => Some(idx),
            RabitqIndex::BruteForce(_) => None,
        }
    }

    /// Returns a reference to the BruteForce index if it is one.
    pub fn as_brute_force(&self) -> Option<&BruteForceRabitqIndex> {
        match self {
            RabitqIndex::BruteForce(idx) => Some(idx),
            RabitqIndex::Ivf(_) => None,
        }
    }

    /// Returns a mutable reference to the IVF index if it is one.
    pub fn as_ivf_mut(&mut self) -> Option<&mut IvfRabitqIndex> {
        match self {
            RabitqIndex::Ivf(idx) => Some(idx),
            RabitqIndex::BruteForce(_) => None,
        }
    }

    /// Returns a mutable reference to the BruteForce index if it is one.
    pub fn as_brute_force_mut(&mut self) -> Option<&mut BruteForceRabitqIndex> {
        match self {
            RabitqIndex::BruteForce(idx) => Some(idx),
            RabitqIndex::Ivf(_) => None,
        }
    }
}
