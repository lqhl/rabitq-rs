use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Directory of all posting lists with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingListDirectory {
    pub entries: Vec<PostingListEntry>,
    pub centroid_to_posting: HashMap<u32, u32>,
}

/// Metadata for a single posting list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingListEntry {
    pub cluster_id: u32,
    pub centroid_id: u32,
    pub disk_offset: u64,
    pub size_bytes: u32,
    pub num_vectors: u32,
    pub avg_norm: f32,
}

impl PostingListDirectory {
    /// Create a new empty directory
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            centroid_to_posting: HashMap::new(),
        }
    }

    /// Add an entry to the directory
    pub fn add_entry(&mut self, entry: PostingListEntry) {
        let posting_id = self.entries.len() as u32;
        self.centroid_to_posting
            .insert(entry.centroid_id, posting_id);
        self.entries.push(entry);
    }

    /// Get posting list ID from centroid ID
    pub fn get_posting_id(&self, centroid_id: u32) -> Option<u32> {
        self.centroid_to_posting.get(&centroid_id).copied()
    }

    /// Get total number of posting lists
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if directory is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Estimate memory usage in bytes
    pub fn memory_size(&self) -> usize {
        self.entries.len() * std::mem::size_of::<PostingListEntry>()
            + self.centroid_to_posting.len() * (std::mem::size_of::<u32>() * 2)
    }
}

impl Default for PostingListDirectory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directory_operations() {
        let mut dir = PostingListDirectory::new();

        assert!(dir.is_empty());
        assert_eq!(dir.len(), 0);

        let entry = PostingListEntry {
            cluster_id: 0,
            centroid_id: 42,
            disk_offset: 0,
            size_bytes: 1024,
            num_vectors: 100,
            avg_norm: 1.5,
        };

        dir.add_entry(entry);

        assert!(!dir.is_empty());
        assert_eq!(dir.len(), 1);
        assert_eq!(dir.get_posting_id(42), Some(0));
        assert_eq!(dir.get_posting_id(99), None);
    }
}
