//! MSTG: Multi-Scale Tree Graph
//!
//! A hybrid memory-disk approximate nearest neighbor search system combining:
//! - SPANN's inverted index architecture
//! - RaBitQ quantization
//! - HNSW graph navigation

// Core data structures
pub mod config;
pub mod metadata;
pub mod posting_list;
pub mod scalar_quant;

// Algorithms
pub mod builder;
pub mod closure;
pub mod clustering;
pub mod hnsw;
pub mod index;
pub mod io;
pub mod search;

// Re-exports
pub use closure::ClosureAssigner;
pub use clustering::{Cluster, HierarchicalClustering};
pub use config::{MstgConfig, ScalarPrecision, SearchParams};
pub use hnsw::CentroidIndex;
pub use index::{MstgIndex, SearchResult};
pub use metadata::{PostingListDirectory, PostingListEntry};
pub use posting_list::{PostingList, QuantizedVectorWithId};
pub use scalar_quant::{BF16Vector, FP32Vector, QuantizedVector};
