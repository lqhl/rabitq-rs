pub mod io;
pub mod ivf;

mod kmeans;
mod math;
mod quantizer;
mod rotation;

pub use ivf::{IvfRabitqIndex, SearchParams, SearchResult};
pub use quantizer::{QuantizedVector, RabitqConfig};
pub use rotation::RotatorType;

// Re-export RoaringBitmap for user convenience
pub use roaring::RoaringBitmap;

#[cfg(test)]
mod tests;

/// Distance metric supported by the RaBitQ IVF index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance (L2).
    L2,
    /// Inner product (maximum similarity).
    InnerProduct,
}

/// Errors that can occur when building or querying the RaBitQ IVF index.
#[derive(thiserror::Error, Debug)]
pub enum RabitqError {
    /// Returned when the dimension of an input vector does not match the trained index.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// Returned when an invalid configuration is supplied.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    /// Returned when the index has not been trained before use.
    #[error("index is empty; call `train` first")]
    EmptyIndex,
    /// Returned when persistence encounters an I/O failure.
    #[error("i/o error while reading or writing an index: {0}")]
    Io(#[from] std::io::Error),
    /// Returned when the persisted bytes are inconsistent or corrupt.
    #[error("invalid persisted index: {0}")]
    InvalidPersistence(&'static str),
}
