pub mod ivf;

mod kmeans;
mod math;
mod quantizer;
mod rotation;

pub use ivf::{IvfRabitqIndex, SearchParams, SearchResult};
pub use quantizer::{QuantizedVector, RabitqConfig};

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
#[derive(thiserror::Error, Debug, PartialEq)]
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
}
