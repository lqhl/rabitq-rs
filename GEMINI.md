# GEMINI.md

## Project Overview

**Name:** RaBitQ-RS
**Type:** Rust Library with Python Bindings
**Purpose:** High-performance vector quantization and approximate nearest neighbor (ANN) search library.
**Core Technology:**
*   **RaBitQ:** Rapid Binary Quantization for compressing high-dimensional vectors.
*   **IVF:** Inverted File Index (production-ready).
*   **MSTG:** Multi-Scale Tree Graph (experimental, hybrid memory-disk index).
*   **HNSW:** Hierarchical Navigable Small World graphs (used for centroid navigation in MSTG).

**Key Features:**
*   **High Compression:** Up to 32x memory reduction compared to raw vectors.
*   **High Accuracy:** Better recall than traditional Product Quantization (PQ) or Scalar Quantization (SQ).
*   **SIMD Optimization:** Accelerated distance computations (AVX2/AVX-512).
*   **Python Integration:** Seamless bindings via `pyo3` and `maturin`.

## Directory Structure

*   `src/`: Core Rust source code.
    *   `lib.rs`: Crate entry point.
    *   `ivf.rs`: Production-ready IVF implementation.
    *   `mstg/`: Experimental MSTG implementation (Hierarchical clustering, HNSW navigation).
    *   `quantizer.rs`: RaBitQ quantization logic.
    *   `simd.rs`: Low-level SIMD intrinsics.
    *   `python_bindings.rs`: Python API definitions.
*   `python/`: Python package source (wrapper code).
*   `examples/`: Rust usage examples.
*   `benchmarks/`: Benchmark scripts and data.
*   `docs/`: Design documents (e.g., `MSTG_SPEC.md`, `SPANN.pdf`).
*   `ann_benchmarks_templates/`: Integration with `ann-benchmarks`.
*   `Cargo.toml`: Rust package configuration.
*   `pyproject.toml`: Python package configuration (Maturin).
*   `Makefile`: Automation for build, test, and benchmarks.

## Build and Usage

### Prerequisites
*   Rust toolchain (stable)
*   Python 3.8+
*   `maturin` (for Python bindings)

### Rust
*   **Build:** `cargo build --release`
*   **Test:** `cargo test`
*   **Lint:** `cargo clippy`
*   **Format:** `cargo fmt`
*   **Run CLI:** `cargo run --release --bin ivf_rabitq -- --help`

### Python
*   **Build & Install (Dev):** `make build-python` (uses `maturin develop`)
*   **Build Wheel:** `make install-python`
*   **Test:** `make test-python` (runs `test_python_bindings.py`)
*   **Clean:** `make clean-python`

### Benchmarking
*   **Local:** `make bench-local` (requires `ann-benchmarks` repo sibling)
*   **Docker:** `make bench-docker`

## MSTG Architecture (Experimental)

MSTG is designed for billion-scale datasets, combining memory efficiency with high recall.

**Three-Tier Design:**
1.  **Memory Tier:** HNSW graph for fast centroid navigation (~10-15% of vectors). Centroids can be scalar quantized (BF16, FP16, INT8).
2.  **Metadata Tier:** Directory of posting lists (offsets, sizes, quantization configs).
3.  **Disk Tier:** RaBitQ-compressed posting lists stored sequentially.

**Key Algorithms:**
*   **Hierarchical Balanced Clustering:** Adaptive multi-scale clustering.
*   **Closure Assignment:** Vectors assigned to multiple clusters (RNG rule) for better recall at boundaries.
*   **Dynamic Pruning:** Query-aware pruning of search space.
*   **FastScan:** Batch distance computation using SIMD.

## Development Conventions

*   **Code Style:** Standard Rust `cargo fmt`.
*   **Testing:** Comprehensive unit tests (`cargo test`) and Python integration tests.
*   **Performance:** Heavy use of `rayon` for parallelism and explicit SIMD for critical paths.
*   **Safety:** `unsafe` code is limited to SIMD and necessary FFI, but the safe API is preferred.
*   **Platform:** x86_64 is fully supported. ARM64 (Apple Silicon) has known issues with distance calculations.

## Useful Commands

*   **Check ARM64 issues:** See `ARM64_KNOWN_ISSUES.md` (if investigating M1/M2 Mac bugs).
*   **Generate Documentation:** `cargo doc --open`
