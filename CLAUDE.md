# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This is a pure-Rust implementation of the RaBitQ quantization scheme with IVF (Inverted File) search capabilities. The project mirrors the behavior of the C++ RaBitQ Library and provides tools to reproduce the GIST benchmark pipeline.

## Key Commands

### Build, Test, and Lint
```bash
cargo fmt                                    # Format code (run before commits)
cargo clippy --all-targets --all-features    # Lint (must pass cleanly)
cargo test                                   # Run unit and integration tests

# SIMD/AVX-512 builds (optional, for maximum performance)
cargo build --release                        # Stable Rust, uses AVX2 (3-5x speedup)
cargo +nightly build --release --features avx512  # Nightly, uses AVX-512 (6-10x speedup)
cargo +nightly test --features avx512        # Test with AVX-512 enabled
```

### Benchmark Evaluation (ivf_rabitq binary)
```bash
# Build index
cargo run --release --bin ivf_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --centroids data/gist/gist_centroids_4096.fvecs \
    --assignments data/gist/gist_clusterids_4096.ivecs \
    --bits 3 \
    --save data/gist/index.bin

# Query with benchmark mode (nprobe sweep + 5 rounds)
cargo run --release --bin ivf_rabitq -- \
    --load data/gist/index.bin \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --benchmark

# Query at specific nprobe (no sweep)
cargo run --release --bin ivf_rabitq -- \
    --load data/gist/index.bin \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --nprobe 1024 \
    --top-k 100

# Build and query in one command
cargo run --release --bin ivf_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --nlist 4096 \
    --bits 3 \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --benchmark

# Smoke test with limited data
cargo run --release --bin ivf_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --nlist 4096 \
    --bits 3 \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --max-base 1000 \
    --max-queries 50 \
    --nprobe 64
```

### Python Helper (Pre-clustering with FAISS)
```bash
python python/ivf.py \
    data/gist/gist_base.fvecs \
    4096 \
    data/gist/gist_centroids_4096.fvecs \
    data/gist/gist_clusterids_4096.ivecs \
    l2
```

## Architecture

### Core Module Organization
- **`src/lib.rs`**: Re-exports public API (`IvfRabitqIndex`, `SearchParams`, `SearchResult`, `QuantizedVector`, `RabitqConfig`, `Metric`)
- **`src/ivf.rs`**: IVF + RaBitQ index implementation
  - `IvfRabitqIndex::train()`: Train from scratch with built-in k-means
  - `IvfRabitqIndex::train_with_clusters()`: Train using pre-computed centroids and assignments (FAISS-compatible)
  - `IvfRabitqIndex::search()`: Performs approximate nearest-neighbor search with configurable `nprobe` and `top_k`
  - Serialization: `save()` and `load()` for index persistence with CRC32 checksums
- **`src/quantizer.rs`**: Core RaBitQ quantization logic
  - `quantize_with_centroid()`: Quantizes residual vectors (data - centroid) into compact codes with configurable `RabitqConfig`
  - Supports both 1-bit binary codes and extended multi-bit codes
  - Computes distance estimation factors (`f_add`, `f_rescale`) for L2 and inner-product metrics
- **`src/rotation.rs`**: Random orthonormal rotation matrix generation (Haar distribution)
- **`src/kmeans.rs`**: Lightweight k-means implementation for in-crate training
- **`src/math.rs`**: Low-level vector operations (dot product, L2 distance, subtraction, norms)
- **`src/io.rs`**: Parsers for `.fvecs` and `.ivecs` file formats (GIST/SIFT datasets)
- **`src/bin/ivf_rabitq.rs`**: CLI for benchmarking IVF + RaBitQ on any .fvecs dataset
  - Three modes: Index, Query, or IndexAndQuery (auto-detected from arguments)
  - `--benchmark` flag enables C++-style nprobe sweep + 5-round benchmark
  - Without `--benchmark`, evaluates at specific nprobe with detailed latency stats
- **`src/tests.rs`**: Regression tests with seeded RNGs for deterministic validation

### Training Workflows
1. **Pre-clustered (FAISS-compatible)**: Use `python/ivf.py` to generate centroids and assignments, then call `IvfRabitqIndex::train_with_clusters()`
2. **Fully Rust-based**: Call `IvfRabitqIndex::train()` with `nlist` parameter; k-means runs internally

### Distance Metrics
- **L2**: Euclidean distance
- **InnerProduct**: Maximum similarity (for cosine-like comparisons)

### Search Implementation
- **FastScan V2** (default): Optimized batch SIMD search **supporting all bit configurations**
  - **bits=1** (ex_bits=0, binary only): 5-10x faster than naive, maximum speed
  - **bits=3** (ex_bits=2): 3-5x faster than naive, good balance of speed and precision
  - **bits=7** (ex_bits=6, standard): 2-3x faster than naive, high precision
  - Uses quantized lookup tables (LUT) for fast binary code distance estimation
  - Extended codes (ex_bits > 0) computed exactly for final distance refinement
  - Binary code bit order: MSB-first (matches C++ reference implementation)
  - Processes vectors in batches of 32 for optimal SIMD performance
- **Performance vs Precision Trade-off**: Higher bits = better precision, slightly lower speedup
- **Recommendation**: Use bits=7 for production (best accuracy), bits=1 for maximum speed

### Search Parameters
- **`nprobe`**: Number of nearest clusters to probe during search (higher = better recall, slower)
- **`top_k`**: Number of nearest neighbors to return

### Quantization Details
- **`total_bits`**: Controls quantization granularity (typically 7 bits)
  - 1 bit encodes sign (binary_code)
  - Remaining bits encode magnitude refinement (ex_code)
- **Residual quantization**: Vectors are quantized relative to their cluster centroid
- **Distance estimation**: Pre-computed factors (`f_add`, `f_rescale`) enable fast approximate distance calculation from quantized codes
- **`faster_config`**: Optional optimization that precomputes a constant scaling factor
  - Default: compute optimal scaling factor per vector (slower, more accurate)
  - With `--faster-config`: use precomputed constant (100-500x faster, <1% accuracy loss)
  - Recommended for large datasets (>100K vectors)

## Data Files
- Large datasets live in `data/` (git-ignored)
- Use `--max-base` and `--max-queries` flags for CI-friendly smoke tests
- GIST dataset: 1M base vectors (960 dims), 1000 queries, ground truth for recall@100

## Testing Conventions
- All tests use seeded RNGs (`StdRng::seed_from_u64`) for reproducibility
- Cover both L2 and InnerProduct metrics when modifying quantizer or IVF logic
- Keep smoke tests fast with limited data caps

## Publishing to crates.io
1. Update `version` in `Cargo.toml` (semver)
2. Run `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test`
3. Create a new release tag (e.g., `v0.1.3`) and push to GitHub
4. The GitHub Actions workflow will automatically publish the new version to crates.io

## Commit Style
- Short Title Case subjects (e.g., "Rewrite K-Means Trainer")
- Keep under 72 characters
- Describe behavior changes, not implementation details
- RaBitQ 不是 Residual-Aware Bit Quantization 的缩写