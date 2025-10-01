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
```

### Benchmark Evaluation (GIST binary)
```bash
# Full benchmark run
cargo run --release --bin gist -- \
    --base data/gist/gist_base.fvecs \
    --centroids data/gist/gist_centroids_4096.fvecs \
    --assignments data/gist/gist_clusterids_4096.ivecs \
    --queries data/gist/gist_query.fvecs \
    --groundtruth data/gist/gist_groundtruth.ivecs \
    --bits 7 \
    --top-k 100 \
    --nprobe 1024 \
    --metric l2 \
    --seed 1337

# Smoke test with limited data
cargo run --release --bin gist -- \
    --base data/gist/gist_base.fvecs \
    --nlist 4096 \
    --queries data/gist/gist_query.fvecs \
    --groundtruth data/gist/gist_groundtruth.ivecs \
    --bits 7 \
    --top-k 100 \
    --nprobe 1024 \
    --metric l2 \
    --max-base 1000 \
    --max-queries 50

# Save/load index for repeated benchmarks
cargo run --release --bin gist -- \
    --base data/gist/gist_base.fvecs \
    --nlist 4096 \
    --bits 7 \
    --metric l2 \
    --save-index data/gist/gist_rbq.idx \
    ...

cargo run --release --bin gist -- \
    --load-index data/gist/gist_rbq.idx \
    --queries data/gist/gist_query.fvecs \
    ...
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
  - `quantize_with_centroid()`: Quantizes residual vectors (data - centroid) into compact codes
  - Supports both 1-bit binary codes and extended multi-bit codes
  - Computes distance estimation factors (`f_add`, `f_rescale`) for L2 and inner-product metrics
- **`src/rotation.rs`**: Random orthonormal rotation matrix generation (Haar distribution)
- **`src/kmeans.rs`**: Lightweight k-means implementation for in-crate training
- **`src/math.rs`**: Low-level vector operations (dot product, L2 distance, subtraction, norms)
- **`src/io.rs`**: Parsers for `.fvecs` and `.ivecs` file formats (GIST/SIFT datasets)
- **`src/bin/gist.rs`**: CLI for benchmarking IVF + RaBitQ on GIST dataset
- **`src/tests.rs`**: Regression tests with seeded RNGs for deterministic validation

### Training Workflows
1. **Pre-clustered (FAISS-compatible)**: Use `python/ivf.py` to generate centroids and assignments, then call `IvfRabitqIndex::train_with_clusters()`
2. **Fully Rust-based**: Call `IvfRabitqIndex::train()` with `nlist` parameter; k-means runs internally

### Distance Metrics
- **L2**: Euclidean distance
- **InnerProduct**: Maximum similarity (for cosine-like comparisons)

### Search Parameters
- **`nprobe`**: Number of nearest clusters to probe during search (higher = better recall, slower)
- **`top_k`**: Number of nearest neighbors to return

### Quantization Details
- **`total_bits`**: Controls quantization granularity (typically 7 bits)
  - 1 bit encodes sign (binary_code)
  - Remaining bits encode magnitude refinement (ex_code)
- **Residual quantization**: Vectors are quantized relative to their cluster centroid
- **Distance estimation**: Pre-computed factors (`f_add`, `f_rescale`) enable fast approximate distance calculation from quantized codes

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
2. Run `cargo fmt && cargo clippy --all-targets --all-features && cargo test`
3. Run `cargo package` to validate bundle
4. Run `cargo publish` to release

## Commit Style
- Short Title Case subjects (e.g., "Rewrite K-Means Trainer")
- Keep under 72 characters
- Describe behavior changes, not implementation details