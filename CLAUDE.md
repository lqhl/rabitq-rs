# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This is a pure-Rust implementation of the RaBitQ quantization scheme with two search backends:
- **IVF (Inverted File)**: Flat clustering for in-memory search, suitable for <100M vectors
- **MSTG (Multi-Scale Tree Graph)**: Hierarchical clustering + HNSW for disk-based search, designed for >100M vectors

The project mirrors the behavior of the C++ RaBitQ Library and provides tools to reproduce the GIST benchmark pipeline.

## Key Commands

### Build, Test, and Lint
```bash
cargo fmt                                    # Format code (run before commits)
cargo clippy --all-targets --all-features    # Lint (must pass cleanly)
cargo test                                   # Run unit and integration tests
```

### Benchmark Evaluation

#### IVF-RaBitQ (ivf_rabitq binary)
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

#### MSTG-RaBitQ (mstg_rabitq binary)
```bash
# Build and benchmark in one command
cargo run --release --bin mstg_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --bits 7 \
    --k 16 \
    --target-cluster-size 100 \
    --benchmark \
    --faster-config

# Quick smoke test
cargo run --release --bin mstg_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --max-base 10000 \
    --max-queries 100 \
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
- **`src/lib.rs`**: Re-exports public API (`IvfRabitqIndex`, `MstgRabitqIndex`, `SearchParams`, `SearchResult`, `QuantizedVector`, `RabitqConfig`, `Metric`)
- **`src/ivf.rs`**: IVF + RaBitQ index implementation (flat clustering, in-memory)
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
- **`src/mstg.rs`**: MSTG + RaBitQ index implementation (hierarchical clustering, disk-ready)
  - `MstgRabitqIndex::train()`: Build multi-level k-means tree + HNSW graph on medoids
  - `MstgRabitqIndex::search()`: HNSW-accelerated cluster selection + RaBitQ scan
  - Designed for large-scale data with on-disk cluster storage (future)
  - See `docs/MSTG_DESIGN.md` for detailed architecture
- **`src/bin/ivf_rabitq.rs`**: CLI for benchmarking IVF + RaBitQ
  - Three modes: Index, Query, or IndexAndQuery
  - `--benchmark` flag enables nprobe sweep + 5-round benchmark
- **`src/bin/mstg_rabitq.rs`**: CLI for benchmarking MSTG + RaBitQ
  - Builds multi-level tree with configurable branching factor (`--k`) and cluster size (`--target-cluster-size`)
  - Same benchmark mode as IVF for direct comparison
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
- **`faster_config`**: Optional optimization that precomputes a constant scaling factor
  - Default: compute optimal scaling factor per vector (slower, more accurate)
  - With `--faster-config`: use precomputed constant (100-500x faster, <1% accuracy loss)
  - Recommended for large datasets (>100K vectors)

## Index Selection Guide

### IVF-RaBitQ
- **Configuration**: `nlist ≈ sqrt(N) × 2` (e.g., 2000 for 1M vectors)
- **Best for**: <100M vectors, all in-memory, ultra-low latency (<5ms)
- **Memory**: Stores all centroids + quantized codes in RAM
- **Production ready**: Yes

### MSTG-RaBitQ
- **Configuration**: `target_cluster_size = 100` → `num_clusters ≈ N / 100` (e.g., 10,000 for 1M vectors)
- **Best for**: >100M vectors, disk-based storage, cost-sensitive deployments
- **Memory**: Only medoids + HNSW graph in RAM (~500MB for 100M vectors), clusters on SSD
- **Production ready**: Experimental (in-memory prototype complete, disk-based in development)
- **Key advantage**: Can serve 100M vectors with <2GB RAM using NVMe SSD

See `docs/MSTG_DESIGN.md` for detailed comparison and roadmap.

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