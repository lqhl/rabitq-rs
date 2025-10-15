# RaBitQ-rs

[![Crates.io](https://img.shields.io/crates/v/rabitq-rs.svg)](https://crates.io/crates/rabitq-rs)
[![Documentation](https://docs.rs/rabitq-rs/badge.svg)](https://docs.rs/rabitq-rs)
[![Build Status](https://github.com/lqhl/rabitq-rs/workflows/CI/badge.svg)](https://github.com/lqhl/rabitq-rs/actions)
[![License](https://img.shields.io/crates/l/rabitq-rs.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

Pure Rust implementation of the RaBitQ vector quantization algorithm with IVF (Inverted File) search capabilities.

**Up to 32× memory compression** with significantly higher accuracy than traditional Product Quantization (PQ) or Scalar Quantization (SQ).

## Why RaBitQ?

RaBitQ delivers exceptional compression-accuracy trade-offs for approximate nearest neighbor search:

- **vs. Raw Vectors**: Achieve **up to 32× memory reduction** with configurable compression ratios
- **vs. PQ/SQ**: Substantially higher accuracy, especially at high recall targets (90%+)
- **How it Works**: Combines 1-bit binary codes with multi-bit refinement codes for precise distance estimation

The quantization applies to residual vectors (data - centroid), enabling accurate reconstruction even at extreme compression rates.

## Why This Rust Implementation?

This library provides a **feature-complete** RaBitQ + IVF search engine with all optimizations from the original C++ implementation:

- ✅ **Faster Config Mode**: 100-500× faster index building with <1% accuracy loss
- ✅ **FHT Rotation**: Fast Hadamard Transform for efficient orthonormal rotations
- ✅ **SIMD Accelerated**: Optimized distance computations on modern CPUs
- ✅ **Memory Safe**: No segfaults, no unsafe code in dependencies
- ✅ **Complete IVF Pipeline**: Training (built-in k-means or FAISS-compatible), search, persistence

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rabitq-rs = "0.3"
```

Build an index and search in ~10 lines:

```rust
use rabitq_rs::{IvfRabitqIndex, SearchParams, Metric, RotatorType};
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;

    // Generate 10,000 random vectors
    let dataset: Vec<Vec<f32>> = (0..10_000)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Train index with 256 clusters, 7-bit quantization
    let index = IvfRabitqIndex::train(
        &dataset,
        256,                         // nlist (number of clusters)
        7,                           // total_bits (1 sign + 6 magnitude)
        Metric::L2,
        RotatorType::FhtKacRotator,  // Fast Hadamard Transform
        42,                          // random seed
        false,                       // use_faster_config
    )?;

    // Search: probe 32 clusters, return top 10 neighbors
    let params = SearchParams::new(10, 32);
    let results = index.search(&dataset[0], params)?;

    println!("Top neighbor ID: {}, distance: {}", results[0].id, results[0].score);
    Ok(())
}
```

## Usage

### Training with Built-in K-Means

```rust
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType};

let index = IvfRabitqIndex::train(
    &vectors,
    1024,                        // nlist: number of IVF clusters
    7,                           // total_bits: quantization precision
    Metric::L2,                  // or Metric::InnerProduct
    RotatorType::FhtKacRotator,
    12345,                       // seed for reproducibility
    false,                       // use_faster_config
)?;
```

### Training with Pre-computed Clusters (FAISS-Compatible)

If you already have centroids and cluster assignments from FAISS or another tool:

```rust
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType};

let index = IvfRabitqIndex::train_with_clusters(
    &vectors,
    &centroids,      // shape: [nlist, dim]
    &assignments,    // shape: [num_vectors], cluster ID per vector
    7,               // total_bits
    Metric::L2,
    RotatorType::FhtKacRotator,
    42,              // seed
    false,           // use_faster_config
)?;
```

### Using Faster Config for Large Datasets

For datasets >100K vectors, enable `faster_config` to accelerate training by **100-500×** with minimal accuracy loss (<1%):

```rust
let index = IvfRabitqIndex::train(
    &vectors,
    4096,
    7,
    Metric::L2,
    RotatorType::FhtKacRotator,
    42,
    true,  // use_faster_config = true
)?;
```

### Searching

```rust
use rabitq_rs::SearchParams;

let params = SearchParams::new(
    100,   // top_k: return 100 nearest neighbors
    64,    // nprobe: search 64 nearest clusters
);

let results = index.search(&query_vector, params)?;

for result in results.iter().take(10) {
    println!("ID: {}, Distance: {}", result.id, result.score);
}
```

**Parameter Tuning**:
- `nprobe`: Higher = better recall, slower search (typical: 10-1024)
- `top_k`: Number of neighbors to return
- `total_bits`: More bits = higher accuracy, larger index (typical: 3-7)

### Saving and Loading

```rust
// Save index to disk (with CRC32 checksum)
index.save("my_index.bin")?;

// Load index from disk
let loaded_index = IvfRabitqIndex::load("my_index.bin")?;
```

## CLI Tool: Benchmark on GIST-1M

The `ivf_rabitq` binary lets you evaluate RaBitQ on standard datasets like GIST-1M:

### 1. Download GIST-1M Dataset

```bash
mkdir -p data/gist
wget -P data/gist ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf data/gist/gist.tar.gz -C data/gist
```

### 2. Build Index

```bash
cargo run --release --bin ivf_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --nlist 4096 \
    --bits 3 \
    --faster-config \
    --save data/gist/index.bin
```

### 3. Run Benchmark

```bash
cargo run --release --bin ivf_rabitq -- \
    --load data/gist/index.bin \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --benchmark
```

This performs an automatic `nprobe` sweep and reports:

```
nprobe | QPS     | Recall@100
-------|---------|------------
5      | 12500   | 45.2%
10     | 8200    | 62.8%
20     | 5100    | 75.4%
...
```

### Alternative: Pre-cluster with FAISS

For C++ library compatibility, generate clusters using the provided Python helper:

```bash
python python/ivf.py \
    data/gist/gist_base.fvecs \
    4096 \
    data/gist/gist_centroids_4096.fvecs \
    data/gist/gist_clusterids_4096.ivecs \
    l2

cargo run --release --bin ivf_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --centroids data/gist/gist_centroids_4096.fvecs \
    --assignments data/gist/gist_clusterids_4096.ivecs \
    --bits 3 \
    --save data/gist/index.bin
```

## Testing

Run the test suite before committing changes:

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```

All tests use seeded RNGs for reproducible results.

## Citation

If you use RaBitQ in your research, please cite the original paper:

```bibtex
@inproceedings{guo2024rabitq,
  title={RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search},
  author={Guo, Jianyang and Wang, Zihan and Tang, Yun and Wang, Jihao and Cai, Ruihang and Cai, Peng and Zhang, Xiyuan and Wang, Jianye and Zhao, Jun},
  booktitle={Proceedings of the ACM on Management of Data},
  year={2024}
}
```

**Original C++ Implementation**: [VectorDB-NTU/RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
