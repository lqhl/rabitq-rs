# RaBitQ-rs

[![Crates.io](https://img.shields.io/crates/v/rabitq-rs.svg)](https://crates.io/crates/rabitq-rs)
[![Documentation](https://docs.rs/rabitq-rs/badge.svg)](https://docs.rs/rabitq-rs)
[![Build Status](https://github.com/lqhl/rabitq-rs/workflows/CI/badge.svg)](https://github.com/lqhl/rabitq-rs/actions)
[![License](https://img.shields.io/crates/l/rabitq-rs.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

Pure Rust implementation of advanced vector quantization and search algorithms:
- **RaBitQ quantization** with IVF (Inverted File) search - Production ready
- **MSTG (Multi-Scale Tree Graph)** index - ⚠️ Experimental (v0.5.0)

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

## MSTG: Multi-Scale Tree Graph Index (⚠️ Experimental)

**New in v0.5.0**: MSTG is an experimental hybrid index combining hierarchical clustering with graph navigation.

> **⚠️ Experimental Status**: MSTG is under active development and lacks comprehensive testing and tuning. For production workloads, we recommend using the battle-tested **IVF+RaBitQ** index. MSTG is provided for research and experimentation purposes.

### Key Features

- **Hierarchical Balanced Clustering**: Adaptive multi-scale clustering with configurable branching factor
- **Closure Assignment**: Vectors can belong to multiple clusters (RNG rule) for better recall
- **HNSW Graph Navigation**: Fast approximate nearest centroid search
- **Flexible Precision**: FP32, BF16, FP16, or INT8 for centroids
- **Dynamic Pruning**: Adaptive cluster selection based on query-centroid distances
- **Python Bindings**: Seamless integration with NumPy and ann-benchmarks

### When to Use MSTG vs IVF+RaBitQ

| Feature | IVF+RaBitQ | MSTG (Experimental) |
|---------|------------|---------------------|
| **Maturity** | ✅ Production Ready | ⚠️ Experimental |
| **Testing** | ✅ Comprehensive | ⚠️ Limited |
| **Best For** | All use cases (10K-10M+) | Research & experimentation |
| **Recall** | 85-95% (well-tuned) | 90-98% (theoretical) |
| **Build Time** | Fast (k-means) | Moderate (hierarchical) |
| **Query Speed** | Fast | Very Fast (HNSW navigation) |
| **Memory** | Low | Low-Medium (configurable) |
| **Stability** | ✅ Stable | ⚠️ API may change |
| **Documentation** | ✅ Complete | 🚧 In progress |

**Recommendation**: **Use IVF+RaBitQ for production workloads**. It's well-tested, documented, and optimized. Consider MSTG only if you're researching advanced indexing techniques or willing to help with testing and tuning.

## Platform Support

**x86_64 (Intel/AMD)**: ✅ Fully supported and tested  
**ARM64 (Apple Silicon, ARM servers)**: ⚠️ **Known issues** - See [ARM64_KNOWN_ISSUES.md](ARM64_KNOWN_ISSUES.md)

> **ARM64 Notice**: The current implementation has critical bugs on ARM64/AArch64 platforms that affect both L2 and Inner Product metrics. Distance calculations produce incorrect results, making the library unreliable on Apple Silicon Macs and ARM servers. We're actively investigating these issues. For now, **production use on ARM64 is NOT recommended**.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rabitq-rs = "0.5"
```

### IVF+RaBitQ Index (Recommended for Production)

Build an IVF+RaBitQ index and search in ~10 lines:

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

### MSTG Index (⚠️ Experimental - For Research Only)

Build an MSTG index and search in ~10 lines:

```rust
use rabitq_rs::mstg::{MstgIndex, MstgConfig, SearchParams};
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;

    // Generate 10,000 random vectors
    let dataset: Vec<Vec<f32>> = (0..10_000)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Build index with default configuration
    let index = MstgIndex::build(&dataset, MstgConfig::default())?;

    // Search: balanced mode with top 10 results
    let params = SearchParams::balanced(10);
    let results = index.search(&dataset[0], &params);

    println!("Top neighbor ID: {}, distance: {:.6}", results[0].vector_id, results[0].distance);
    Ok(())
}
```

> **⚠️ Note**: MSTG is experimental and not recommended for production use. Use IVF+RaBitQ for stable, well-tested performance.

## Usage

### IVF+RaBitQ Index

#### Training with Built-in K-Means

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

#### Training with Pre-computed Clusters (FAISS-Compatible)

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

#### Using Faster Config for Large Datasets

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

#### Searching

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

#### Saving and Loading

```rust
// Save index to disk (with CRC32 checksum)
index.save("my_index.bin")?;

// Load index from disk
let loaded_index = IvfRabitqIndex::load("my_index.bin")?;
```

### MSTG Index (⚠️ Experimental)

#### Building an MSTG Index

```rust
use rabitq_rs::mstg::{MstgIndex, MstgConfig, ScalarPrecision};
use rabitq_rs::Metric;

let mut config = MstgConfig::default();

// Configure clustering
config.max_posting_size = 5000;     // Max vectors per cluster
config.branching_factor = 10;       // Hierarchical branching
config.balance_weight = 1.0;        // Balance vs purity trade-off

// Configure closure assignment (multi-assignment)
config.closure_epsilon = 0.15;      // Distance threshold
config.max_replicas = 8;            // Max assignments per vector

// Configure quantization
config.rabitq_bits = 7;             // 3-8 bits (7 recommended)
config.faster_config = true;        // Fast mode
config.metric = Metric::L2;         // or Metric::InnerProduct

// Configure HNSW graph
config.hnsw_m = 32;                 // Connectivity
config.hnsw_ef_construction = 200;  // Build quality
config.centroid_precision = ScalarPrecision::BF16; // FP32/BF16/FP16/INT8

let index = MstgIndex::build(&vectors, config)?;
```

#### Searching

```rust
use rabitq_rs::mstg::SearchParams;

// Option 1: Pre-defined modes
let params = SearchParams::high_recall(100);  // 95%+ recall
let params = SearchParams::balanced(100);     // ~90% recall (default)
let params = SearchParams::low_latency(100);  // ~80% recall, fastest

// Option 2: Custom parameters
let params = SearchParams::new(
    150,   // ef_search: candidates to explore
    0.6,   // pruning_epsilon: dynamic pruning threshold
    100,   // top_k: number of results
);

let results = index.search(&query_vector, &params);

for result in results.iter().take(10) {
    println!("ID: {}, Distance: {:.6}", result.vector_id, result.distance);
}
```

**Parameter Tuning**:
- `ef_search`: Higher = better recall, slower (typical: 50-300)
- `pruning_epsilon`: Higher = more clusters searched (typical: 0.4-0.8)
- `max_posting_size`: Smaller = more clusters, better balance (typical: 3000-10000)
- `rabitq_bits`: More bits = higher accuracy (typical: 5-7)

> **⚠️ Reminder**: MSTG is experimental. For production use, refer to the IVF+RaBitQ documentation above.

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

## Python Bindings

RaBitQ-RS provides Python bindings for the MSTG (Multi-Scale Tree Graph) index, enabling seamless integration with Python machine learning workflows and ann-benchmarks.

### Installation

```bash
# Install dependencies
pip install maturin numpy

# Build and install from source
maturin develop --release --features python
```

### Quick Start (Python)

```python
import numpy as np
from rabitq_rs import MstgIndex

# Generate sample data
data = np.random.randn(10000, 128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

# Create and build index
index = MstgIndex(
    dimension=128,
    metric="euclidean",          # or "angular"
    max_posting_size=5000,
    rabitq_bits=7,               # 3-8 bits
    faster_config=True,          # Fast quantization mode
    centroid_precision="bf16"    # "fp32", "bf16", "fp16", or "int8"
)

index.fit(data)

# Search
index.set_query_arguments(ef_search=150, pruning_epsilon=0.6)
results = index.query(query, k=10)

# Results: numpy array of shape (k, 2) with [id, distance]
for neighbor_id, distance in results:
    print(f"ID: {int(neighbor_id)}, Distance: {distance:.6f}")
```

### Configuration Parameters

**Index Parameters (Build Time)**:
- `dimension`: Vector dimensionality (required)
- `metric`: Distance metric - "euclidean" (L2) or "angular" (inner product)
- `max_posting_size`: Maximum vectors per cluster (default: 5000)
- `rabitq_bits`: Quantization bits, 3-8 (default: 7)
- `faster_config`: Fast quantization mode (default: True)
- `centroid_precision`: Centroid storage - "fp32", "bf16", "fp16", "int8" (default: "bf16")
- `hnsw_m`: HNSW connectivity (default: 32)
- `hnsw_ef_construction`: HNSW build quality (default: 200)

**Query Parameters (Search Time)**:
- `ef_search`: Centroid candidates to explore (default: 150)
- `pruning_epsilon`: Dynamic pruning threshold (default: 0.6)
- `k`: Number of neighbors to return

### Batch Queries

```python
# Query multiple vectors efficiently
queries = np.random.randn(100, 128).astype(np.float32)
results_list = index.batch_query(queries, k=10)

# Returns list of numpy arrays, one per query
for i, results in enumerate(results_list):
    print(f"Query {i}: Top neighbor ID = {int(results[0, 0])}")
```

### Memory Usage

```python
memory_bytes = index.get_memory_usage()
print(f"Index size: {memory_bytes / 1024**2:.2f} MB")
```

### ann-benchmarks Integration

RaBitQ-RS includes a complete ann-benchmarks wrapper for standardized evaluation:

```bash
# Run benchmarks
cd ann_benchmarks
python run.py --algorithm rabitq-mstg --dataset fashion-mnist-784-euclidean
```

See `ann_benchmarks/README.md` for detailed benchmark configuration and usage.

## Testing

Run the test suite before committing changes:

```bash
# Rust tests
cargo fmt
cargo clippy --all-targets --all-features
cargo test

# Python tests
python test_python_bindings.py
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
