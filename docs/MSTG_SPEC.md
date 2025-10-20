# MSTG: Multi-Scale Tree Graph Specification

## Overview

MSTG (Multi-Scale Tree Graph) is a hybrid memory-disk approximate nearest neighbor search (ANNS) system that combines the best aspects of three state-of-the-art techniques:

1. **SPANN's inverted index architecture** - Memory-efficient partitioning with disk-based posting lists
2. **RaBitQ quantization** - Compact vector compression with accurate distance estimation
3. **HNSW graph navigation** - Fast, high-quality graph-based centroid search

## Motivation

### Problems with Existing Approaches

**SPANN Limitations:**
- Stores full-precision vectors on disk → High disk I/O cost
- Uses SPTAG for centroid navigation → Less optimal than HNSW
- No compression within posting lists → Larger disk reads

**DiskANN Limitations:**
- Uses lossy PQ compression → Reduced recall quality
- Graph traversal requires many random disk accesses
- Difficult to tune for different recall/latency tradeoffs

**IVF-RaBitQ (Current Implementation) Limitations:**
- Simple flat IVF index → Poor scalability for billion-scale data
- No boundary vector handling → Lower recall on cluster boundaries
- Fixed nprobe parameter → Not query-adaptive

### MSTG Advantages

1. **Reduced Disk I/O**: RaBitQ compresses vectors 10-30x → Smaller posting lists
2. **Better Recall**: Closure assignment + accurate RaBitQ distance estimation
3. **Faster Centroid Search**: HNSW navigates centroids more efficiently than SPTAG
4. **Query-Adaptive**: Dynamic pruning adjusts search cost per query
5. **Scalable**: Hierarchical clustering handles billion-scale datasets

## Architecture

### Three-Tier Design

```
┌─────────────────────────────────────────────────────────┐
│                    TIER 1: MEMORY                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │         HNSW Graph (Centroid Navigation)        │   │
│  │  - Stores centroid representatives             │   │
│  │  - Fast approximate nearest centroid search    │   │
│  │  - ~10-15% of total vectors as centroids       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   TIER 2: METADATA                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Posting List Directory                 │   │
│  │  - Disk offsets for each posting list          │   │
│  │  - Posting list sizes                          │   │
│  │  - RaBitQ config per cluster                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    TIER 3: DISK                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         RaBitQ-Compressed Posting Lists         │   │
│  │  Posting List 1: [quantized vectors + metadata]│   │
│  │  Posting List 2: [quantized vectors + metadata]│   │
│  │  ...                                            │   │
│  │  Posting List N: [quantized vectors + metadata]│   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Centroid Navigation (HNSW)

**Purpose**: Fast retrieval of nearest cluster centroids

**Data Stored**:
- Centroid representative vectors (scalar quantized: bf16, fp32, fp16, or int8)
- HNSW graph structure (edges at multiple layers)
- Mapping: centroid_id → posting_list_id

**Why HNSW over SPTAG**:
- Better recall/QPS tradeoff
- More mature and widely tested
- Simpler parameter tuning (only ef_search)
- Better theoretical guarantees

**Scalar Quantization for Centroids**:

MSTG supports pluggable scalar quantization for centroid vectors to reduce memory:

| Precision | Bytes/dim | Memory (100K centroids, 960D) | Accuracy Loss |
|-----------|-----------|-------------------------------|---------------|
| fp32      | 4         | ~384 MB                      | 0% (baseline) |
| bf16      | 2         | ~192 MB                      | <0.1%         |
| fp16      | 2         | ~192 MB                      | <0.1%         |
| int8      | 1         | ~96 MB                       | ~1-2%         |

**Default: bf16** (2x memory savings with negligible accuracy loss)

**Why bf16 over fp16**:
- Same memory savings as fp16
- Better dynamic range (same exponent bits as fp32)
- Hardware accelerated on modern CPUs (AVX-512 BF16, ARM SVE2)
- More stable for distance computations (fewer overflows)

**Memory Cost** (with bf16):
- N_centroids vectors × dimension × 2 bytes (bf16)
- Graph edges: ~M × N_centroids × 4 bytes (M ≈ 16-32)
- For 1B vectors with 100K centroids (10%), 960 dims: ~192 MB vectors + ~50 MB graph = **~240 MB**
- **50% memory reduction** compared to fp32 (was 450 MB)

#### 2. Posting Lists (RaBitQ-Compressed)

**Structure of Each Posting List**:
```
PostingList {
    cluster_id: u32,
    centroid: Vec<f32>,              // Full-precision centroid
    size: u32,                        // Number of vectors
    rabitq_config: RabitqConfig,      // Per-cluster quantization params
    vectors: Vec<QuantizedVector>,    // RaBitQ codes
}

QuantizedVector {
    vector_id: u64,                   // Original vector ID
    binary_code: BitVec,              // Sign bits
    ex_code: Vec<u8>,                 // Magnitude bits
    norm: f32,                        // Vector norm
}
```

**Disk Layout**:
- Posting lists stored sequentially on disk
- Optional: Group nearby posting lists for better locality
- Each posting list is a self-contained unit (can be loaded independently)

**Compression Ratio**:
- Original: 960 × 4 bytes = 3840 bytes/vector
- RaBitQ (7 bits): ~120 bytes + overhead ≈ 140 bytes/vector
- Compression: **~27x**

#### 3. Metadata Index

**Posting List Directory**:
```rust
struct PostingListDirectory {
    entries: Vec<PostingListEntry>,
}

struct PostingListEntry {
    cluster_id: u32,
    centroid_id: u32,           // ID in HNSW graph
    disk_offset: u64,           // Byte offset in posting list file
    size_bytes: u32,            // Size on disk
    num_vectors: u32,           // Number of vectors in list
    avg_vector_norm: f32,       // For query-aware pruning
}
```

**Memory Cost**:
- ~32 bytes per posting list
- For 100K posting lists: ~3 MB

## Key Algorithms

### 1. Index Building

#### Phase 1: Hierarchical Balanced Clustering

```
Input: Dataset X (n vectors), max_posting_size L, branching_factor k
Output: N posting lists with balanced sizes

Algorithm:
1. Initialize: clusters = [X]  // Start with all data
2. While max(cluster.size) > L:
     For each large_cluster in clusters where size > L:
         // Balanced k-means with constraint
         subclusters = balanced_kmeans(large_cluster, k, balance_weight=λ)
         Replace large_cluster with subclusters
3. Final number of clusters N = len(clusters)
```

**Parameters**:
- `max_posting_size`: 5000-10000 vectors (target ~500KB-1MB compressed)
- `branching_factor`: 8-16 (SPANN uses ~10)
- `balance_weight`: 1.0 (equal weight between clustering quality and balance)

**Complexity**: O(n × d × k × log_k(N))

#### Phase 2: Closure Assignment (Boundary Expansion)

```
Input: Vectors X, centroids C, epsilon_1 (closure threshold)
Output: Expanded posting lists

Algorithm:
1. For each vector x in X:
     distances = [dist(x, c) for c in C]
     sorted_indices = argsort(distances)
     closest_dist = distances[sorted_indices[0]]

     assigned_clusters = []
     For i in sorted_indices:
         if distances[i] <= (1 + epsilon_1) × closest_dist:
             assigned_clusters.append(i)
         else:
             break

     // Apply RNG rule to reduce redundancy
     final_clusters = apply_rng_rule(assigned_clusters, distances)

     For cluster_id in final_clusters:
         posting_lists[cluster_id].append(x)

RNG Rule:
- Skip cluster_j if exists cluster_i where:
    dist(centroid_i, x) < dist(centroid_i, centroid_j)
- Ensures geometric diversity of assigned clusters
```

**Parameters**:
- `epsilon_1`: 0.1-0.2 (SPANN uses ~0.1 for recall@1)
- `max_replicas`: 8 (limit total copies per vector)

**Effect**:
- Increases posting list size by ~20-30%
- Significantly improves recall for boundary vectors

#### Phase 3: RaBitQ Quantization

```
Algorithm:
1. For each posting_list:
     centroid = posting_list.centroid
     vectors = posting_list.vectors

     // Compute residuals
     residuals = [v - centroid for v in vectors]

     // Train RaBitQ for this cluster
     config = RabitqConfig::train(
         residuals,
         total_bits=7,  // 1 sign + 6 magnitude
         metric=L2
     )

     // Quantize all vectors
     quantized = [config.quantize(r) for r in residuals]

     // Store
     posting_list.rabitq_config = config
     posting_list.quantized_vectors = quantized
```

**Key Decision**: Per-cluster vs. global quantization
- **Per-cluster** (recommended): Better accuracy, more metadata
- **Global**: Less metadata, slightly lower accuracy

#### Phase 4: HNSW Construction

```
Algorithm:
1. Extract centroid representatives:
     For each posting_list:
         rep = vector closest to centroid
         representatives.append(rep)

2. Build HNSW index:
     hnsw = HNSW::new(
         representatives,
         M=32,           // Max edges per layer
         ef_construction=200,
         distance=L2
     )

3. Store mapping:
     centroid_id → posting_list_id
```

### 2. Search Algorithm

#### Query Processing

```
Input: Query q, target_recall R, top_k
Output: k approximate nearest neighbors

Algorithm:
1. // Coarse search: Find candidate clusters
   ef_search = adaptive_ef(target_recall)  // e.g., 100-500
   centroid_candidates = hnsw.search(q, ef_search)

2. // Query-aware dynamic pruning
   epsilon_2 = get_pruning_threshold(target_recall)  // e.g., 0.6 for recall@1
   closest_dist = centroid_candidates[0].distance

   selected_clusters = []
   For (centroid_id, dist) in centroid_candidates:
       if dist <= (1 + epsilon_2) × closest_dist:
           selected_clusters.append(centroid_id)
       else:
           break

3. // Load posting lists from disk (parallel I/O)
   posting_lists = parallel_load(selected_clusters)

4. // Fine search: RaBitQ distance computation
   all_candidates = []
   For plist in posting_lists:
       centroid = plist.centroid
       config = plist.rabitq_config

       // Compute approximate distances using RaBitQ
       For qvec in plist.quantized_vectors:
           dist_approx = config.estimate_distance(
               q - centroid,  // Query residual
               qvec
           )
           all_candidates.append((qvec.vector_id, dist_approx))

5. // Top-k selection
   candidates = top_k_by_distance(all_candidates, top_k × 10)

6. // Optional: Rerank with full precision
   if use_reranking:
       full_vectors = load_full_vectors(candidates)
       results = rerank(q, full_vectors, top_k)
   else:
       results = candidates[:top_k]

   return results
```

#### Adaptive Parameter Selection

```rust
fn adaptive_ef(target_recall: f32) -> usize {
    // Empirically determined mapping
    match target_recall {
        r if r >= 0.99 => 500,
        r if r >= 0.95 => 300,
        r if r >= 0.90 => 150,
        r if r >= 0.80 => 100,
        _ => 50,
    }
}

fn get_pruning_threshold(target_recall: f32) -> f32 {
    // epsilon_2 for dynamic pruning
    // Higher recall → larger epsilon → more clusters
    if target_recall >= 0.95 {
        0.8  // Search more clusters
    } else if target_recall >= 0.90 {
        0.6
    } else {
        0.4  // Search fewer clusters
    }
}
```

## Implementation Plan

### Phase 1: Core Data Structures (Week 1-2)

**Files to Create**:
- `src/mstg/mod.rs` - Main module
- `src/mstg/posting_list.rs` - Posting list structure
- `src/mstg/metadata.rs` - Directory and metadata
- `src/mstg/hnsw_wrapper.rs` - HNSW integration

**Dependencies to Add**:
```toml
[dependencies]
hnsw = "0.11"  # Or hnsw_rs crate
rayon = "1.7"  # Parallel processing
memmap2 = "0.9"  # Memory-mapped disk I/O
half = "2.3"  # fp16 and bf16 support
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"  # Fast serialization
```

**Key Structures**:
```rust
pub struct MstgIndex {
    // Memory tier
    hnsw: Hnsw,
    centroid_to_posting: HashMap<u32, u32>,

    // Metadata tier
    directory: PostingListDirectory,

    // Disk tier
    posting_list_file: MmapMut,

    // Configuration
    config: MstgConfig,
}

pub struct MstgConfig {
    // Clustering
    max_posting_size: usize,
    branching_factor: usize,
    balance_weight: f32,

    // Closure assignment
    closure_epsilon: f32,
    max_replicas: usize,

    // RaBitQ
    rabitq_bits: usize,
    faster_config: bool,

    // HNSW
    hnsw_m: usize,
    hnsw_ef_construction: usize,

    // Scalar quantization for centroids
    centroid_precision: ScalarPrecision,

    // Search
    default_ef_search: usize,
    pruning_epsilon: f32,
}

/// Scalar quantization precision for centroid vectors in HNSW
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarPrecision {
    /// Full precision (4 bytes/dim)
    FP32,
    /// Brain floating point (2 bytes/dim, hardware accelerated)
    BF16,
    /// Half precision (2 bytes/dim)
    FP16,
    /// 8-bit integer quantization (1 byte/dim, requires training)
    INT8,
}

impl ScalarPrecision {
    pub fn bytes_per_dim(&self) -> usize {
        match self {
            ScalarPrecision::FP32 => 4,
            ScalarPrecision::BF16 => 2,
            ScalarPrecision::FP16 => 2,
            ScalarPrecision::INT8 => 1,
        }
    }

    pub fn memory_multiplier(&self) -> f32 {
        match self {
            ScalarPrecision::FP32 => 1.0,
            ScalarPrecision::BF16 => 0.5,
            ScalarPrecision::FP16 => 0.5,
            ScalarPrecision::INT8 => 0.25,
        }
    }
}
```

### Phase 2: Hierarchical Balanced Clustering (Week 3)

**Implementation**:
- Extend existing `src/kmeans.rs` with balanced constraint
- Add hierarchical wrapper
- Test on SIFT1M with various branching factors

**Algorithm**:
```rust
impl MstgIndex {
    fn hierarchical_balanced_clustering(
        &self,
        data: &[Vec<f32>],
        max_size: usize,
        k: usize,
    ) -> Vec<Cluster> {
        let mut clusters = vec![Cluster::new(data)];

        while clusters.iter().any(|c| c.size() > max_size) {
            let mut new_clusters = Vec::new();

            for cluster in clusters {
                if cluster.size() > max_size {
                    // Split using balanced k-means
                    let subclusters = balanced_kmeans(
                        cluster.data(),
                        k,
                        self.config.balance_weight,
                    );
                    new_clusters.extend(subclusters);
                } else {
                    new_clusters.push(cluster);
                }
            }

            clusters = new_clusters;
        }

        clusters
    }
}
```

### Phase 3: Closure Assignment with RNG (Week 4)

**Implementation**:
- Add closure assignment logic
- Implement RNG rule for diversity
- Measure posting list size increase

**Test**: Verify boundary vector recall improvement

### Phase 4: RaBitQ Integration (Week 5-6)

**Modifications**:
- Create per-cluster RaBitQ quantization
- Store quantized vectors in posting lists
- Implement distance estimation with cluster centroids

**Key Function**:
```rust
impl PostingList {
    fn quantize_vectors(
        &mut self,
        vectors: &[Vec<f32>],
        config: &RabitqConfig,
    ) {
        // Compute residuals
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| subtract(v, &self.centroid))
            .collect();

        // Train RaBitQ on residuals
        self.rabitq_config = RabitqConfig::train(
            &residuals,
            config.total_bits,
            Metric::L2,
        );

        // Quantize
        self.quantized_vectors = residuals
            .iter()
            .map(|r| self.rabitq_config.quantize(r))
            .collect();
    }
}
```

### Phase 5: HNSW Integration with Scalar Quantization (Week 7)

**Tasks**:
- Integrate HNSW library (likely `hnsw` or `hnsw_rs` crate)
- Implement scalar quantization layer (bf16, fp16, int8)
- Build HNSW on quantized centroid representatives
- Implement centroid search with quantized distance computation

**HNSW Selection**:
- Use `instant-distance` or `hnsw_rs` crate
- Requirements: Fast construction, L2 distance, tunable ef_search
- Must support custom distance functions for quantized vectors

**Scalar Quantization Implementation**:

The scalar quantization layer sits between the HNSW index and the original centroids:

```rust
/// Trait for scalar-quantized vectors
pub trait QuantizedVector: Clone + Send + Sync {
    /// Quantize a full-precision vector
    fn quantize(vector: &[f32]) -> Self;

    /// Dequantize back to fp32 for distance computation
    fn dequantize(&self) -> Vec<f32>;

    /// Compute distance directly in quantized space (faster)
    fn distance_to(&self, other: &Self) -> f32;

    /// Memory size in bytes
    fn memory_size(&self) -> usize;
}

/// BF16 (Brain Float16) quantized vector
#[derive(Clone)]
pub struct BF16Vector {
    data: Vec<u16>,  // BF16 encoded as u16
}

impl QuantizedVector for BF16Vector {
    fn quantize(vector: &[f32]) -> Self {
        let data = vector.iter().map(|&x| fp32_to_bf16(x)).collect();
        Self { data }
    }

    fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&x| bf16_to_fp32(x)).collect()
    }

    fn distance_to(&self, other: &Self) -> f32 {
        // Compute L2 distance in bf16 space with fp32 accumulator
        let mut sum = 0.0f32;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = bf16_to_fp32(*a) - bf16_to_fp32(*b);
            sum += diff * diff;
        }
        sum
    }

    fn memory_size(&self) -> usize {
        self.data.len() * 2
    }
}

/// FP16 quantized vector
#[derive(Clone)]
pub struct FP16Vector {
    data: Vec<half::f16>,
}

/// INT8 quantized vector (requires training for scale/offset)
#[derive(Clone)]
pub struct INT8Vector {
    data: Vec<i8>,
    scale: f32,
    offset: f32,
}

// Conversion functions
fn fp32_to_bf16(x: f32) -> u16 {
    // BF16: sign(1) + exponent(8) + mantissa(7)
    // Simply truncate fp32 mantissa from 23 to 7 bits
    (x.to_bits() >> 16) as u16
}

fn bf16_to_fp32(x: u16) -> f32 {
    // Expand bf16 back to fp32 by shifting and zero-padding mantissa
    f32::from_bits((x as u32) << 16)
}
```

### Phase 6: Disk I/O and Serialization (Week 8)

**Implementation**:
- Posting list serialization format
- Memory-mapped file I/O
- Parallel loading of multiple posting lists
- Checksum validation (CRC32, similar to IVF)

**Disk Format**:
```
[Header]
- Magic bytes: "MSTG"
- Version: u32
- Num posting lists: u32
- Directory offset: u64

[Posting Lists]
- Sequential posting list data

[Directory]
- Array of PostingListEntry
- CRC32 checksum
```

### Phase 7: Search Implementation (Week 9-10)

**Components**:
1. HNSW centroid search
2. Query-aware dynamic pruning
3. Parallel posting list loading
4. RaBitQ distance computation
5. Top-k aggregation

**Optimizations**:
- Batch distance computations
- SIMD for RaBitQ (reuse existing code)
- Parallel posting list processing

### Phase 8: Testing and Benchmarking (Week 11-12)

**Test Suite**:
1. **Unit tests**: Each component in isolation
2. **Integration tests**: End-to-end search
3. **Benchmark tests**: Compare against IVF-RaBitQ and DiskANN

**Datasets**:
- SIFT1M (development)
- GIST1M (validation)
- DEEP1B (final benchmark, if available)

**Metrics**:
- Recall@1, Recall@10, Recall@100
- Query latency (p50, p95, p99)
- Memory usage
- Disk I/O (number of posting lists loaded)
- Index build time

### Phase 9: CLI and Documentation (Week 13)

**Binary**:
- `src/bin/mstg.rs` - Similar to `ivf_rabitq.rs`
- Modes: build, search, benchmark

**Documentation**:
- Algorithm explanation
- Usage examples
- Performance tuning guide

## Experimental Validation

### Comparison Baselines

1. **IVF-RaBitQ** (current implementation)
2. **SPANN** (if available, or simulate with full-precision posting lists)
3. **DiskANN** (reference from paper)

### Expected Performance

**Compared to IVF-RaBitQ**:
- ✅ Better recall at same nprobe (due to closure assignment)
- ✅ Faster centroid search (HNSW vs linear scan)
- ✅ Query-adaptive (dynamic pruning)
- ⚠️ Slower index build (hierarchical clustering)
- ⚠️ Slightly more memory (HNSW graph)

**Compared to SPANN**:
- ✅ 10-30x less disk I/O (RaBitQ compression)
- ✅ Faster distance computation (quantized codes)
- ✅ Better recall/latency tradeoff
- ✅ Smaller memory footprint (compressed posting lists)

### Ablation Studies

Test contribution of each component:
1. **Baseline**: Simple IVF + RaBitQ
2. **+ Hierarchical clustering**: vs flat k-means
3. **+ Closure assignment**: Effect on recall
4. **+ RNG rule**: Effect on redundancy
5. **+ HNSW**: vs linear centroid search
6. **+ Dynamic pruning**: vs fixed nprobe

## Tuning Guidelines

### Index Build Parameters

**For 1M vectors**:
```rust
MstgConfig {
    max_posting_size: 5000,
    branching_factor: 10,
    closure_epsilon: 0.15,
    max_replicas: 8,
    rabitq_bits: 7,
    hnsw_m: 32,
    hnsw_ef_construction: 200,
    centroid_precision: ScalarPrecision::BF16,  // 2x memory savings
}
```

**For 100M+ vectors**:
```rust
MstgConfig {
    max_posting_size: 8000,
    branching_factor: 8,
    closure_epsilon: 0.10,
    max_replicas: 6,
    rabitq_bits: 7,
    faster_config: true,  // Use precomputed scaling
    hnsw_m: 32,
    hnsw_ef_construction: 400,
    centroid_precision: ScalarPrecision::BF16,  // 2x memory savings
}
```

**For memory-constrained environments (INT8)**:
```rust
MstgConfig {
    max_posting_size: 8000,
    branching_factor: 8,
    closure_epsilon: 0.10,
    max_replicas: 6,
    rabitq_bits: 7,
    faster_config: true,
    hnsw_m: 32,
    hnsw_ef_construction: 400,
    centroid_precision: ScalarPrecision::INT8,  // 4x memory savings
    // Note: INT8 may lose ~1-2% recall, benchmark before using
}
```

### Search Parameters

**High recall (95%+)**:
```rust
SearchParams {
    ef_search: 300,
    pruning_epsilon: 0.8,
    top_k: 100,
}
```

**Balanced (90% recall)**:
```rust
SearchParams {
    ef_search: 150,
    pruning_epsilon: 0.6,
    top_k: 100,
}
```

**Low latency (80% recall)**:
```rust
SearchParams {
    ef_search: 50,
    pruning_epsilon: 0.4,
    top_k: 100,
}
```

## Future Extensions

### 1. Distributed MSTG
- Partition posting lists across machines
- Use closure assignment for load balancing
- Query routing with HNSW

### 2. GPU Acceleration
- GPU-accelerated RaBitQ distance computation
- Batch query processing

### 3. Incremental Updates
- Online insertion into posting lists
- Periodic HNSW rebuilding
- Tombstone-based deletion

### 4. Hybrid Precision
- Store both RaBitQ codes and full vectors
- Use RaBitQ for filtering, full vectors for reranking

## References

1. SPANN Paper (NeurIPS 2021)
2. RaBitQ Paper (VLDB 2024)
3. HNSW Paper (TPAMI 2018)
4. DiskANN Paper (NeurIPS 2019)

## Open Questions

1. **Per-cluster vs global RaBitQ**: Which is better?
   - Hypothesis: Per-cluster is better (residuals are more uniform)

2. **HNSW parameters**: Optimal M and ef_construction?
   - Need empirical tuning on GIST/SIFT

3. **Closure epsilon**: How does it vary by dataset?
   - May need dataset-dependent defaults

4. **Reranking**: Is it necessary with RaBitQ?
   - RaBitQ is already quite accurate, may not need reranking

## Success Criteria

MSTG is successful if it achieves:

1. ✅ **90% recall@10** in <2ms on GIST1M (with ~100MB memory)
2. ✅ **2x better QPS** than IVF-RaBitQ at same recall
3. ✅ **10x less disk I/O** than SPANN at same recall
4. ✅ **Index build** completes in <30 minutes for GIST1M
5. ✅ **Memory usage** <20% of dataset size

---

**Document Version**: 1.0
**Author**: Claude Code
**Date**: 2025-10-20
