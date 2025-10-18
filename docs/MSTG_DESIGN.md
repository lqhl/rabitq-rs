# MSTG-RaBitQ Design Document

## Overview

MSTG (Multi-Scale Tree Graph) is a hierarchical clustering approach combined with HNSW graph indexing, designed for **disk-based large-scale vector search**. Unlike traditional IVF which uses flat clustering, MSTG uses multi-level k-means trees with many small clusters that can be stored on NVMe SSD while keeping only metadata in memory.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────┐
│         MSTG-RaBitQ Index                   │
├─────────────────────────────────────────────┤
│                                             │
│  In-Memory (Light):                         │
│  ├─ Medoids (cluster representatives)       │
│  ├─ HNSW Graph (cluster selection)          │
│  └─ Cluster metadata                        │
│                                             │
│  On-Disk (Heavy):                           │
│  └─ Quantized vectors per cluster           │
│                                             │
└─────────────────────────────────────────────┘
```

### Multi-Level K-Means Tree

- **Branching factor**: k (default: 16)
- **Target cluster size**: 100 vectors per cluster
- **Tree depth**: Adaptively determined by data size
- **Construction**: Recursive splitting with early stopping

For N vectors:
- Target clusters: `N / target_cluster_size`
- IVF equivalent: `sqrt(N) × 2` (much fewer)

Example for 1M vectors:
- **MSTG**: ~10,000 clusters × 100 vectors
- **IVF**: ~2,000 clusters × 500 vectors

### HNSW Graph Index

- Built on cluster **medoids** (representative vectors)
- Provides O(log n) cluster selection vs IVF's O(n)
- Configuration:
  - `ef_construction`: 200 (build quality)
  - `ef_search`: 400 (query quality)
  - `M`: 16 (connectivity)

### RaBitQ Quantization

Each cluster stores vectors quantized with RaBitQ:
- Binary codes + extended codes
- Residual quantization relative to cluster medoid
- Distance estimation factors pre-computed

## Memory Footprint

### 1M Vectors (960 dim)

**IVF (nlist=2000, all in-memory)**:
```
Centroids:    2,000 × 960 × 4B = 7.3 MB
Codes:        1M × ~200 bytes = 200 MB
Total:        ~210 MB (all in memory)
```

**MSTG (10K clusters, disk-based)**:
```
In-Memory:
  Medoids:    10,000 × 960 × 4B = 36.8 MB
  HNSW:       ~8 MB (graph structure)
  Metadata:   ~1 MB
  Subtotal:   ~46 MB (resident)

On-Disk:
  Clusters:   10,000 × 20 KB = 200 MB

Query-time (nprobe=50):
  Loaded:     50 × 20 KB = 1 MB
  Total:      46 MB + 1 MB = 47 MB
```

### 100M Vectors (Scaling)

**IVF (nlist=20K)**:
```
Memory:       ~20 GB (all in-memory)
Limitation:   Requires high-memory machine
```

**MSTG (100K clusters)**:
```
Resident:     ~500 MB (medoids + HNSW)
Disk:         ~2 GB (cluster data)
Query-time:   500 MB + (nprobe × 20 KB)
Advantage:    Can serve 100M vectors with <2GB RAM
```

## Query Process

### Current (In-Memory Demo)

```rust
pub fn search(&self, query: &[f32], params: MstgSearchParams)
    -> Result<Vec<MstgSearchResult>, RabitqError>
{
    // 1. Rotate query
    let rotated = self.rotator.rotate(query);

    // 2. HNSW search for nearest clusters
    let cluster_ids = self.hnsw.search(&query_medoid, nprobe);

    // 3. Scan selected clusters (all in memory)
    for cluster_id in cluster_ids {
        let cluster = &self.clusters[cluster_id];
        // Compute distances using RaBitQ codes
    }

    // 4. Return top-k results
}
```

### Future (Disk-Based)

```rust
pub fn search_disk_based(&self, query: &[f32], params: MstgSearchParams)
    -> Result<Vec<MstgSearchResult>, RabitqError>
{
    // 1. Rotate query (in-memory, fast)
    let rotated = self.rotator.rotate(query);

    // 2. HNSW search (in-memory, O(log n))
    let cluster_ids = self.hnsw.search(&query_medoid, nprobe);

    // 3. Batch load clusters from SSD (I/O)
    let clusters = self.load_clusters_from_disk(&cluster_ids)?;

    // 4. Scan loaded clusters (in-memory)
    let results = self.search_in_clusters(&clusters, query, top_k);

    // 5. Release cluster memory
    Ok(results)
}
```

## Performance Analysis

### Training Time (1M vectors)

**IVF (nlist=2000)**:
- K-means: ~270s (single large clustering)
- Quantization: ~5s
- Total: ~275s

**MSTG (10K clusters)**:
- Multi-level k-means: ~180-240s (many small clusterings)
- HNSW build: ~10-20s
- Quantization: ~5s
- Total: ~200-270s (similar to IVF)

### Query Latency (nprobe=50)

**IVF (in-memory)**:
```
Centroid scan:   2,000 × 1 μs = 2 ms
Cluster scan:    50 × 500 × 0.5 μs = 12.5 ms
Total:           ~14.5 ms
```

**MSTG (disk-based, NVMe)**:
```
HNSW search:     log(10K) × 10 μs = 130 μs
Disk load:       50 × 20 KB @ 7GB/s = 143 μs (sequential)
                 or 50 × 100 μs = 5 ms (random)
Cluster scan:    50 × 100 × 0.5 μs = 2.5 ms
Total:           ~2.8 ms (seq) or ~7.6 ms (random)
```

**With Cache (80% hit rate)**:
```
Disk load:       10 × 100 μs = 1 ms
Total:           ~3.6 ms
```

## Implementation Roadmap

### Phase 1: In-Memory Prototype ✅

- [x] Multi-level k-means tree construction
- [x] HNSW graph on medoids
- [x] RaBitQ quantization per cluster
- [x] Search implementation
- [x] Serialization/deserialization
- [x] Fix cluster count calculation bug
- [x] Add ef_search parameter

### Phase 2: Disk-Based Prototype (Future)

- [ ] Separate cluster data from index metadata
- [ ] Implement cluster file format
- [ ] On-demand cluster loading
- [ ] Batch I/O for multiple clusters
- [ ] Benchmark disk vs memory overhead

### Phase 3: Production Optimizations (Future)

- [ ] LRU cache for hot clusters
- [ ] Asynchronous prefetching
- [ ] Cluster compression (zstd)
- [ ] Multi-threaded I/O
- [ ] HNSW serialization support

## Key Fixes Applied

### 1. Cluster Count Explosion (Critical)

**Problem**: Creating 65,536 clusters instead of target 10,000

```rust
// Before (wrong):
let depth = ((target as f64).ln() / (k as f64).ln()).ceil() as usize;
let actual = k.pow(depth as u32);  // k^4 = 16^4 = 65,536

// After (fixed):
// Early stopping before overshooting
if current_level.len() >= target / 2
   && estimated_next > target * 2 {
    break;
}
```

**Impact**: 6x faster training, much lower memory usage

### 2. HNSW Search Quality

**Problem**: ef_search not configured (default=1, too low)

```rust
// Added to MstgConfig:
pub hnsw_ef_search: usize,  // default: 400

// Passed to builder:
Builder::default()
    .ef_construction(ef_construction)
    .ef_search(ef_search)  // Now configurable
    .build(medoids, cluster_ids)
```

**Impact**: Better recall at low nprobe values

## Use Cases

### When to Use MSTG

✅ **Large-Scale Data (>100M vectors)**
- Memory limited, disk abundant
- Cost-sensitive deployments

✅ **Medium Latency Tolerance (3-10ms)**
- Can trade latency for scale
- Batch/offline workloads

✅ **Non-Uniform Access Patterns**
- LRU cache helps hot clusters
- HNSW adapts to data distribution

### When to Use IVF

✅ **Small-Medium Scale (<100M vectors)**
- Can fit in memory
- Simpler deployment

✅ **Ultra-Low Latency (<3ms)**
- No disk I/O overhead
- Predictable performance

✅ **Production Stability**
- Well-tested, mature
- Minimal dependencies

## Configuration Guidelines

### Cluster Count Selection

```rust
// For N vectors:
let target_cluster_size = 100;  // Sweet spot for disk I/O
let num_clusters = N / target_cluster_size;

// Examples:
// 1M vectors    → 10,000 clusters
// 10M vectors   → 100,000 clusters
// 100M vectors  → 1,000,000 clusters
```

### HNSW Parameters

```rust
MstgConfig {
    k: 16,                      // Tree branching factor
    target_cluster_size: 100,   // Vectors per cluster
    hnsw_m: 16,                 // HNSW connectivity
    hnsw_ef_construction: 200,  // Build quality
    hnsw_ef_search: 400,        // Query quality (>= max nprobe)
    ..Default::default()
}
```

### Query Parameters

```rust
MstgSearchParams {
    top_k: 100,         // Results to return
    nprobe: 50,         // Clusters to search
}

// Trade-off:
// - Higher nprobe → better recall, slower query
// - Lower nprobe → faster query, lower recall
```

## Serialization Format

### Index File Structure

```
┌─────────────────────────────────────┐
│ Magic: "MSTG" (4 bytes)             │
│ Version: 1 (4 bytes)                │
├─────────────────────────────────────┤
│ Metadata:                           │
│  - dim, padded_dim                  │
│  - metric, rotator_type             │
│  - ex_bits                          │
├─────────────────────────────────────┤
│ Rotator data (serialized)           │
├─────────────────────────────────────┤
│ Clusters (num_clusters):            │
│  For each cluster:                  │
│   - Medoid vector                   │
│   - Vector IDs                      │
│   - Quantized vectors               │
├─────────────────────────────────────┤
│ CRC32 Checksum                      │
└─────────────────────────────────────┘
```

**Note**: HNSW graph is currently rebuilt on load (not serialized)

## Comparison: IVF vs MSTG

| Aspect | IVF | MSTG |
|--------|-----|------|
| **Clustering** | Flat k-means | Hierarchical tree |
| **Cluster Selection** | Linear scan O(k) | HNSW O(log k) |
| **Cluster Count** | ~sqrt(N)×2 | ~N/100 |
| **Memory Model** | All in-memory | Hybrid (mem + disk) |
| **Scalability** | <100M vectors | >100M vectors |
| **Training Time** | Fast (single k-means) | Medium (multi-level) |
| **Query Latency** | ~5-15ms (in-mem) | ~3-10ms (cached) |
| **Index Size** | Smaller | Larger (HNSW) |
| **Production Ready** | ✅ Yes | ⚠️ Experimental |

## References

- RaBitQ Paper: Residual-Aware Bit Quantization
- HNSW Paper: Efficient and robust approximate nearest neighbor search
- Multi-level k-means: Hierarchical clustering for large datasets

## Future Work

1. **Disk-based implementation**: Complete the on-disk cluster storage
2. **HNSW persistence**: Serialize graph to avoid rebuild
3. **Advanced caching**: Implement LRU with prefetching
4. **Compression**: Add cluster compression support
5. **Distributed version**: Shard clusters across machines
6. **Benchmarking**: Compare with production systems at 100M+ scale

---

**Status**: In-memory prototype complete, ready for disk-based development
**Version**: 0.4.0
**Last Updated**: 2025-10-18
