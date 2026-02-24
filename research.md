# MSTG (Multi-Scale Tree Graph) Research Report: Implementation vs. Specification

## Overview

This report provides a deep dive into the `rabitq-rs` codebase, comparing the theoretical design outlined in `docs/MSTG_SPEC.md` against the actual Rust implementation living within `src/mstg/`. 

While many core algorithmic components—such as hierarchical clustering and closure assignments—have been beautifully and fully realized, the project is still a work-in-progress. Crucially, the foundational hybrid memory-disk structure and several planned memory optimizations are either missing, incomplete, or currently implemented in a way that contradicts the specification.

---

## 1. The Three-Tier Architecture: Disk I/O vs. In-Memory
**Specification**: 
MSTG is designed to be a hybrid memory-disk index, segmented into three functional tiers:
1. **Memory Tier**: HNSW graph for fast centroid cluster navigation.
2. **Metadata Tier**: A directory mapping posting list metadata to exact disk offsets.
3. **Disk Tier**: RaBitQ-compressed posting lists stored sequentially on disk to minimize RAM usage. They are hypothesized to be dynamically loaded in parallel using `memmap2` when matching query candidates.

**Implementation Reality**:
The entire architecture operates **strictly in-memory**. 
* The `MstgIndex` struct (in `index.rs`) retains all posting lists uniformly via `pub posting_lists: Vec<PostingList>`.
* `PostingListDirectory` (in `metadata.rs`) is effectively an empty structural placeholder. It handles no explicit disk-offset logic.
* `io.rs` implements standard bulk serialization using `bincode`. When saving, it blindly writes the entire index structure into a singular `.mstg` file. When loaded, everything is placed directly into RAM.
* There is no parallel dynamic loading of posting lists during query searches.
* Files explicitly earmarked for search and build operations in the specification (`builder.rs` and `search.rs`) exist as 6-line stubs containing the comment `// Placeholder for future implementation (Milestone 6/7)`. Instead, search routines are baked directly into `index.rs`.

---

## 2. HNSW Centroid Navigation and Scalar Quantization
**Specification**:
The system states that HNSW navigates scalar-quantized representations of centroids (supporting `fp32`, `bf16`, `fp16`, or `int8`). For example, defaulting to `bf16` trims the memory payload of billions-scale dataset centroids by 50% with near-invisible accuracy losses.

**Implementation Reality**:
This optimization is currently superficial and introduces **memory bloat** rather than reducing it.
* The `fp16` and `int8` precision options defined in the config cause `panic!` loops within `CentroidIndex::build` (`_ => panic!("Unsupported precision: {:?}", precision)`).
* While `FP32` and `BF16` conversions do execute to populate the `CentroidData` enum, **neither is used for the actual HNSW search**.
* Because `hnsw_rs::prelude::Hnsw<f32, DistL2>` is strictly typed for `f32`, the graph receives raw pointers to a redundant `Box<[Vec<f32>]>` containing full-precision centroids.
* Consequently, memory estimations scale up to compute both the `f32` list **and** the arbitrarily converted `BF16` list, completely defeating the purpose of the optimization.

---

## 3. Index Building: Clustering and Closure Assignment
**Specification**: 
Building indices includes Phase 1: Hierarchical Balanced Clustering (recursively splitting large groups using balanced k-means) and Phase 2: Closure Assignment (duplicating vectors along cluster boundaries, relying on an RNG (Relative Neighborhood Graph) rule to cull geometric redundancy).

**Implementation Reality**:
This logic **perfectly mirrors the specification**.
* `clustering.rs` elegantly establishes K-means bifurcation routines (`HierarchicalClustering`), forcing large parent nodes to fork vectors into constrained subgroups while shifting borderline data elements dynamically to stabilize variances.
* `closure.rs` fulfills Phase 2 perfectly by checking all centroid offsets relative to an `epsilon` proximity band. Furthermore, `ClosureAssigner::apply_rng_rule` explicitly evaluates triangles of distances bridging queries, candidates, and previously selected targets precisely as described to limit replication.

---

## 4. RaBitQ Distance Computations and Queries
**Specification**:
The spec details processing residuals relative to a cluster's centroid via RaBitQ multi-scale 7-bit quantization and utilizing Dynamic Pruning (query aware adaptive search parameters) during graph traversal.

**Implementation Reality**:
This is **functionally complete**.
* In `posting_list.rs`, the lists rigorously generate relative offsets and instantiate optimized `RabitqConfig` contexts exclusively assigned to that localized data blob.
* `index.rs` demonstrates operational `dynamic_prune`, taking raw centroid proximity vectors and limiting the candidates returned by an algorithmic percentage (`params.pruning_epsilon`).
* Distance execution properly pivots over SIMD-intrinsic batch calculations (`FastScan` arrays), ensuring that even within an all-in-memory constraint, localized searches execute rapidly in parallel across standard CPU boundaries.
