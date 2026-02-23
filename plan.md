# MSTG Implementation Plan

This plan outlines the steps required to align the current MSTG (Multi-Scale Tree Graph) Rust implementation with the design specification detailed in `docs/MSTG_SPEC.md`.

## Phase 1: Code Restructuring and Organization

The purpose of this phase is to reorganize the code based on the intended modular structure, mitigating the bloat currently present in `index.rs`.

- **Step 1.1:** Move index building logic out of `index.rs` into `builder.rs`. Create a clear `MstgBuilder` struct to orchestrate hierarchical clustering, closure assignment, and index serialization.
- **Step 1.2:** Move search logic (FastScan batch processing, query-aware pruning) out of `index.rs` into `search.rs`. This will isolate query execution.
- **Step 1.3:** Clean up `index.rs` so it primarily acts as the core data structure definition and entry point, referencing the builder and search components.

## Phase 2: True Scalar Quantization for HNSW

The current HNSW implementation parses quantized structures but retains full precision `f32` vectors for the `hnsw_rs` index map, resulting in double memory usage.

- **Step 2.1:** Implement custom `Distance` traits required by the `hnsw_rs` crate for each quantized type (`BF16Vector`, `FP16Vector`, `INT8Vector`). These distance functions should calculate approximated L2 distances directly in the quantized space, accumulating directly over `f32` variables.
- **Step 2.2:** Modify the `CentroidIndex` struct to drop the redundant `Box<[Vec<f32>]>` array. The HNSW graph must utilize the quantized vector array directly.
- **Step 2.3:** Add parsing/training logic to support the unimplemented `FP16` and `INT8` quantization streams without panicking during `CentroidIndex::build`.

## Phase 3: Metadata Directory and Disk-Tier Serialization

The fundamental purpose of MSTG is its hybrid memory-disk nature, which is currently bypassed in favor of a 100% in-memory implementation.

- **Step 3.1:** Rewrite `PostingListDirectory` in `metadata.rs`. Implement tracking records (`PostingListEntry`) containing the `cluster_id`, `disk_offset`, `size_bytes`, `num_vectors`, and `avg_vector_norm`.
- **Step 3.2:** Refactor `io.rs` to write posting lists linearly to disk, updating the byte offsets directly into the `PostingListDirectory` during serialization.
- **Step 3.3:** Decouple `MstgIndex` from holding an active `Vec<PostingList>`. Swap this for a direct `mmap` reference (using the `memmap2` crate) or an active File Handle over the `.mstg` payload.

## Phase 4: Dynamic Query and Parallel I/O Loading

Once the posting lists reside entirely on disk, searches need to dynamically select and load posting list sectors on-demand.

- **Step 4.1:** Update `search.rs` to intersect the centroid candidates resolved from HNSW with the `PostingListDirectory`.
- **Step 4.2:** Retrieve requested byte ranges from the memory-mapped index utilizing the resolved disk offsets. Parallelize this loading routine using RaBitQ bindings in Rayon.
- **Step 4.3:** Ensure deserialized slices dynamically convert to FastScan SIMD instructions without bottlenecking the search process, and dispose of the parsed data to prevent memory climbing.

## Phase 5: Testing, Validation, and Benchmarking

- **Step 5.1:** Author dedicated integration tests measuring memory differences after the HNSW scalar quantization fix.
- **Step 5.2:** Ensure I/O disk-tier tests measure expected behavior: initializing `MstgIndex` runs immediately without holding large `Vec` queues.
- **Step 5.3:** Run benchmarking comparisons against the existing `IVF-RaBitQ` implementations to ensure the hybrid memory-disk tradeoff behaves correctly against speed and recall benchmarks.
