# MSTG Implementation TODOs

## Phase 1: Code Restructuring and Organization
- [x] 1.1 Move index building logic out of `index.rs` into `builder.rs`. Create `MstgBuilder`.
- [x] 1.2 Move search logic out of `index.rs` into `search.rs`.
- [x] 1.3 Clean up `index.rs`.

## Phase 2: True Scalar Quantization for HNSW
- [x] 2.1 Implement `Distance` traits in `hnsw_rs` for quantized types (`BF16Vector`, `FP16Vector`, `INT8Vector`).
- [x] 2.2 Modify `CentroidIndex` to drop the redundant `Box<[Vec<f32>]>` array and use quantized vectors.
- [x] 2.3 Add parsing/training logic for `FP16` and `INT8` quantization to avoid panics.

## Phase 3: Metadata Directory and Disk-Tier Serialization
- [x] 3.1 Rewrite `PostingListDirectory` in `metadata.rs` with `PostingListEntry`.
- [x] 3.2 Refactor `io.rs` to write posting lists linearly to disk and update offsets.
- [x] 3.3 Update `MstgIndex` to use an enum `PostingDataSource` instead of `Vec<PostingList>`.

## Phase 4: Dynamic Query and Parallel I/O Loading
- [x] 4.1 Update `search.rs` to intersect centroid candidates with `PostingListDirectory`.
- [x] 4.2 Retrieve byte ranges from the memory-mapped index or disk.
- [x] 4.3 Dynamically convert deserialized slices to FastScan SIMD instructions during search.

## Phase 5: Testing, Validation, and Benchmarking
- [x] 5.1 Add integration tests measuring memory differences after HNSW scalar quantization fix.
- [x] 5.2 Add I/O disk-tier tests.
- [x] 5.3 Run benchmarking comparisons.
