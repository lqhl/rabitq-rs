# Filtered Vector Search

This document describes the filtered vector search feature added to rabitq-rs.

## Overview

Filtered vector search allows you to restrict the search space to a specific subset of vectors by providing a RoaringBitmap containing valid vector IDs. This is useful for:

- **Metadata filtering**: Search only among vectors matching certain criteria
- **Access control**: Restrict search results based on permissions
- **Multi-tenancy**: Isolate searches within tenant-specific data
- **Time-based filtering**: Search only recent or historical data

## API

### Core Method

```rust
pub fn search_filtered(
    &self,
    query: &[f32],
    params: SearchParams,
    filter: &RoaringBitmap,
) -> Result<Vec<SearchResult>, RabitqError>
```

**Parameters:**
- `query`: The query vector
- `params`: Search parameters (top_k, nprobe)
- `filter`: A RoaringBitmap containing valid candidate vector IDs

**Returns:**
- A vector of search results, sorted by score, containing only IDs present in the filter

### Usage Example

```rust
use rabitq_rs::{IvfRabitqIndex, Metric, RoaringBitmap, RotatorType, SearchParams};

// Train index
let index = IvfRabitqIndex::train(
    &data,
    16,                        // nlist
    7,                         // bits
    Metric::L2,
    RotatorType::FhtKacRotator,
    42,                        // seed
    false,                     // faster_config
)?;

// Create filter (only search among IDs 10-99)
let mut filter = RoaringBitmap::new();
for id in 10..100 {
    filter.insert(id);
}

// Search with filter
let params = SearchParams::new(10, 32); // top_k=10, nprobe=32
let results = index.search_filtered(&query, params, &filter)?;

// All returned IDs will be in range [10, 100)
for result in results {
    assert!(filter.contains(result.id as u32));
}
```

## CLI Support

The `ivf_rabitq` binary supports filtered search via the `--filter` option:

```bash
# Build index
cargo run --release --bin ivf_rabitq -- \
    --base data/gist/gist_base.fvecs \
    --nlist 4096 \
    --bits 3 \
    --save index.bin

# Search with filter (filter.ivecs contains valid IDs)
cargo run --release --bin ivf_rabitq -- \
    --load index.bin \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --filter filter.ivecs \
    --nprobe 1024
```

**Note:** The filter file should be in `.ivecs` format containing the list of valid vector IDs.

**Limitation:** Filter is not currently supported in benchmark mode (`--benchmark` flag).

## Implementation Details

### Performance Characteristics

- **Early filtering**: Candidates are filtered immediately after cluster traversal, before distance computation
- **Zero-copy check**: Uses RoaringBitmap's efficient `contains()` method (O(1) average)
- **Minimal overhead**: For sparse filters (few IDs), filtering adds negligible overhead
- **Works with all metrics**: Compatible with both L2 and InnerProduct metrics

### Memory Usage

RoaringBitmap is highly efficient:
- Typical compression: 10-1000x better than naive bit arrays
- Small bitmaps (< 4096 IDs): ~16-32 bytes
- Large bitmaps (millions of IDs): typically 1-5 bytes per ID

### Filter Behavior

- **Empty filter**: Returns empty results
- **Full filter** (all IDs): Equivalent to regular search
- **Sparse filter**: Efficiently skips non-matching candidates
- **Dense filter**: Minimal overhead compared to unfiltered search

## Testing

The implementation includes comprehensive tests:

```bash
# Run filtered search tests
cargo test filtered_search

# Run all tests
cargo test

# Run example
cargo run --release --example filtered_search
```

## API Exports

The following types are re-exported for convenience:

```rust
pub use rabitq_rs::{
    IvfRabitqIndex,     // Main index type
    SearchParams,        // Search configuration
    SearchResult,        // Search result entry
    RoaringBitmap,       // Filter bitmap (re-exported from roaring crate)
    Metric,              // Distance metric (L2, InnerProduct)
    RotatorType,         // Rotation method
};
```

## Dependencies

Filtered search adds the `roaring` crate (v0.10) as a dependency:

```toml
[dependencies]
roaring = "0.10"
```

## Future Enhancements

Potential improvements:

1. **Cluster-level filtering**: Pre-filter clusters based on ID ranges
2. **Benchmark mode support**: Integrate filtering into nprobe sweep
3. **Multiple filters**: Support AND/OR combinations of filters
4. **Filter statistics**: Report filter hit rates and pruning effectiveness
5. **Async filtering**: Support dynamic filter updates during search
