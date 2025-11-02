# Next Session: Rewrite Rust Packing System to Match C++

## Objective

Completely rewrite the Rust ex-code packing system to match C++'s SIMD-optimized format, eliminating the unpacking overhead and achieving C++-level performance.

## Background Context

**Current Status**:
- Rust achieves ~85% of C++ performance
- Main bottleneck: ~15% overhead from unpacking ex-codes before dot product
- Root cause: Incompatible packing formats prevent direct SIMD operations on packed data

**C++ Advantage**:
- Never unpacks ex-codes
- Uses specialized SIMD functions (`ip_func_`) operating directly on packed data
- Packing format optimized for AVX-512 SIMD extraction

**Critical Files to Read**:
1. `/data00/home/liuqin.v/workspace/rabitq-rs/docs/PACKING_FORMAT_ANALYSIS.md` - Complete technical analysis
2. `/data00/home/liuqin.v/workspace/RaBitQ-Library/include/rabitqlib/quantization/pack_excode.hpp` - C++ packing reference
3. `/data00/home/liuqin.v/workspace/RaBitQ-Library/include/rabitqlib/utils/space.hpp` - C++ SIMD functions reference

## Task Breakdown

### Phase 1: Implement C++-Compatible Packing (Priority: 1/3/7-bit)

**Files to Modify**:
- `src/simd.rs` - Rewrite `pack_ex_code` functions

**Tasks**:
1. **1-bit packing** (ex_bits=0, binary-only RaBitQ):
   - Reference: `pack_excode.hpp:13-30` (`packing_1bit_excode`)
   - Pack 16 codes into uint16 (instead of 2 bytes)
   - Test: Verify bit-exact match with C++ output

2. **2-bit packing** (3-bit total RaBitQ):
   - Reference: `pack_excode.hpp:32-54` (`packing_2bit_excode`)
   - Pack 16 codes into int32 with specific bit arrangement
   - Formula: `compact = (code3 << 6) | (code2 << 4) | (code1 << 2) | code0`
   - Test: Verify against C++ reference implementation

3. **6-bit packing** (7-bit total RaBitQ):
   - Reference: `pack_excode.hpp:175-210` (`packing_6bit_excode`)
   - Split into 4-bit + 2-bit parts
   - Pack 16 codes into 12 bytes (8 + 4)
   - Test: Verify bit-exact compatibility

**Implementation Template**:
```rust
// src/simd.rs

/// Pack 2-bit ex-codes (C++-compatible format)
/// Reference: C++ pack_excode.hpp:32-54
pub fn pack_ex_code_2bit_cpp_compat(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    debug_assert_eq!(dim % 16, 0, "dim must be multiple of 16 for 2-bit");
    debug_assert_eq!(packed.len(), dim / 16 * 4);

    let mut ex_idx = 0;
    let mut packed_idx = 0;

    while ex_idx < dim {
        // Load 16 u16 codes
        let codes: [u16; 16] = // ...

        // Group into 4 int32s (as in C++)
        let code0: i32 = // codes[0..4] lower 2 bits
        let code1: i32 = // codes[4..8] lower 2 bits
        let code2: i32 = // codes[8..12] lower 2 bits
        let code3: i32 = // codes[12..16] lower 2 bits

        // Combine with C++ bit arrangement
        let compact: i32 = (code3 << 6) | (code2 << 4) | (code1 << 2) | code0;

        // Write as 4 bytes
        packed[packed_idx..packed_idx + 4].copy_from_slice(&compact.to_le_bytes());

        ex_idx += 16;
        packed_idx += 4;
    }
}
```

### Phase 2: Implement Specialized SIMD Dot Product Functions

**Files to Modify**:
- `src/simd.rs` - Add new functions

**Tasks**:
1. **2-bit SIMD dot product**:
   - Reference: `space.hpp:293-316` (`ip16_fxu2_avx512`)
   - Port AVX-512 → AVX2 (current Rust target)
   - Directly operate on C++-packed data
   - No unpacking step!

2. **6-bit SIMD dot product**:
   - Reference: `space.hpp:456-486` (`ip16_fxu6_avx512`)
   - Port to AVX2
   - Handle 4-bit + 2-bit split format

**Implementation Template**:
```rust
// src/simd.rs

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn ip_packed_ex2_cpp_compat_avx2(
    query: &[f32],
    packed_ex_code: &[u8],
    padded_dim: usize
) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mask = _mm_set1_epi8(0b00000011);

    let mut query_ptr = query.as_ptr();
    let mut code_ptr = packed_ex_code.as_ptr();

    for _ in 0..(padded_dim / 16) {
        // Load 4 bytes (int32) containing 16 2-bit codes
        let compact = *(code_ptr as *const i32);

        // Unpack using C++ method: shifts + mask
        let code_i32 = _mm_set_epi32(
            compact >> 6,
            compact >> 4,
            compact >> 2,
            compact
        );
        let code_masked = _mm_and_si128(code_i32, mask);

        // Convert to float and FMA
        // (AVX2 requires more steps than AVX-512)
        let code_f32 = // ... convert i8 -> i32 -> f32
        let query_vec = _mm256_loadu_ps(query_ptr);
        sum = _mm256_fmadd_ps(code_f32, query_vec, sum);

        query_ptr = query_ptr.add(16);
        code_ptr = code_ptr.add(4);
    }

    // Horizontal sum
    // ...
}
```

### Phase 3: Function Dispatch System

**Files to Modify**:
- `src/simd.rs` - Add dispatch function
- `src/ivf.rs` - Use function pointers

**Tasks**:
1. Create function pointer type:
```rust
pub type ExIpFunc = fn(&[f32], &[u8], usize) -> f32;
```

2. Implement selector:
```rust
pub fn select_excode_ipfunc(ex_bits: usize) -> ExIpFunc {
    match ex_bits {
        0 | 1 => ip_packed_ex1_cpp_compat,
        2 => ip_packed_ex2_cpp_compat,  // 3-bit total
        6 => ip_packed_ex6_cpp_compat,  // 7-bit total
        _ => panic!("Unsupported ex_bits: {}", ex_bits),
    }
}
```

3. Store function pointer in `IvfRabitqIndex`:
```rust
pub struct IvfRabitqIndex {
    // ... existing fields
    ip_func: ExIpFunc,  // NEW: Store at index build time
}
```

4. Update search to use function pointer (no unpacking!):
```rust
// src/ivf.rs:~1956
let ex_code_packed = &cluster.ex_codes_packed[global_idx];

// BEFORE (current): Unpack then dot product
// let ex_code_unpacked = unpack_ex_code(...);
// let ex_dot = dot_u16_f32(&ex_code_unpacked, &query);

// AFTER (C++-style): Direct dot product on packed data
let ex_dot = (self.ip_func)(
    &query_precomp.rotated_query,
    ex_code_packed,
    self.padded_dim
);
```

### Phase 4: Update Quantization and Index Building

**Files to Modify**:
- `src/quantizer.rs` - Use new packing functions
- `src/ivf.rs` - Update ClusterData construction

**Tasks**:
1. Replace `pack_ex_code` calls with C++-compatible versions
2. Ensure dimension padding (dim % 16 == 0 or dim % 64 == 0)
3. Update `ClusterData` to use new format
4. Initialize `ip_func` during index construction

### Phase 5: Testing and Validation

**Critical Tests**:
1. **Packing format verification**:
```rust
#[test]
fn test_pack_ex2_cpp_compat() {
    let ex_code = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let mut packed_rust = vec![0u8; 4];
    let mut packed_cpp = vec![0u8; 4];  // From C++ reference

    pack_ex_code_2bit_cpp_compat(&ex_code, &mut packed_rust, 16);

    // Load C++ output (pre-computed or via FFI)
    let cpp_expected = [0x1B, 0x1B, 0x1B, 0x1B];  // Example

    assert_eq!(packed_rust, cpp_expected, "Packing format mismatch!");
}
```

2. **SIMD dot product verification**:
```rust
#[test]
fn test_ip_packed_ex2_vs_cpp() {
    let query = vec![1.0f32; 960];
    let ex_code = (0..960).map(|i| (i % 4) as u16).collect::<Vec<_>>();

    // Pack using C++-compatible format
    let mut packed = vec![0u8; 960 / 16 * 4];
    pack_ex_code_2bit_cpp_compat(&ex_code, &mut packed, 960);

    // Method 1: Packed SIMD (new)
    let result_packed = ip_packed_ex2_cpp_compat(&query, &packed, 960);

    // Method 2: Reference (unpack + dot)
    let result_ref = dot_u16_f32(&ex_code, &query);

    assert!((result_packed - result_ref).abs() < 1e-3);
}
```

3. **End-to-end search correctness**:
```bash
cargo test fastscan_matches_naive_bits3 --release
cargo test fastscan_matches_naive_bits7 --release
```

4. **Performance benchmark**:
```rust
// Compare search time before/after
// Target: Match C++ performance (remove ~15% overhead)
```

## Expected Outcomes

### Performance Gains
- **Before**: ~85% of C++ speed (unpacking overhead)
- **After**: ~95-100% of C++ speed (zero unpacking)
- **Speedup**: ~12-18% improvement in search time

### Memory Impact
- Current: 4.4× memory savings via on-demand unpacking
- After: May need to pre-unpack or keep packed format
- Trade-off: Speed vs memory (configurable?)

## Potential Challenges

### Challenge 1: AVX2 vs AVX-512

**Problem**: C++ uses AVX-512, Rust currently targets AVX2

**Solution**:
- Port AVX-512 operations to AVX2 (requires more instructions)
- OR: Add AVX-512 feature flag (nightly Rust)
- OR: Provide both implementations with runtime selection

### Challenge 2: Bit Manipulation Complexity

**Problem**: C++ uses complex bit shifts and masks

**Solution**:
- Carefully study C++ implementation line-by-line
- Add extensive comments explaining bit layout
- Write unit tests for each packing configuration

### Challenge 3: Dimension Padding Requirements

**Problem**: Different bit configs require different padding (16 vs 64)

**Solution**:
- Enforce padding during index construction
- Update `DynamicRotator` to respect requirements
- Add validation in packing functions

## Development Workflow

### Recommended Order

1. **Day 1-2**: Implement 2-bit packing (C++-compatible)
   - Start with simplest non-trivial case
   - Write packing function
   - Write unpacking function
   - Verify bit-exact match with C++ reference

2. **Day 3-4**: Implement 2-bit SIMD dot product
   - Port C++ AVX-512 to Rust AVX2
   - Test against reference implementation
   - Benchmark performance

3. **Day 5-6**: Implement 6-bit packing + SIMD
   - More complex bit manipulation
   - Split 4-bit + 2-bit format
   - Extensive testing

4. **Day 7**: Function dispatch system
   - Add function pointers
   - Update search logic
   - Integration testing

5. **Day 8-9**: Testing and optimization
   - Run full test suite
   - Performance benchmarking
   - Bug fixes

6. **Day 10**: Documentation and cleanup
   - Update CLAUDE.md
   - Add code comments
   - Write performance report

### Code Review Checkpoints

- [ ] Packing format matches C++ bit-exactly
- [ ] SIMD dot product produces correct results
- [ ] No performance regression
- [ ] All existing tests pass
- [ ] Memory usage acceptable
- [ ] Code well-documented

## Success Criteria

### Must Have
1. ✅ Bit-exact packing format match with C++
2. ✅ Correct search results (all tests pass)
3. ✅ No unpacking overhead (direct SIMD on packed data)
4. ✅ Support 1/3/7-bit configurations

### Should Have
1. ✅ Performance within 5% of C++
2. ✅ Memory usage not worse than 2× current
3. ✅ Clean, maintainable code

### Nice to Have
1. ✅ AVX-512 support (feature flag)
2. ✅ Runtime CPU feature detection
3. ✅ Adaptive packing format selection

## Reference Commands

```bash
# Build with AVX2 (current)
cargo build --release

# Build with AVX-512 (future)
cargo +nightly build --release --features avx512

# Run tests
cargo test --release

# Benchmark
cargo run --release --bin ivf_rabitq -- \
    --load data/gist/index.bin \
    --queries data/gist/gist_query.fvecs \
    --gt data/gist/gist_groundtruth.ivecs \
    --benchmark

# Compare with C++
cd /data00/home/liuqin.v/workspace/RaBitQ-Library
# (build and run C++ benchmark)
```

## Questions to Resolve

1. Should we keep both packing formats (old + new) with a feature flag?
2. How to handle dimension padding requirements in public API?
3. Should we pre-compute and cache unpacked codes as fallback?
4. AVX-512 support: nightly-only or dual implementation?

## Final Note

This is a **significant refactoring** that touches core performance-critical code. Proceed incrementally with extensive testing at each step. The goal is to eliminate the 15% unpacking overhead while maintaining correctness and code quality.

**Estimated Total Time**: 10-15 days of focused development

Good luck! 🚀
