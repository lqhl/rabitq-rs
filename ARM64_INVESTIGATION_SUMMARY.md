# ARM64 Investigation Summary

**Date:** 2024-01-XX  
**Goal:** Enable full test suite on ARM64/AArch64 platforms (Apple Silicon Macs)  
**Status:** ❌ Uncovered critical bugs preventing ARM64 support  

## What We Attempted

The original goal was to fix tests that were disabled on non-x86_64 architectures with the `#[cfg_attr(not(target_arch = "x86_64"), ignore = "...")]` attribute. The plan was to:

1. **Relax test tolerances** to accommodate minor floating-point differences between x86_64 AVX2 and ARM64 NEON SIMD implementations
2. **Use platform-specific expected values** where necessary
3. **Enable all tests on ARM64** to ensure full functionality

## What We Discovered

During investigation, we uncovered **critical bugs** in the ARM64 implementation that go far beyond simple floating-point precision differences:

### 🔴 Critical Bug #1: Negative L2 Distances

L2 (Euclidean) distance calculations produce **mathematically impossible negative values** on ARM64:

```rust
// Example output on ARM64:
Query 0: [1.0, 0.0, 0.0, 0.0]
Results:
  ID=192, score=-2573.178955  ❌ Negative!
  ID=0,   score=-2285.571045  ❌ Negative!
  ID=5,   score=-1921.903931  ❌ Negative!
```

**Expected:** L2 squared distances should always be ≥ 0  
**Actual on ARM64:** Almost all distances are negative  
**Impact:** L2 metric completely unusable on ARM64

### 🟡 Critical Bug #2: Inner Product Ranking Issues

Inner Product metric doesn't produce negative values, but **ranking is incorrect**:

```rust
// Query vector should rank itself first (highest inner product)
Query 0: [1.0, 0.0, 0.0, 0.0]
Results:
  [0] ID=1, score=14.710691   ❌ Wrong vector ranked first!
  [1] ID=0, score=11.342443   ← This is the query vector!
```

**Expected:** Query vector finds itself with highest score  
**Actual on ARM64:** Query vector often not ranked first  
**Impact:** Search results unreliable on ARM64

## Root Cause Analysis

### ✅ What's NOT Broken

- **Basic SIMD operations:** ARM NEON `dot()` and `l2_distance_sqr()` produce correct results for unquantized vectors
- **Heap sorting logic:** BinaryHeap comparison and result sorting are correct
- **Compiler optimizations:** Bug occurs in both debug and release builds

### 🔍 Suspected Root Causes

The bugs appear to stem from **quantized distance estimation** in the RaBitQ algorithm:

1. **Quantization factors calculation** (`src/quantizer.rs`):
   - `f_add`, `f_rescale`, `f_add_ex`, `f_rescale_ex` may be calculated incorrectly on ARM64
   - Platform-dependent numerical instability in intermediate computations
   - Possible issues with random rotation (FHT-Kac) producing subtly different results

2. **Distance reconstruction** (`src/ivf.rs:2120-2130`):
   - Formula: `distance = f_add + g_add + f_rescale * total_term`
   - May accumulate errors differently on ARM64 vs x86_64
   - LUT (Lookup Table) quantization may behave differently

3. **SIMD operations on packed codes** (`src/simd.rs`):
   - Different implementations for AVX2/AVX-512 (x86) vs NEON (ARM)
   - Bit packing/unpacking routines may have subtle differences
   - Extended code inner product computation

## What We Accomplished

Despite not achieving the original goal, we made valuable improvements:

### ✅ Improvements Made

1. **Cross-Platform Float Comparison Utilities** (`src/tests.rs`):
   ```rust
   mod float_comparison {
       pub fn score_acceptable_1bit(fs: f32, nv: f32) -> bool;
       pub fn score_acceptable_3bit(fs: f32, nv: f32) -> bool;
       pub fn score_acceptable_7bit(fs: f32, nv: f32) -> bool;
   }
   ```
   These functions provide architecture-aware tolerance checking that will be useful once the underlying bugs are fixed.

2. **Comprehensive Documentation**:
   - **ARM64_KNOWN_ISSUES.md**: Detailed analysis of bugs, reproduction steps, debug guidance
   - **README.md**: Added platform support notice warning users
   - **Test comments**: Updated ignore messages to reference documentation

3. **Diagnostic Examples**:
   - `examples/test_l2_arm.rs`: Demonstrates L2 distance bug
   - `examples/test_ip_arm.rs`: Demonstrates IP ranking issues

4. **Clearer Test Failures**:
   - Tests now skip on ARM64 with informative messages
   - Running tests shows: `93 passed; 0 failed; 14 ignored`
   - Users know exactly why tests are skipped

### 📋 Tests Currently Disabled on ARM64

```
tests::ivf_search_recovers_identical_vectors
tests::fastscan_matches_naive_l2
tests::fastscan_matches_naive_ip
tests::fastscan_matches_naive_bits3_l2
tests::fastscan_matches_naive_bits3_ip
tests::fastscan_matches_naive_bits7_ip
tests::one_bit_search_has_no_extended_pruning
tests::test_batch_search_matches_per_vector_l2
```

## Next Steps to Fix ARM64 Support

### 🎯 Immediate Actions

1. **Add instrumentation** to quantization and distance calculation:
   ```rust
   #[cfg(target_arch = "aarch64")]
   eprintln!("DEBUG: f_add={}, f_rescale={}, distance={}", ...);
   ```

2. **Compare quantization outputs** on same data:
   - Run identical test on x86_64 and ARM64
   - Save quantized vectors, factors, and distances
   - Identify where values diverge

3. **Test simplified configurations**:
   - Use identity rotator (no rotation) to rule out FHT issues
   - Test bits=1 only (no extended codes) to isolate binary quantization
   - Test with a single cluster (no IVF) to isolate quantization from clustering

### 🔬 Deep Investigation

4. **Profile numerical stability**:
   - Check for potential overflow/underflow in quantization formulas
   - Verify that NEON floating-point behavior matches AVX2
   - Test with different rounding modes

5. **Compare with C++ reference**:
   - Run C++ RaBitQ on ARM64 with same data
   - Check if C++ has the same issues (likely not, since it's tested)
   - Identify algorithmic differences

6. **Audit SIMD implementations**:
   - Line-by-line comparison of AVX2 vs NEON code paths
   - Verify bit packing/unpacking produces identical results
   - Check LUT building logic on ARM64

### 📝 Testing Strategy

7. **Create minimal reproduction**:
   - Simplest possible test case (4 vectors, 2D)
   - Print all intermediate values
   - Manually verify formula step-by-step

8. **Add comprehensive unit tests**:
   - Test quantization factors calculation in isolation
   - Test distance reconstruction with known inputs
   - Test SIMD operations against scalar reference

## Recommendations

### For Users

- **Do NOT use this library on ARM64 platforms** for production workloads
- **x86_64 platforms work correctly** - fully tested and recommended
- Monitor ARM64_KNOWN_ISSUES.md for updates on fix progress

### For Contributors

- **ARM64 investigation is high priority** - blocks Apple Silicon adoption
- **Start with test_l2_arm.rs example** - simplest reproduction
- **Focus on quantizer.rs first** - most likely source of bug
- **Compare numerical outputs** - instrument code heavily for debugging

## Conclusion

While we did not achieve the original goal of enabling all tests on ARM64, we:

1. ✅ Discovered critical bugs that would have caused silent failures in production
2. ✅ Documented the issues comprehensively for future debugging
3. ✅ Improved test infrastructure with better float comparison utilities
4. ✅ Created diagnostic tools to aid investigation
5. ✅ Prevented users from unknowingly using broken code on ARM64

**The bugs are real, reproducible, and need deep algorithmic investigation.** Simply relaxing test tolerances would have hidden severe correctness issues. The right next step is fixing the underlying quantization and distance estimation bugs on ARM64.

---

**Status Tracking:**
- Issue: [Link to GitHub issue once created]
- Contact: [Maintainer info]
- Last Updated: 2024-01-XX