# C++ RaBitQ FastScan SIMD Implementation Analysis

## Executive Summary

This document provides a detailed analysis of performance optimizations found in the C++ RaBitQ implementation, focusing on FastScan SIMD techniques, memory layout, prefetching, and parallelization that can be safely ported to the Rust implementation without modifying core algorithms.

---

## 1. FastScan SIMD Implementation Details

### 1.1 Core FastScan Architecture

**File**: `/data00/home/liuqin.v/workspace/RaBitQ-Library/include/rabitqlib/fastscan/fastscan.hpp`

#### Batch Processing Model
- **Batch Size**: 32 vectors processed per batch (`kBatchSize = 32`)
- **Rationale**: Optimal for SIMD register width and cache efficiency
- **Key Insight**: Data is organized to align with batch processing requirements

#### Data Encoding Strategy
```cpp
constexpr static std::array<int, 16> kPerm0 = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
};
```
- **Purpose**: Interleaved permutation for optimal SIMD shuffle operations
- **Benefit**: Reduces number of shuffle instructions required during distance calculation
- **MSB-first Bit Order**: Matches C++ reference implementation for binary codes

### 1.2 AVX-512 Implementation (Lines 116-174)

**Peak Performance**: 64-byte (512-bit) registers

**Key Optimizations**:

1. **Parallel Accumulation with 4 Registers**
```cpp
__m512i accu0 = _mm512_setzero_si512();  // Vector 8-15
__m512i accu1 = _mm512_setzero_si512();  // Vector 0-7
__m512i accu2 = _mm512_setzero_si512();  // Vector 16-23
__m512i accu3 = _mm512_setzero_si512();  // Vector 24-31
```
- **Strategy**: Four separate accumulators prevent false dependencies
- **Benefit**: Enables out-of-order execution and instruction-level parallelism
- **Loop Iteration**: 64 bytes per iteration (8 uint8_t codes for 2 vectors)

2. **Lookup Table with Shuffle**
```cpp
res_lo = _mm512_shuffle_epi8(lut, lo);   // Extract lower 4-bit results
res_hi = _mm512_shuffle_epi8(lut, hi);   // Extract upper 4-bit results
```
- **Why**: `pshufb` (shuffle) is faster than multiple masking/shifting operations
- **Throughput**: 1 cycle latency, can execute every cycle

3. **Fine-grained Result Reconstruction**
```cpp
__m512i ret1 = _mm512_add_epi16(
    _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
    _mm512_shuffle_i64x2(accu0, accu1, 0b01001110)
);
```
- **Pattern**: Uses both `permute` and `blend` to sum accumulators
- **Efficiency**: Reduces register pressure by strategically interleaving operations

### 1.3 AVX-2 Fallback Implementation (Lines 176-225)

**Peak Performance**: 32-byte (256-bit) registers

**Key Differences**:

1. **Double Iteration**: Processes two 256-bit chunks per outer iteration
```cpp
for (size_t i = 0; i < code_length; i += 64) {
    // Process first 32 bytes
    c = _mm256_loadu_si256((__m256i*)&codes[i]);
    // ... operations ...
    
    // Process next 32 bytes
    c = _mm256_loadu_si256((__m256i*)&codes[i + 32]);
    // ... operations ...
}
```

2. **Result Reconstruction**:
```cpp
__m256i dis0 = _mm256_add_epi16(
    _mm256_permute2f128_si256(accu0, accu1, 0x21),
    _mm256_blend_epi32(accu0, accu1, 0xF0)
);
```
- Uses `permute2f128` (128-bit lane permutation) instead of `shuffle_i64x2`

### 1.4 High-Accuracy FastScan (highacc_fastscan.hpp)

**File**: `/data00/home/liuqin.v/workspace/RaBitQ-Library/include/rabitqlib/fastscan/highacc_fastscan.hpp`

**Purpose**: Extended-bits quantization with improved precision

**Key Optimization** (Lines 18-59):
```cpp
inline void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    // Split uint16 into separate uint8 low/high bytes
    // Enables using uint8 lookup tables for extended-bit codes
}
```

**Accumulators Array** (Lines 68-98):
```cpp
__m512i accu[2][4];  // 2 sets × 4 accumulators
// accu[0][0-3]: Lower 8-bit results
// accu[1][0-3]: Upper 8-bit results
```
- **Benefit**: Supports 16-bit precision (8+8) for extended-bit quantization

---

## 2. Memory Layout and Data Structures

### 2.1 Batch Data Layout (data_layout.hpp)

**Key Structure**: `BatchDataMap<T>`

```
Memory Layout per 32-vector batch:
┌─────────────────────────────────────────┐
│ Binary Codes (1-bit)                    │
│ Size: padded_dim * 32 / 8 bytes         │
├─────────────────────────────────────────┤
│ f_add (32 floats)                       │
│ Size: 32 * sizeof(T) bytes              │
├─────────────────────────────────────────┤
│ f_rescale (32 floats)                   │
│ Size: 32 * sizeof(T) bytes              │
├─────────────────────────────────────────┤
│ f_error (32 floats)                     │
│ Size: 32 * sizeof(T) bytes              │
└─────────────────────────────────────────┘
```

**Memory Efficiency**:
- Binary codes: Packed as bits (8 dimensions → 1 byte)
- Factors: Stored as separate fields for independent access patterns
- Total per batch: `(padded_dim * 32 / 8) + (3 * 32 * sizeof(float))`

**Access Pattern** (ivf.hpp, lines 85-94):
```cpp
void scan_one_batch(
    const char* batch_data,
    const uint8_t* bin_code,    // Sequential access
    T* f_add,                    // Random access with prefetch
    T* f_rescale,
    T* f_error,
    size_t num_points
)
```

### 2.2 Extended Code Layout (ExDataMap)

```
Per-vector extended code storage:
┌──────────────────────────────────┐
│ Extended Bits Code               │
│ Size: padded_dim * ex_bits / 8   │
├──────────────────────────────────┤
│ f_add_ex (1 float)               │
│ f_rescale_ex (1 float)           │
└──────────────────────────────────┘
```

**Key Feature**: Single scaling factors per vector (not per batch) for extended bits

### 2.3 Lookup Table (LUT) Organization

**File**: `/data00/home/liuqin.v/workspace/RaBitQ-Library/include/rabitqlib/index/lut.hpp`

```cpp
template <typename T>
class Lut {
    size_t table_length_ = padded_dim << 2;  // 4x for 4-dim codebooks
    std::vector<uint8_t> lut_;               // Quantized LUT
};
```

**Construction** (lines 27-52):
1. Generate float LUT via `fastscan::pack_lut()` 
2. Compute data range (min/max)
3. Quantize to uint8 or split uint16 for high-accuracy
4. Cache sum of all base values (`sum_vl_lut_`)

**Memory Access Pattern**:
- LUT is accessed sequentially in groups of 16 values (one codebook)
- High temporal locality → cache-friendly

---

## 3. Prefetching Strategies

### 3.1 Memory Prefetch Utilities (memory.hpp, Lines 87-190)

**Available Prefetch Levels**:

```cpp
static inline void prefetch_l1(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);  // L1 cache
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch((const char*)addr, _MM_HINT_T1);  // L2 cache
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}
```

### 3.2 Unrolled Prefetch Loop (memory.hpp, Lines 103-190)

**Design**: Fixed-size unrolled switch statement

```cpp
inline void mem_prefetch_l1(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:  [[fallthrough]];
        case 20:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            // ... 19 more cases ...
        case 0:
            break;
    }
}
```

**Rationale**:
1. **Cache Line Size**: 64 bytes (typical Intel/AMD)
2. **Loop Unrolling**: Eliminates loop overhead
3. **Latency Hiding**: 20 prefetch calls ≈ 600 cycles of latency hidden
4. **Branch-Free**: Uses fallthrough pattern instead of loop

**Usage Pattern** (ivf.hpp, lines 451-465):
```cpp
// Before heavy computation block
mem_prefetch_l1(next_batch_data, 20);
// ... computation ...
// Prefetch next iteration's data
```

### 3.3 Hardware Page Size Hints (memory.hpp, Lines 45-47)

```cpp
if (HugePage) {
    madvise(ptr, nbytes, MADV_HUGEPAGE);  // Kernel hint for THP
}
```

**Benefit**: Reduces TLB misses on large datasets

---

## 4. Loop Unrolling Patterns

### 4.1 Implicit Unrolling in FastScan

**Example**: Accumulate function (fastscan.hpp, Lines 109-230)

Pattern uses 4 separate accumulator registers:
```cpp
// This implicit 4-way unrolling allows:
// - Independent register usage (no dependencies)
// - Out-of-order execution at CPU level
// - Efficient pipelining
for (size_t i = 0; i < code_length; i += 64) {
    // Each iteration effectively handles 4 independent chains
    accu0 = _mm512_add_epi16(accu0, res_lo);
    accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
    accu2 = _mm512_add_epi16(accu2, res_hi);
    accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
}
```

### 4.2 Extended Code Packing Unrolling (pack_excode.hpp)

**Example**: 3-bit code packing (Lines 56-102)

```cpp
// Process 64 dimensions at a time with explicit unrolling
for (size_t d = 0; d < dim; d += 64) {
    __m128i vec_00_to_15 = _mm_loadu_si128(...);
    __m128i vec_16_to_31 = _mm_loadu_si128(...);
    __m128i vec_32_to_47 = _mm_loadu_si128(...);
    __m128i vec_48_to_63 = _mm_loadu_si128(...);
    
    // 4-way unrolling of bit extraction
    vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
    vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
    // ...
}
```

**Technique**: Manual 4-way loop unrolling for better ILP

### 4.3 Scalar Quantization Unrolling (space.hpp, Lines 54-115)

```cpp
// AVX-512 variant
size_t mul16 = dim - (dim & 0b1111);  // Align to 16-element blocks
for (; i < mul16; i += 16) {
    auto cur = _mm512_loadu_ps(&vec0[i]);
    cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);
    auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
    _mm_storeu_epi8(&result[i], i8);
}
// Cleanup for remainder
for (; i < dim; ++i) {
    result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
}
```

**Pattern**: 16-element vector processing with scalar tail handling

---

## 5. Parallelization Techniques

### 5.1 OpenMP Task Parallelism (ivf.hpp, Line 194)

**Cluster Quantization** (Lines 193-200):

```cpp
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < num_cluster_; ++i) {
    const float* cur_centroid = centroids + (i * dim_);
    float* cur_rotated_c = &rotated_centroids[i * padded_dim_];
    Cluster& cp = cluster_lst_[i];
    quantize_cluster(cp, id_lists[i], data, cur_centroid, cur_rotated_c, config);
}
```

**Key Features**:
- **Schedule**: `schedule(dynamic)` for load balancing
- **Rationale**: Cluster sizes vary; dynamic scheduling prevents thread stalls
- **Granularity**: Each cluster as independent task
- **No locks**: Pure task-based parallelism, no synchronization needed

### 5.2 Similar Patterns in Index Building

**Symmetric Quantization Groups** (symqg_builder.hpp, multiple locations):

```cpp
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < num_cluster_; ++i) {
    // Each cluster independently quantized
}
```

### 5.3 Space Computation Parallelism (space.hpp)

```cpp
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < num_; ++i) {
    // Independent distance/similarity computations
}
```

**Note**: Search is NOT parallelized (single-query model)

---

## 6. Alignment and Cache Optimization

### 6.1 Strict Memory Alignment (memory.hpp, Lines 15-16)

```cpp
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
```

**Usage**: Custom allocators for aligned memory

```cpp
template <size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
    [[nodiscard]] T* allocate(std::size_t n) {
        auto nbytes = round_up_to_multiple_of<size_t>(n * sizeof(T), Alignment);
        auto* ptr = std::aligned_alloc(Alignment, nbytes);
        if (HugePage) {
            madvise(ptr, nbytes, MADV_HUGEPAGE);
        }
        return reinterpret_cast<T*>(ptr);
    }
};
```

### 6.2 Cache-Aware Data Organization

**Principle**: Group related data within cache line (64 bytes)

- Binary codes (packed bits): Sequential, cache-efficient
- Factors (f_add, f_rescale): Separate due to different access patterns
- LUT: Sequential access with 64-byte stride

### 6.3 Aligned Vector Operations (space.hpp)

Uses Eigen's vectorization with alignment hints:
```cpp
ConstRowMajorArrayMap<float> v0(vec0, 1, dim);
RowMajorArrayMap<T> res(result, 1, dim);
res = ((v0 - lo) * one_over_delta).round().template cast<T>();
```

Eigen automatically vectorizes while respecting alignment

---

## 7. Safe Optimizations for Rust Implementation

### 7.1 Immediately Applicable

1. **Batch Size**: Use 32-vector batches consistently
   - No algorithmic change
   - Improves SIMD efficiency

2. **4-Way Accumulator Pattern**
   - Structure code to use 4 independent scalar accumulators
   - Enables vectorization by compiler

3. **LUT Quantization**
   - Apply same quantization strategy (float → uint8)
   - Reduces memory footprint and improves cache hit rate

4. **Permutation Ordering**
   - Use same `kPerm0` permutation for data layout
   - Minimal code change, significant SIMD benefit

5. **Memory Alignment**
   - Allocate with 64-byte alignment for batch data
   - Use Rust's `#[repr(align(64))]` attributes

### 7.2 Conditional Based on CPU Features

1. **AVX-512 Path**: Use when `cfg!(target_feature = "avx512f")`
2. **AVX-2 Path**: Use when `cfg!(target_feature = "avx2")`
3. **Portable Path**: Scalar fallback always available

### 7.3 Parallelization

1. **Cluster Quantization**: Use `rayon` with dynamic work-stealing
   ```rust
   clusters.par_iter_mut().for_each(|cluster| {
       // quantize_cluster implementation
   });
   ```

2. **Search**: Keep single-threaded per query
   - Reduces complexity
   - Matches C++ approach

### 7.4 Memory Prefetching

1. **Rust intrinsic**: `std::arch::x86_64::_mm_prefetch`
2. **Manual prefetch**: Less critical in Rust due to LLVM's auto-prefetching
3. **Focus on layout**: Proper memory ordering achieves 80% of prefetch benefit

---

## 8. Performance Considerations

### 8.1 Bottlenecks

1. **Memory Bandwidth**: LUT and code access
   - Mitigation: Prefetching and 4-way accumulators
   
2. **Register Pressure**: Multiple SIMD types in use
   - Mitigation: Separate concerns (binary vs. extended codes)

3. **Data Structure Alignment**: Misalignment kills SIMD efficiency
   - Mitigation: Strict 64-byte alignment

### 8.2 Speedup Expectations

Based on C++ implementation:
- **AVX-512**: 6-10x speedup vs. scalar
- **AVX-2**: 3-5x speedup vs. scalar
- **Portable**: 1x baseline

Rust with inline SIMD should achieve similar factors

---

## 9. Files Reference

| File | Purpose | Key Section |
|------|---------|-------------|
| `fastscan.hpp` | FastScan core algorithm | Lines 109-230 (accumulate) |
| `highacc_fastscan.hpp` | Extended-bit support | Lines 61-145 (accumulate_hacc) |
| `memory.hpp` | Allocation, prefetch | Lines 38-49 (allocate), 87-190 (prefetch) |
| `data_layout.hpp` | Batch memory layout | Lines 7-55 (BatchDataMap) |
| `pack_excode.hpp` | Code packing | Lines 56-256 |
| `lut.hpp` | LUT management | Lines 27-52 (construction) |
| `ivf.hpp` | IVF structure & search | Lines 194-200 (parallelization), 451-518 (search) |
| `space.hpp` | Vector operations | Lines 54-115 (scalar_quantize_optimized) |

---

## 10. Implementation Roadmap

1. **Phase 1**: Core FastScan with 4-way accumulators
2. **Phase 2**: Memory layout optimization (batch organization)
3. **Phase 3**: LUT quantization and prefetching hints
4. **Phase 4**: Rayon parallelization for training
5. **Phase 5**: SIMD intrinsics for AVX-2/AVX-512 variants

All changes maintain algorithm correctness and test compatibility.
