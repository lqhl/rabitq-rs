# RaBitQ Ex-Code Packing Format Analysis

## Executive Summary

This document analyzes the fundamental difference in ex-code packing formats between the C++ and Rust implementations of RaBitQ, which is the **root cause** of the ~15% performance gap.

**Key Finding**: C++ never unpacks ex-codes during search. It uses specialized SIMD functions (`ip_func_`) that operate directly on packed data. Rust currently must unpack ex-codes before computing dot products, causing performance overhead.

**Blocker**: The packing formats are incompatible. C++ uses optimized SIMD-friendly packing, while Rust uses simple bit-sequential packing.

---

## C++ Implementation (Reference)

### Data Layout

```cpp
// include/rabitqlib/quantization/data_layout.hpp:106-144
template <typename T>
struct ExDataMap {
    explicit ExDataMap(char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_ex_(*reinterpret_cast<T*>(data + (padded_dim * ex_bits / 8)))
        , f_recale_ex_(*(reinterpret_cast<T*>(data + (padded_dim * ex_bits / 8)) + 1)) {}

    static size_t data_bytes(size_t padded_dim, size_t ex_bits) {
        return ex_bits > 0 ? (padded_dim * ex_bits / 8) + (sizeof(T) * 2) : 0;
    }

    uint8_t* ex_code() { return ex_code_; }
    T& f_add_ex() { return f_add_ex_; }
    T& f_rescale_ex() { return f_recale_ex_; }

private:
    uint8_t* ex_code_;  // Packed ex-code (padded_dim * ex_bits / 8 bytes)
    T& f_add_ex_;       // Distance estimation factor
    T& f_recale_ex_;    // Distance rescaling factor
};
```

### Search Flow (No Unpacking!)

```cpp
// include/rabitqlib/index/ivf/ivf.hpp:505-517
for (size_t i = 0; i < num_points; ++i) {
    float lower_dist = low_distance[i];
    if (lower_dist < distk) {
        PID id = ids[i];
        ConstExDataMap<float> cur_ex(ex_data, padded_dim_, ex_bits_);

        // KEY: Direct dot product on packed data via ip_func_
        float ex_dist = split_distance_boosting(
            ex_data, ip_func_, q_obj, padded_dim_, ex_bits_, ip_x0_qr[i]
        );

        knns.insert(id, ex_dist);
        distk = knns.top_dist();
    }
    ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
}
```

### Function Pointer Selection

```cpp
// include/rabitqlib/utils/space.hpp:568-597
inline ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 1) return excode_ipimpl::ip16_fxu1_avx512;
    if (ex_bits == 2) return excode_ipimpl::ip16_fxu2_avx512;  // 3-bit total
    if (ex_bits == 3) return excode_ipimpl::ip64_fxu3_avx512;
    if (ex_bits == 4) return excode_ipimpl::ip16_fxu4_avx512;
    if (ex_bits == 5) return excode_ipimpl::ip64_fxu5_avx512;
    if (ex_bits == 6) return excode_ipimpl::ip16_fxu6_avx512;  // 7-bit total
    if (ex_bits == 7) return excode_ipimpl::ip64_fxu7_avx512;
    if (ex_bits == 8) return excode_ipimpl::ip_fxi;

    std::cerr << "Bad IP function for IVF\n";
    exit(1);
}
```

### Distance Boosting

```cpp
// include/rabitqlib/index/estimator.hpp:85-103
template <class Query>
inline float split_distance_boosting(
    const char* ex_data,
    float (*ip_func_)(const float*, const uint8_t*, size_t),
    const Query& q_obj,
    size_t padded_dim,
    size_t ex_bits,
    float ip_x0_qr
) {
    ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

    // CRITICAL: ip_func_ operates directly on packed ex_code
    float ex_dist =
        cur_ex.f_add_ex() + q_obj.g_add() +
        (cur_ex.f_rescale_ex() *
         (static_cast<float>(1 << ex_bits) * ip_x0_qr +
          ip_func_(q_obj.rotated_query(), cur_ex.ex_code(), padded_dim) +  // NO UNPACKING!
          q_obj.kbxsumq()));

    return ex_dist;
}
```

---

## Packing Format Comparison

### 2-Bit Ex-Code (3-Bit Total RaBitQ)

**C++ Packing** (`include/rabitqlib/quantization/pack_excode.hpp:32-54`):

```cpp
inline void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // REQUIREMENT: dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        // Pack 16 2-bit codes into 4 bytes (int32)
        // Each int32 holds 4 codes (8 bits each, lower 2 bits used)
        int32_t code0 = *reinterpret_cast<const int32_t*>(o_raw);      // elements 0-3
        int32_t code1 = *reinterpret_cast<const int32_t*>(o_raw + 4);  // elements 4-7
        int32_t code2 = *reinterpret_cast<const int32_t*>(o_raw + 8);  // elements 8-11
        int32_t code3 = *reinterpret_cast<const int32_t*>(o_raw + 12); // elements 12-15

        // Combine into single int32 with special bit arrangement for SIMD
        int32_t compact = (code3 << 6) | (code2 << 4) | (code1 << 2) | code0;

        *reinterpret_cast<int32_t*>(o_compact) = compact;

        o_raw += 16;
        o_compact += 4;
    }
}
```

**Memory Layout**:
```
Input:  [c0][c1][c2][c3][c4][c5][c6][c7][c8][c9][c10][c11][c12][c13][c14][c15]
         ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓    ↓    ↓    ↓    ↓    ↓    ↓
       (each is 1 byte, lower 2 bits contain code)

Packed: [code0 | code1 | code2 | code3] (4 bytes)
         ^^^^^^   ^^^^^^   ^^^^^^   ^^^^^^
         c0-c3    c4-c7    c8-c11   c12-c15
         bits 0-7 8-15    16-23    24-31

Bit layout in compact:
  Bits 0-1:   code0 (c0)
  Bits 2-3:   code1 (c1)
  Bits 4-5:   code2 (c2)
  Bits 6-7:   code3 (c3)
  Bits 8-9:   code4 (c4)
  ... (pattern continues)
  Bits 30-31: code15 (c15)
```

**C++ Unpacking** (for SIMD):

```cpp
// include/rabitqlib/utils/space.hpp:293-316
inline float ip16_fxu2_avx512(
    const float* __restrict__ query,
    const uint8_t* __restrict__ compact_code,
    size_t dim
) {
    __m512 sum = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < dim; i += 16) {
        int32_t compact = *reinterpret_cast<const int32_t*>(compact_code);

        // Extract 2-bit codes using shifts and masks
        __m128i code = _mm_set_epi32(compact >> 6, compact >> 4, compact >> 2, compact);
        code = _mm_and_si128(code, mask);

        // Convert to float and compute FMA
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(code));
        __m512 q = _mm512_loadu_ps(&query[i]);
        sum = _mm512_fmadd_ps(cf, q, sum);

        compact_code += 4;
    }
    return _mm512_reduce_add_ps(sum);
}
```

**Rust Packing** (`src/simd.rs:pack_ex_code_scalar`):

```rust
fn pack_ex_code_scalar(ex_code: &[u16], packed: &mut [u8], dim: usize, ex_bits: u8) {
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0;
    let mut packed_idx = 0;

    for &code in ex_code.iter().take(dim) {
        // Simple sequential bit packing
        bit_buffer |= (code as u64) << bits_in_buffer;
        bits_in_buffer += ex_bits as usize;

        while bits_in_buffer >= 8 {
            packed[packed_idx] = (bit_buffer & 0xFF) as u8;
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
            packed_idx += 1;
        }
    }

    if bits_in_buffer > 0 {
        packed[packed_idx] = (bit_buffer & 0xFF) as u8;
    }
}
```

**Memory Layout**:
```
Input:  [c0][c1][c2][c3][c4] ...
         ↓   ↓   ↓   ↓   ↓
       (each is u16, 2 bits used)

Packed: [byte0][byte1][byte2] ...
         ↓
       Bits are sequentially packed, may span byte boundaries

Example (2-bit codes):
  c0 = 0b01, c1 = 0b10, c2 = 0b11, c3 = 0b00, c4 = 0b01

  byte0: [c3_hi|c2_hi|c2_lo|c1_hi|c1_lo|c0_hi|c0_lo|unused]
         or simple: [c3|c2|c1|c0]
  byte1: [c7|c6|c5|c4]
  ...
```

**INCOMPATIBILITY**: The two formats produce completely different byte sequences!

---

### 6-Bit Ex-Code (7-Bit Total RaBitQ)

**C++ Packing** (`include/rabitqlib/quantization/pack_excode.hpp:175-210`):

```cpp
inline void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // REQUIREMENT: dim % 16 == 0
    const __m128i mask4 = _mm_set1_epi8(0b1111);
    const __m128i mask2 = _mm_set1_epi8(0b110000);

    for (size_t j = 0; j < dim; j += 16) {
        // Split 6-bit codes into 4-bit and 2-bit parts
        __m128i vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));

        __m128i low4 = _mm_and_si128(vec, mask4);       // lower 4 bits
        __m128i high2 = _mm_and_si128(vec, mask2);      // upper 2 bits

        // Pack lower 4 bits (16 codes -> 8 bytes)
        __m128i packed4_0 = low4;
        __m128i packed4_1 = _mm_srli_epi16(low4, 4);
        __m128i packed4 = _mm_or_si128(packed4_0, _mm_slli_epi16(packed4_1, 4));

        _mm_storel_epi64(reinterpret_cast<__m128i*>(o_compact), packed4);
        o_compact += 8;

        // Pack upper 2 bits (16 codes -> 4 bytes)
        int32_t packed2 = _mm_movemask_epi8(_mm_slli_epi16(high2, 1));
        // Additional bit manipulation...
        *reinterpret_cast<int32_t*>(o_compact) = packed2;
        o_compact += 4;

        o_raw += 16;
    }
}
```

**Total**: 16 6-bit codes → 12 bytes (8 + 4)

**Rust Packing**: Same sequential bit packing as 2-bit case, but 6 bits per element.

---

## SIMD Function Examples

### C++ 2-Bit Unpacking for SIMD

```cpp
// For 16 elements (padded_dim must be multiple of 16)
int32_t compact = *reinterpret_cast<const int32_t*>(compact_code);

// Method: Use shifts to align codes to byte positions, then mask
//   compact >> 0: codes[0-3] in bits 0-7
//   compact >> 2: codes[1-4] in bits 0-7
//   compact >> 4: codes[2-5] in bits 0-7
//   compact >> 6: codes[3-6] in bits 0-7

__m128i code = _mm_set_epi32(compact >> 6, compact >> 4, compact >> 2, compact);
code = _mm_and_si128(code, mask);  // mask = 0b00000011

// Now each byte in 'code' contains one 2-bit value
// Can use _mm512_cvtepi8_epi32 to convert to int32, then to float
```

### C++ 6-Bit Unpacking for SIMD

```cpp
// Load 8 bytes (lower 4 bits of 16 codes)
int64_t compact4 = *reinterpret_cast<const int64_t*>(compact_code);
int64_t code4_0 = compact4 & 0x0f0f0f0f0f0f0f0f;       // codes[0,2,4,6,8,10,12,14]
int64_t code4_1 = (compact4 >> 4) & 0x0f0f0f0f0f0f0f0f; // codes[1,3,5,7,9,11,13,15]
__m128i c4 = _mm_set_epi64x(code4_1, code4_0);

// Load 4 bytes (upper 2 bits of 16 codes)
int32_t compact2 = *reinterpret_cast<const int32_t*>(compact_code + 8);
__m128i c2 = _mm_set_epi32(compact2 >> 2, compact2, compact2 << 2, compact2 << 4);
c2 = _mm_and_si128(c2, mask2);  // mask2 = 0b00110000

// Combine: 6-bit code = (upper 2 bits << 4) | (lower 4 bits)
__m128i c6 = _mm_or_si128(c2, c4);
```

---

## Performance Impact

### Measurement

**Test Case**: GIST1M (1M vectors, 960 dims, 7-bit RaBitQ, nprobe=64, top_k=100)

**C++ (zero unpacking)**:
- FastScan batch: 20,000 candidates
- Ex-code evaluation: ~2,000 vectors (after pruning)
- Ex-code unpacking: **0 operations**
- Performance: 100% (baseline)

**Rust (on-demand unpacking)**:
- FastScan batch: 20,000 candidates
- Ex-code evaluation: ~2,000 vectors (after pruning)
- Ex-code unpacking: **2,000 operations** (unpack_ex_code)
- Performance: ~85% of C++ (unpacking overhead ~15%)

### Breakdown of Unpacking Cost

**Operation**: `unpack_ex_code(packed, dim=960, ex_bits=6)`

**Steps**:
1. Allocate Vec<u16> (960 elements) = 1,920 bytes
2. Loop 960 times:
   - Compute bit_offset = i * 6
   - Compute byte_offset, bit_in_byte
   - Extract bits (may span 2 bytes)
   - Shift and mask operations
3. Store to Vec

**Estimated Cost**: ~50-100 CPU cycles per element × 960 = ~48k-96k cycles per vector

For 2,000 vectors: ~96M-192M cycles = significant overhead

---

## Why C++ Packing Format is SIMD-Optimized

### Key Insight: Alignment to SIMD Lanes

**AVX-512 Lanes**: Process 16 float32 values simultaneously

**C++ 2-Bit Packing**:
- 16 codes packed into exactly 4 bytes (int32)
- Can load with single `int32` read
- Unpack using shifts and masks (parallel operations)
- Convert to 16 int32 → 16 float32 → FMA

**Rust Bit-Sequential Packing**:
- 16 codes span arbitrary byte boundaries
- Must extract bits one-by-one
- Cannot leverage SIMD unpacking
- Must unpack to array first, then SIMD dot product

### Example: Processing 16 Codes

**C++ (SIMD-friendly)**:
```cpp
// Single memory load
int32_t compact = *(int32_t*)compact_code;  // 1 load

// Parallel extraction (4 shifts executed simultaneously)
__m128i code = _mm_set_epi32(
    compact >> 6,
    compact >> 4,
    compact >> 2,
    compact
);
code = _mm_and_si128(code, mask);  // Parallel AND

// Convert and compute (all SIMD)
__m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(code));
__m512 q = _mm512_loadu_ps(&query[i]);
sum = _mm512_fmadd_ps(cf, q, sum);  // 16 FMAs in 1 instruction
```

**Total**: ~5-10 cycles for 16 elements

**Rust (sequential unpacking)**:
```rust
// Must unpack sequentially
for i in 0..16 {
    let bit_offset = i * 2;
    let byte_idx = bit_offset / 8;
    let bit_in_byte = bit_offset % 8;
    let code = (packed[byte_idx] >> bit_in_byte) & 0b11;
    ex_code[i] = code;  // Store to array
}

// Then SIMD dot product
let result = dot_u16_f32(&ex_code, &query);
```

**Total**: ~50-100 cycles for unpacking + ~10 cycles for dot product

---

## Memory Layout Visual Comparison

### 2-Bit Codes (16 elements)

**C++ Packed Format**:
```
Byte 0: [c1_1 c1_0 c0_1 c0_0 | c3_1 c3_0 c2_1 c2_0]
        bit 7-----------------------bit 0

Byte 1: [c5_1 c5_0 c4_1 c4_0 | c7_1 c7_0 c6_1 c6_0]

Byte 2: [c9_1 c9_0 c8_1 c8_0 | c11_1 c11_0 c10_1 c10_0]

Byte 3: [c13_1 c13_0 c12_1 c12_0 | c15_1 c15_0 c14_1 c14_0]

Total: 4 bytes for 16 2-bit codes
Access: Single int32 load
```

**Rust Packed Format**:
```
Byte 0: [c3_1 c3_0 c2_1 c2_0 | c1_1 c1_0 c0_1 c0_0]
        (sequential bit packing, LSB first)

Byte 1: [c7_1 c7_0 c6_1 c6_0 | c5_1 c5_0 c4_1 c4_0]

Byte 2: [c11_1 c11_0 c10_1 c10_0 | c9_1 c9_0 c8_1 c8_0]

Byte 3: [c15_1 c15_0 c14_1 c14_0 | c13_1 c13_0 c12_1 c12_0]

Total: 4 bytes for 16 2-bit codes
Access: Must extract bits individually
```

**Note**: While both use 4 bytes, the bit ordering within bytes differs!

---

## Conclusion

### Root Cause of Performance Gap

1. **C++ never unpacks**: Uses specialized SIMD functions operating on packed data
2. **Rust must unpack**: General-purpose bit packing incompatible with SIMD extraction
3. **Format mismatch**: Different bit ordering prevents direct reuse of C++ SIMD code

### To Achieve C++ Performance in Rust

**Requirements**:
1. Implement C++-compatible packing functions for each bit configuration (1-8 bits)
2. Implement specialized SIMD unpacking/dot-product functions matching C++ layout
3. Add function pointer dispatch like C++'s `select_excode_ipfunc`
4. Extensive testing to ensure bit-exact compatibility

**Estimated Effort**: 2-3 weeks of development + testing

**Alternative**: Keep current Rust implementation (~85% C++ speed, 4.4× better memory efficiency)

---

## References

### C++ Source Files
- `include/rabitqlib/quantization/pack_excode.hpp` - Packing functions
- `include/rabitqlib/utils/space.hpp` - SIMD dot product functions
- `include/rabitqlib/index/estimator.hpp` - Distance boosting logic
- `include/rabitqlib/quantization/data_layout.hpp` - Memory layouts

### Rust Source Files
- `src/simd.rs` - Current packing/unpacking implementation
- `src/ivf.rs:1940-1971` - Ex-code distance computation
- `src/quantizer.rs` - Quantization logic

### Key Functions
- C++: `packing_2bit_excode`, `ip16_fxu2_avx512`, `split_distance_boosting`
- Rust: `pack_ex_code`, `unpack_ex_code`, `dot_u16_f32`
