//! SIMD-optimized bit packing functions for RaBitQ codes
//!
//! This module provides optimized implementations for packing and unpacking
//! binary codes and extended codes. On x86_64 with AVX512, it uses SIMD
//! instructions. On other platforms (including ARM), it falls back to
//! portable scalar implementations.

/// Pack binary codes (1 bit per element) into bytes
///
/// # Arguments
/// * `binary_code` - Input binary code (1 byte per element, value 0 or 1)
/// * `packed` - Output buffer for packed data
/// * `dim` - Dimension (number of elements)
#[inline]
pub fn pack_binary_code(binary_code: &[u8], packed: &mut [u8], dim: usize) {
    debug_assert_eq!(binary_code.len(), dim);
    debug_assert_eq!(packed.len(), (dim + 7) / 8);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        pack_binary_code_avx512(binary_code, packed, dim);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        pack_binary_code_scalar(binary_code, packed, dim);
    }
}

/// Unpack binary codes from packed bytes
///
/// # Arguments
/// * `packed` - Input packed data
/// * `binary_code` - Output buffer for unpacked binary code
/// * `dim` - Dimension (number of elements)
#[inline]
pub fn unpack_binary_code(packed: &[u8], binary_code: &mut [u8], dim: usize) {
    debug_assert_eq!(packed.len(), (dim + 7) / 8);
    debug_assert_eq!(binary_code.len(), dim);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        unpack_binary_code_avx512(packed, binary_code, dim);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        unpack_binary_code_scalar(packed, binary_code, dim);
    }
}

/// Pack extended codes (ex_bits per element) into bytes
///
/// # Arguments
/// * `ex_code` - Input extended code (u16 per element)
/// * `packed` - Output buffer for packed data
/// * `dim` - Dimension (number of elements)
/// * `ex_bits` - Number of bits per element (1-8)
#[inline]
pub fn pack_ex_code(ex_code: &[u16], packed: &mut [u8], dim: usize, ex_bits: u8) {
    debug_assert_eq!(ex_code.len(), dim);
    debug_assert!(ex_bits > 0 && ex_bits <= 8);

    if ex_bits == 0 {
        return;
    }

    let expected_bytes = ((dim * ex_bits as usize) + 7) / 8;
    debug_assert_eq!(packed.len(), expected_bytes);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        pack_ex_code_avx512(ex_code, packed, dim, ex_bits);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        pack_ex_code_scalar(ex_code, packed, dim, ex_bits);
    }
}

/// Unpack extended codes from packed bytes
///
/// # Arguments
/// * `packed` - Input packed data
/// * `ex_code` - Output buffer for unpacked extended code
/// * `dim` - Dimension (number of elements)
/// * `ex_bits` - Number of bits per element (1-8)
#[inline]
pub fn unpack_ex_code(packed: &[u8], ex_code: &mut [u16], dim: usize, ex_bits: u8) {
    debug_assert_eq!(ex_code.len(), dim);
    debug_assert!(ex_bits <= 8);

    if ex_bits == 0 {
        ex_code.fill(0);
        return;
    }

    let expected_bytes = ((dim * ex_bits as usize) + 7) / 8;
    debug_assert_eq!(packed.len(), expected_bytes);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        unpack_ex_code_avx512(packed, ex_code, dim, ex_bits);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        unpack_ex_code_scalar(packed, ex_code, dim, ex_bits);
    }
}

// ============================================================================
// Scalar implementations (portable, used on ARM and as fallback)
// ============================================================================

#[inline]
fn pack_binary_code_scalar(binary_code: &[u8], packed: &mut [u8], dim: usize) {
    packed.fill(0);
    for (i, &bit) in binary_code.iter().take(dim).enumerate() {
        if bit != 0 {
            // MSB-first packing to match C++ implementation
            // bit 0 goes to MSB (position 7), bit 7 goes to LSB (position 0)
            packed[i / 8] |= 1 << (7 - (i % 8));
        }
    }
}

#[inline]
fn unpack_binary_code_scalar(packed: &[u8], binary_code: &mut [u8], dim: usize) {
    for i in 0..dim {
        // MSB-first unpacking to match C++ implementation
        // bit at position 7 (MSB) is dimension 0, bit at position 0 (LSB) is dimension 7
        binary_code[i] = if (packed[i / 8] & (1 << (7 - (i % 8)))) != 0 {
            1
        } else {
            0
        };
    }
}

#[inline]
fn pack_ex_code_scalar(ex_code: &[u16], packed: &mut [u8], dim: usize, ex_bits: u8) {
    packed.fill(0);
    let bits_per_element = ex_bits as usize;

    for (i, &code) in ex_code.iter().take(dim).enumerate() {
        let bit_offset = i * bits_per_element;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let mut remaining_bits = bits_per_element;
        let mut remaining_code = code;
        let mut current_byte = byte_offset;
        let mut current_bit = bit_in_byte;

        while remaining_bits > 0 {
            let bits_in_current_byte = (8 - current_bit).min(remaining_bits);
            let mask = ((1u16 << bits_in_current_byte) - 1) as u8;
            packed[current_byte] |= ((remaining_code & mask as u16) as u8) << current_bit;

            remaining_code >>= bits_in_current_byte;
            remaining_bits -= bits_in_current_byte;
            current_byte += 1;
            current_bit = 0;
        }
    }
}

#[inline]
fn unpack_ex_code_scalar(packed: &[u8], ex_code: &mut [u16], dim: usize, ex_bits: u8) {
    let bits_per_element = ex_bits as usize;

    #[allow(clippy::needless_range_loop)]
    for i in 0..dim {
        let bit_offset = i * bits_per_element;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let mut code = 0u16;
        let mut remaining_bits = bits_per_element;
        let mut current_byte = byte_offset;
        let mut current_bit = bit_in_byte;
        let mut shift = 0;

        while remaining_bits > 0 {
            let bits_in_current_byte = (8 - current_bit).min(remaining_bits);
            let mask = ((1u16 << bits_in_current_byte) - 1) as u8;
            let bits = ((packed[current_byte] >> current_bit) & mask) as u16;
            code |= bits << shift;

            shift += bits_in_current_byte;
            remaining_bits -= bits_in_current_byte;
            current_byte += 1;
            current_bit = 0;
        }

        ex_code[i] = code;
    }
}

// ============================================================================
// AVX512 implementations (x86_64 only)
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_binary_code_avx512(binary_code: &[u8], packed: &mut [u8], dim: usize) {
    // For now, use scalar implementation
    // TODO: Implement AVX512 optimized version
    pack_binary_code_scalar(binary_code, packed, dim);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_binary_code_avx512(packed: &[u8], binary_code: &mut [u8], dim: usize) {
    // For now, use scalar implementation
    // TODO: Implement AVX512 optimized version
    unpack_binary_code_scalar(packed, binary_code, dim);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize, ex_bits: u8) {
    // Delegate to specific bit-width optimized functions
    match ex_bits {
        1 => pack_1bit_ex_code_avx512(ex_code, packed, dim),
        2 => pack_2bit_ex_code_avx512(ex_code, packed, dim),
        3 => pack_3bit_ex_code_avx512(ex_code, packed, dim),
        4 => pack_4bit_ex_code_avx512(ex_code, packed, dim),
        5 => pack_5bit_ex_code_avx512(ex_code, packed, dim),
        6 => pack_6bit_ex_code_avx512(ex_code, packed, dim),
        7 => pack_7bit_ex_code_avx512(ex_code, packed, dim),
        8 => {
            // 8-bit is just memcpy
            for (i, &code) in ex_code.iter().take(dim).enumerate() {
                packed[i] = code as u8;
            }
        }
        _ => pack_ex_code_scalar(ex_code, packed, dim, ex_bits),
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize, ex_bits: u8) {
    // Delegate to specific bit-width optimized functions
    match ex_bits {
        1 => unpack_1bit_ex_code_avx512(packed, ex_code, dim),
        2 => unpack_2bit_ex_code_avx512(packed, ex_code, dim),
        3 => unpack_3bit_ex_code_avx512(packed, ex_code, dim),
        4 => unpack_4bit_ex_code_avx512(packed, ex_code, dim),
        5 => unpack_5bit_ex_code_avx512(packed, ex_code, dim),
        6 => unpack_6bit_ex_code_avx512(packed, ex_code, dim),
        7 => unpack_7bit_ex_code_avx512(packed, ex_code, dim),
        8 => {
            // 8-bit is just memcpy
            for (i, code_out) in ex_code.iter_mut().take(dim).enumerate() {
                *code_out = packed[i] as u16;
            }
        }
        _ => unpack_ex_code_scalar(packed, ex_code, dim, ex_bits),
    }
}

// ============================================================================
// AVX512 bit-specific implementations
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_1bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    // Requires dim % 16 == 0 for optimal performance
    let chunks = dim / 16;
    let remainder = dim % 16;

    for chunk in 0..chunks {
        let offset = chunk * 16;
        let mut code = 0u16;
        for i in 0..16 {
            code |= (ex_code[offset + i] & 1) << i;
        }
        let out_offset = chunk * 2;
        packed[out_offset..out_offset + 2].copy_from_slice(&code.to_le_bytes());
    }

    // Handle remainder
    if remainder > 0 {
        let offset = chunks * 16;
        let mut code = 0u16;
        for i in 0..remainder {
            code |= (ex_code[offset + i] & 1) << i;
        }
        let out_offset = chunks * 2;
        packed[out_offset..out_offset + 2].copy_from_slice(&code.to_le_bytes());
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_1bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    let chunks = dim / 16;
    let remainder = dim % 16;

    for chunk in 0..chunks {
        let in_offset = chunk * 2;
        let code = u16::from_le_bytes([packed[in_offset], packed[in_offset + 1]]);
        let out_offset = chunk * 16;
        for i in 0..16 {
            ex_code[out_offset + i] = (code >> i) & 1;
        }
    }

    // Handle remainder
    if remainder > 0 {
        let in_offset = chunks * 2;
        let code = u16::from_le_bytes([packed[in_offset], packed[in_offset + 1]]);
        let out_offset = chunks * 16;
        for i in 0..remainder {
            ex_code[out_offset + i] = (code >> i) & 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_2bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    // Pack 16 2-bit codes into 4 bytes (32 bits)
    let chunks = dim / 16;
    let remainder = dim % 16;

    for chunk in 0..chunks {
        let offset = chunk * 16;
        let mut code = 0u32;
        for i in 0..16 {
            code |= ((ex_code[offset + i] & 0x3) as u32) << (i * 2);
        }
        let out_offset = chunk * 4;
        packed[out_offset..out_offset + 4].copy_from_slice(&code.to_le_bytes());
    }

    // Handle remainder
    if remainder > 0 {
        let offset = chunks * 16;
        let mut code = 0u32;
        for i in 0..remainder {
            code |= ((ex_code[offset + i] & 0x3) as u32) << (i * 2);
        }
        let out_offset = chunks * 4;
        packed[out_offset..out_offset + 4].copy_from_slice(&code.to_le_bytes());
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_2bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    let chunks = dim / 16;
    let remainder = dim % 16;

    for chunk in 0..chunks {
        let in_offset = chunk * 4;
        let code = u32::from_le_bytes([
            packed[in_offset],
            packed[in_offset + 1],
            packed[in_offset + 2],
            packed[in_offset + 3],
        ]);
        let out_offset = chunk * 16;
        for i in 0..16 {
            ex_code[out_offset + i] = ((code >> (i * 2)) & 0x3) as u16;
        }
    }

    // Handle remainder
    if remainder > 0 {
        let in_offset = chunks * 4;
        let code = u32::from_le_bytes([
            packed[in_offset],
            packed[in_offset + 1],
            packed[in_offset + 2],
            packed[in_offset + 3],
        ]);
        let out_offset = chunks * 16;
        for i in 0..remainder {
            ex_code[out_offset + i] = ((code >> (i * 2)) & 0x3) as u16;
        }
    }
}

// Placeholder implementations for 3-7 bits
// These can be optimized later with proper SIMD intrinsics

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_3bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    pack_ex_code_scalar(ex_code, packed, dim, 3);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_3bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    // Process 8 elements at a time (8 * 3 bits = 24 bits = 3 bytes)
    let chunks = dim / 8;
    let remainder = dim % 8;

    for chunk in 0..chunks {
        let in_offset = chunk * 3;
        let out_offset = chunk * 8;

        // Read 3 bytes = 24 bits containing 8 3-bit values
        let b0 = packed[in_offset] as u32;
        let b1 = packed[in_offset + 1] as u32;
        let b2 = packed[in_offset + 2] as u32;

        // Combine into single 32-bit value
        let data = b0 | (b1 << 8) | (b2 << 16);

        // Extract 8 3-bit values
        ex_code[out_offset] = (data & 0x7) as u16;
        ex_code[out_offset + 1] = ((data >> 3) & 0x7) as u16;
        ex_code[out_offset + 2] = ((data >> 6) & 0x7) as u16;
        ex_code[out_offset + 3] = ((data >> 9) & 0x7) as u16;
        ex_code[out_offset + 4] = ((data >> 12) & 0x7) as u16;
        ex_code[out_offset + 5] = ((data >> 15) & 0x7) as u16;
        ex_code[out_offset + 6] = ((data >> 18) & 0x7) as u16;
        ex_code[out_offset + 7] = ((data >> 21) & 0x7) as u16;
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let offset = chunks * 8;
        unpack_ex_code_scalar(&packed[chunks * 3..], &mut ex_code[offset..], remainder, 3);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_4bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    // Pack 2 4-bit codes into 1 byte
    let chunks = dim / 2;
    let remainder = dim % 2;

    for i in 0..chunks {
        let code0 = ex_code[i * 2] & 0xF;
        let code1 = ex_code[i * 2 + 1] & 0xF;
        packed[i] = (code0 as u8) | ((code1 as u8) << 4);
    }

    if remainder > 0 {
        packed[chunks] = (ex_code[chunks * 2] & 0xF) as u8;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_4bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    let chunks = dim / 2;
    let remainder = dim % 2;

    for i in 0..chunks {
        ex_code[i * 2] = (packed[i] & 0xF) as u16;
        ex_code[i * 2 + 1] = (packed[i] >> 4) as u16;
    }

    if remainder > 0 {
        ex_code[chunks * 2] = (packed[chunks] & 0xF) as u16;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_5bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    pack_ex_code_scalar(ex_code, packed, dim, 5);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_5bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    // Process 8 elements at a time (8 * 5 bits = 40 bits = 5 bytes)
    let chunks = dim / 8;
    let remainder = dim % 8;

    for chunk in 0..chunks {
        let in_offset = chunk * 5;
        let out_offset = chunk * 8;

        // Read 5 bytes = 40 bits containing 8 5-bit values
        let b0 = packed[in_offset] as u64;
        let b1 = packed[in_offset + 1] as u64;
        let b2 = packed[in_offset + 2] as u64;
        let b3 = packed[in_offset + 3] as u64;
        let b4 = packed[in_offset + 4] as u64;

        // Combine into single 64-bit value
        let data = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32);

        // Extract 8 5-bit values
        ex_code[out_offset] = (data & 0x1F) as u16;
        ex_code[out_offset + 1] = ((data >> 5) & 0x1F) as u16;
        ex_code[out_offset + 2] = ((data >> 10) & 0x1F) as u16;
        ex_code[out_offset + 3] = ((data >> 15) & 0x1F) as u16;
        ex_code[out_offset + 4] = ((data >> 20) & 0x1F) as u16;
        ex_code[out_offset + 5] = ((data >> 25) & 0x1F) as u16;
        ex_code[out_offset + 6] = ((data >> 30) & 0x1F) as u16;
        ex_code[out_offset + 7] = ((data >> 35) & 0x1F) as u16;
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let offset = chunks * 8;
        unpack_ex_code_scalar(&packed[chunks * 5..], &mut ex_code[offset..], remainder, 5);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_6bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    pack_ex_code_scalar(ex_code, packed, dim, 6);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_6bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    // Process 4 elements at a time (4 * 6 bits = 24 bits = 3 bytes)
    let chunks = dim / 4;
    let remainder = dim % 4;

    for chunk in 0..chunks {
        let in_offset = chunk * 3;
        let out_offset = chunk * 4;

        // Read 3 bytes = 24 bits containing 4 6-bit values
        let b0 = packed[in_offset] as u32;
        let b1 = packed[in_offset + 1] as u32;
        let b2 = packed[in_offset + 2] as u32;

        // Combine into single 32-bit value
        let data = b0 | (b1 << 8) | (b2 << 16);

        // Extract 4 6-bit values
        ex_code[out_offset] = (data & 0x3F) as u16;
        ex_code[out_offset + 1] = ((data >> 6) & 0x3F) as u16;
        ex_code[out_offset + 2] = ((data >> 12) & 0x3F) as u16;
        ex_code[out_offset + 3] = ((data >> 18) & 0x3F) as u16;
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let offset = chunks * 4;
        unpack_ex_code_scalar(&packed[chunks * 3..], &mut ex_code[offset..], remainder, 6);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn pack_7bit_ex_code_avx512(ex_code: &[u16], packed: &mut [u8], dim: usize) {
    pack_ex_code_scalar(ex_code, packed, dim, 7);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
unsafe fn unpack_7bit_ex_code_avx512(packed: &[u8], ex_code: &mut [u16], dim: usize) {
    // Process 8 elements at a time (8 * 7 bits = 56 bits = 7 bytes)
    let chunks = dim / 8;
    let remainder = dim % 8;

    for chunk in 0..chunks {
        let in_offset = chunk * 7;
        let out_offset = chunk * 8;

        // Read 7 bytes = 56 bits containing 8 7-bit values
        let b0 = packed[in_offset] as u64;
        let b1 = packed[in_offset + 1] as u64;
        let b2 = packed[in_offset + 2] as u64;
        let b3 = packed[in_offset + 3] as u64;
        let b4 = packed[in_offset + 4] as u64;
        let b5 = packed[in_offset + 5] as u64;
        let b6 = packed[in_offset + 6] as u64;

        // Combine into single 64-bit value
        let data = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48);

        // Extract 8 7-bit values
        ex_code[out_offset] = (data & 0x7F) as u16;
        ex_code[out_offset + 1] = ((data >> 7) & 0x7F) as u16;
        ex_code[out_offset + 2] = ((data >> 14) & 0x7F) as u16;
        ex_code[out_offset + 3] = ((data >> 21) & 0x7F) as u16;
        ex_code[out_offset + 4] = ((data >> 28) & 0x7F) as u16;
        ex_code[out_offset + 5] = ((data >> 35) & 0x7F) as u16;
        ex_code[out_offset + 6] = ((data >> 42) & 0x7F) as u16;
        ex_code[out_offset + 7] = ((data >> 49) & 0x7F) as u16;
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let offset = chunks * 8;
        unpack_ex_code_scalar(&packed[chunks * 7..], &mut ex_code[offset..], remainder, 7);
    }
}

/// SIMD-accelerated dot product: u8 vector with f32 vector
/// Converts u8 to f32 and computes dot product
#[inline]
pub fn dot_u8_f32(a: &[u8], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return dot_u8_f32_avx2(a, b);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        return dot_u8_f32_scalar(a, b);
    }
}

/// SIMD-accelerated dot product: u16 vector with f32 vector
/// Converts u16 to f32 and computes dot product
#[inline]
pub fn dot_u16_f32(a: &[u16], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return dot_u16_f32_avx2(a, b);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        return dot_u16_f32_scalar(a, b);
    }
}

#[inline]
fn dot_u8_f32_scalar(a: &[u8], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as f32) * y).sum()
}

#[inline]
fn dot_u16_f32_scalar(a: &[u16], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as f32) * y).sum()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_f32_avx2(a: &[u8], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 u8 values
        let a_u8 = _mm_loadl_epi64(a.as_ptr().add(offset) as *const __m128i);
        // Convert u8 to u32
        let a_u32 = _mm256_cvtepu8_epi32(a_u8);
        // Convert u32 to f32
        let a_f32 = _mm256_cvtepi32_ps(a_u32);

        // Load 8 f32 values
        let b_f32 = _mm256_loadu_ps(b.as_ptr().add(offset));

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(a_f32, b_f32, sum);
    }

    // Horizontal sum
    let mut result = 0.0f32;
    let sum_arr: [f32; 8] = std::mem::transmute(sum);
    for &val in &sum_arr {
        result += val;
    }

    // Handle remainder
    let offset = chunks * 8;
    for i in 0..remainder {
        result += (a[offset + i] as f32) * b[offset + i];
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_u16_f32_avx2(a: &[u16], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 u16 values
        let a_u16 = _mm_loadu_si128(a.as_ptr().add(offset) as *const __m128i);
        // Convert u16 to u32
        let a_u32 = _mm256_cvtepu16_epi32(a_u16);
        // Convert u32 to f32
        let a_f32 = _mm256_cvtepi32_ps(a_u32);

        // Load 8 f32 values
        let b_f32 = _mm256_loadu_ps(b.as_ptr().add(offset));

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(a_f32, b_f32, sum);
    }

    // Horizontal sum
    let mut result = 0.0f32;
    let sum_arr: [f32; 8] = std::mem::transmute(sum);
    for &val in &sum_arr {
        result += val;
    }

    // Handle remainder
    let offset = chunks * 8;
    for i in 0..remainder {
        result += (a[offset + i] as f32) * b[offset + i];
    }

    result
}

// ============================================================================
// FastScan: Batch processing with lookup tables for high-performance search
// ============================================================================

/// Batch size for FastScan (process 32 vectors at once)
pub const FASTSCAN_BATCH_SIZE: usize = 32;

/// Position lookup table for 4-bit codes (all combinations of 4 bits)
const KPOS: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];

/// Permutation for code packing
const KPERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Get the position of the lowest set bit (LOWBIT macro from C++)
#[inline]
fn lowbit(x: usize) -> usize {
    x & (!x + 1)
}

/// Pack lookup table for fast distance estimation
/// For each 4 dimensions, precompute 16 possible values (2^4 combinations)
/// This avoids multiplication during search
///
/// The LUT is stored as i8 values for SIMD shuffle operations
///
/// # Arguments
/// * `query` - Rotated query vector (scaled appropriately)
/// * `lut` - Output lookup table as i8 (size: dim/4 * 16)
pub fn pack_lut_i8(query: &[f32], lut: &mut [i8]) {
    assert!(
        query.len() % 4 == 0,
        "Query dimension must be multiple of 4"
    );
    let num_codebook = query.len() / 4;
    assert_eq!(
        lut.len(),
        num_codebook * 16,
        "LUT size must be num_codebook * 16"
    );

    for i in 0..num_codebook {
        let q_offset = i * 4;
        let lut_offset = i * 16;

        lut[lut_offset] = 0;
        for j in 1..16 {
            let prev_idx = j - lowbit(j);
            let sum = (lut[lut_offset + prev_idx] as f32) + query[q_offset + KPOS[j]];
            lut[lut_offset + j] = sum.round() as i8;
        }
    }
}

/// Pack lookup table as f32 (for non-SIMD paths)
pub fn pack_lut_f32(query: &[f32], lut: &mut [f32]) {
    assert!(
        query.len() % 4 == 0,
        "Query dimension must be multiple of 4"
    );
    let num_codebook = query.len() / 4;
    assert_eq!(
        lut.len(),
        num_codebook * 16,
        "LUT size must be num_codebook * 16"
    );

    for i in 0..num_codebook {
        let q_offset = i * 4;
        let lut_offset = i * 16;

        lut[lut_offset] = 0.0;
        for j in 1..16 {
            let prev_idx = j - lowbit(j);
            lut[lut_offset + j] = lut[lut_offset + prev_idx] + query[q_offset + KPOS[j]];
        }
    }
}

/// Reverse the bit order within a 4-bit pattern
///
/// Converts LSB-first to MSB-first for FastScan compatibility.
/// Example: 0b0001 (LSB=1) -> 0b1000 (MSB=1)
///          0b0110 (LSB=0,1,1) -> 0b0110 (symmetric)
#[inline]
fn reverse_4bit_pattern(pattern: u8) -> u8 {
    ((pattern & 0x01) << 3)
        | ((pattern & 0x02) << 1)
        | ((pattern & 0x04) >> 1)
        | ((pattern & 0x08) >> 3)
}

/// Pack binary codes into FastScan batch format
/// Reorganizes 32 vectors' codes for SIMD-friendly access pattern
///
/// # Arguments
/// * `codes` - Input codes (num_vectors * dim, where dim is in bytes)
/// * `num_vectors` - Number of vectors (will be rounded up to multiple of 32)
/// * `dim_bytes` - Dimension in bytes (each byte holds codes for 8 dimensions)
/// * `packed` - Output packed data
pub fn pack_codes(codes: &[u8], num_vectors: usize, dim_bytes: usize, packed: &mut [u8]) {
    let num_batches = (num_vectors + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    let expected_size = num_batches * FASTSCAN_BATCH_SIZE * dim_bytes;
    assert!(packed.len() >= expected_size, "Packed buffer too small");

    let mut packed_offset = 0;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * FASTSCAN_BATCH_SIZE;
        let batch_end = std::cmp::min(batch_start + FASTSCAN_BATCH_SIZE, num_vectors);

        // Process each byte column
        for col in 0..dim_bytes {
            let mut col_data = [0u8; FASTSCAN_BATCH_SIZE];

            // Extract column from all vectors in this batch
            for (i, vec_idx) in (batch_start..batch_end).enumerate() {
                col_data[i] = codes[vec_idx * dim_bytes + col];
            }

            // Split into upper and lower 4 bits
            let mut col_0 = [0u8; FASTSCAN_BATCH_SIZE]; // upper 4 bits
            let mut col_1 = [0u8; FASTSCAN_BATCH_SIZE]; // lower 4 bits

            for j in 0..FASTSCAN_BATCH_SIZE {
                col_0[j] = col_data[j] >> 4;
                col_1[j] = col_data[j] & 15;
            }

            // Pack according to KPERM0 permutation
            for j in 0..16 {
                let val0 = col_0[KPERM0[j]] | (col_0[KPERM0[j] + 16] << 4);
                let val1 = col_1[KPERM0[j]] | (col_1[KPERM0[j] + 16] << 4);
                packed[packed_offset + j] = val0;
                packed[packed_offset + j + 16] = val1;
            }

            packed_offset += 32;
        }
    }
}

/// Accumulate distances for a batch of 32 vectors using FastScan
/// Uses pre-computed lookup table (LUT) and SIMD shuffle for fast distance estimation
/// Automatically dispatches to AVX-512, AVX2, or scalar implementation based on CPU features
///
/// # Arguments
/// * `packed_codes` - Packed binary codes for 32 vectors
/// * `lut` - Pre-computed lookup table (as i8)
/// * `dim` - Dimension (must be multiple of 16 for SIMD)
/// * `results` - Output distances for 32 vectors
#[inline]
pub fn accumulate_batch_avx2(
    packed_codes: &[u8],
    lut: &[i8],
    dim: usize,
    results: &mut [u16; FASTSCAN_BATCH_SIZE],
) {
    assert!(dim % 16 == 0, "Dimension must be multiple of 16 for SIMD");

    #[cfg(target_arch = "x86_64")]
    {
        // Runtime CPU feature detection with fallback chain: AVX-512 -> AVX2 -> Scalar
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                // Log once per process to avoid spam
                use std::sync::Once;
                static LOG_ONCE: Once = Once::new();
                LOG_ONCE.call_once(|| {
                    eprintln!("[FastScan] Using AVX-512 optimized path");
                });
                unsafe {
                    return accumulate_batch_avx512_impl(packed_codes, lut, dim, results);
                }
            }
        }

        if is_x86_feature_detected!("avx2") {
            use std::sync::Once;
            static LOG_ONCE_AVX2: Once = Once::new();
            LOG_ONCE_AVX2.call_once(|| {
                eprintln!("[FastScan] Using AVX2 optimized path (with prefetch)");
            });
            unsafe {
                return accumulate_batch_avx2_impl(packed_codes, lut, dim, results);
            }
        }
    }

    // Fallback to scalar implementation
    accumulate_batch_scalar(packed_codes, lut, dim, results);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_batch_avx2_impl(
    packed_codes: &[u8],
    lut: &[i8],
    dim: usize,
    results: &mut [u16; FASTSCAN_BATCH_SIZE],
) {
    use std::arch::x86_64::*;

    let code_length = dim << 2; // dim * 4
    let low_mask = _mm256_set1_epi8(0x0f);

    // 4 independent accumulators following C++ logic:
    // accu0: accumulates res_lo (vectors 8-15)
    // accu1: accumulates res_lo >> 8 (vectors 0-7)
    // accu2: accumulates res_hi (vectors 24-31)
    // accu3: accumulates res_hi >> 8 (vectors 16-23)
    let mut accu0 = _mm256_setzero_si256();
    let mut accu1 = _mm256_setzero_si256();
    let mut accu2 = _mm256_setzero_si256();
    let mut accu3 = _mm256_setzero_si256();

    // Process 64 bytes at a time (2 * 32) to match C++ implementation
    let mut i = 0;
    while i + 63 < code_length {
        // Multi-level prefetching for optimal cache utilization
        // L1 cache prefetch (short distance - next iteration)
        if i + 128 < code_length {
            _mm_prefetch(packed_codes.as_ptr().add(i + 128) as *const i8, _MM_HINT_T0);
            _mm_prefetch(lut.as_ptr().add(i + 128) as *const i8, _MM_HINT_T0);
        }
        // L2 cache prefetch (medium distance)
        if i + 256 < code_length {
            _mm_prefetch(packed_codes.as_ptr().add(i + 256) as *const i8, _MM_HINT_T1);
            _mm_prefetch(lut.as_ptr().add(i + 256) as *const i8, _MM_HINT_T1);
        }
        // L3 cache prefetch (long distance)
        if i + 512 < code_length {
            _mm_prefetch(packed_codes.as_ptr().add(i + 512) as *const i8, _MM_HINT_T2);
        }

        // First 32 bytes
        let c = _mm256_loadu_si256(packed_codes.as_ptr().add(i) as *const __m256i);
        let lut_val = _mm256_loadu_si256(lut.as_ptr().add(i) as *const __m256i);
        let lo = _mm256_and_si256(c, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        let res_lo = _mm256_shuffle_epi8(lut_val, lo);
        let res_hi = _mm256_shuffle_epi8(lut_val, hi);

        // Accumulate following C++ pattern
        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        // Second 32 bytes
        let c = _mm256_loadu_si256(packed_codes.as_ptr().add(i + 32) as *const __m256i);
        let lut_val = _mm256_loadu_si256(lut.as_ptr().add(i + 32) as *const __m256i);
        let lo = _mm256_and_si256(c, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        let res_lo = _mm256_shuffle_epi8(lut_val, lo);
        let res_hi = _mm256_shuffle_epi8(lut_val, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        i += 64;
    }

    // Handle remaining bytes
    while i < code_length {
        let c = _mm256_loadu_si256(packed_codes.as_ptr().add(i) as *const __m256i);
        let lut_val = _mm256_loadu_si256(lut.as_ptr().add(i) as *const __m256i);
        let lo = _mm256_and_si256(c, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        let res_lo = _mm256_shuffle_epi8(lut_val, lo);
        let res_hi = _mm256_shuffle_epi8(lut_val, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        i += 32;
    }

    // Remove the influence of upper 8 bits for accu0 and accu2
    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));

    // Final assembly for vectors 0-15
    let dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0),
    );
    _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, dis0);

    // Final assembly for vectors 16-31
    let dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0),
    );
    _mm256_storeu_si256(results.as_mut_ptr().add(16) as *mut __m256i, dis1);
}

/// AVX-512 implementation for accumulate_batch
/// Processes 64 bytes per iteration (double the AVX2 throughput)
/// Requires AVX-512F and AVX-512BW CPU features
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn accumulate_batch_avx512_impl(
    packed_codes: &[u8],
    lut: &[i8],
    dim: usize,
    results: &mut [u16; FASTSCAN_BATCH_SIZE],
) {
    use std::arch::x86_64::*;

    let code_length = dim << 2; // dim * 4
    let low_mask = _mm512_set1_epi8(0x0f);

    let mut accu0 = _mm512_setzero_si512();
    let mut accu1 = _mm512_setzero_si512();
    let mut accu2 = _mm512_setzero_si512();
    let mut accu3 = _mm512_setzero_si512();

    // Process 64 bytes per iteration (single AVX-512 load vs. two AVX2 loads)
    // This is the key performance advantage: half the loop iterations
    for i in (0..code_length).step_by(64) {
        // Load 64 bytes of codes and LUT in one operation
        let c = _mm512_loadu_si512(packed_codes.as_ptr().add(i) as *const i32);
        let lut_val = _mm512_loadu_si512(lut.as_ptr().add(i) as *const i32);

        // Extract low and high nibbles
        // lo contains codes for vectors 0-15, hi contains codes for vectors 16-31
        let lo = _mm512_and_si512(c, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        // Look up distance values using shuffle
        let res_lo = _mm512_shuffle_epi8(lut_val, lo);
        let res_hi = _mm512_shuffle_epi8(lut_val, hi);

        // Accumulate in i16 to avoid overflow
        // Due to data layout (0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15):
        // - accu0: vectors 8-15 (lower 8 bits in each u16)
        // - accu1: vectors 0-7 (upper 8 bits need extraction)
        // - accu2: vectors 24-31 (lower 8 bits)
        // - accu3: vectors 16-23 (upper 8 bits need extraction)
        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    // Remove influence of upper 8 bits from accu0 and accu2
    accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
    accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

    // Combine results from 4 lanes into final output
    // Each accumulator contains 4 x __m128i worth of data that needs to be summed
    // The result is 32 x u16 values that fit in a single __m512i
    let ret1 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
        _mm512_shuffle_i64x2::<0b01001110>(accu0, accu1),
    );
    let ret2 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu2, accu3),
        _mm512_shuffle_i64x2::<0b01001110>(accu2, accu3),
    );

    // Final merge: combine ret1 (vecs 0-15) and ret2 (vecs 16-31)
    let mut ret = _mm512_setzero_si512();
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2::<0b10001000>(ret1, ret2));
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2::<0b11011101>(ret1, ret2));

    // Write back the 32 x u16 results
    _mm512_storeu_si512(results.as_mut_ptr() as *mut i32, ret);
}

/// High-accuracy version of accumulate_batch using int32 accumulators
/// This version prevents overflow for high-dimensional data (dim > 2048)
/// and provides better precision at the cost of slightly lower performance
pub fn accumulate_batch_highacc_avx2(
    packed_codes: &[u8],
    lut_low8: &[u8],  // Low 8 bits of LUT values
    lut_high8: &[u8], // High 8 bits of LUT values
    dim: usize,
    results: &mut [i32; FASTSCAN_BATCH_SIZE],
) {
    assert!(dim % 16 == 0, "Dimension must be multiple of 16 for SIMD");

    #[cfg(target_arch = "x86_64")]
    {
        // Runtime CPU feature detection
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                unsafe {
                    return accumulate_batch_highacc_avx512_impl(
                        packed_codes,
                        lut_low8,
                        lut_high8,
                        dim,
                        results,
                    );
                }
            }
        }

        if is_x86_feature_detected!("avx2") {
            unsafe {
                return accumulate_batch_highacc_avx2_impl(
                    packed_codes,
                    lut_low8,
                    lut_high8,
                    dim,
                    results,
                );
            }
        }
    }

    accumulate_batch_highacc_scalar(packed_codes, lut_low8, lut_high8, dim, results);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_batch_highacc_avx2_impl(
    packed_codes: &[u8],
    lut_low8: &[u8],
    lut_high8: &[u8],
    dim: usize,
    results: &mut [i32; FASTSCAN_BATCH_SIZE],
) {
    use std::arch::x86_64::*;

    let code_length = dim << 2; // dim * 4
    let low_mask = _mm256_set1_epi8(0x0f);

    // Use 32-bit accumulators to prevent overflow
    let mut accu0_low = [_mm256_setzero_si256(); 2];
    let mut accu0_high = [_mm256_setzero_si256(); 2];
    let mut accu1_low = [_mm256_setzero_si256(); 2];
    let mut accu1_high = [_mm256_setzero_si256(); 2];

    // Process 64 bytes per iteration (matching AVX2 implementation)
    for i in (0..code_length).step_by(64) {
        // Process first 32 bytes
        let c0 = _mm256_loadu_si256(packed_codes.as_ptr().add(i) as *const __m256i);
        let lut_low0 = _mm256_loadu_si256(lut_low8.as_ptr().add(i) as *const __m256i);
        let lut_high0 = _mm256_loadu_si256(lut_high8.as_ptr().add(i) as *const __m256i);

        let lo0 = _mm256_and_si256(c0, low_mask);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(c0, 4), low_mask);

        let res_lo0_low = _mm256_shuffle_epi8(lut_low0, lo0);
        let res_lo0_high = _mm256_shuffle_epi8(lut_high0, lo0);
        let res_hi0_low = _mm256_shuffle_epi8(lut_low0, hi0);
        let res_hi0_high = _mm256_shuffle_epi8(lut_high0, hi0);

        // Accumulate with sign extension to 32-bit
        let res_lo0_low_32 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(res_lo0_low, 0));
        let res_lo0_high_32 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(res_lo0_high, 0));

        accu0_low[0] = _mm256_add_epi32(accu0_low[0], res_lo0_low_32);
        accu0_high[0] = _mm256_add_epi32(accu0_high[0], res_lo0_high_32);

        // Process second 32 bytes
        if i + 32 < code_length {
            let c1 = _mm256_loadu_si256(packed_codes.as_ptr().add(i + 32) as *const __m256i);
            let lut_low1 = _mm256_loadu_si256(lut_low8.as_ptr().add(i + 32) as *const __m256i);
            let lut_high1 = _mm256_loadu_si256(lut_high8.as_ptr().add(i + 32) as *const __m256i);

            let lo1 = _mm256_and_si256(c1, low_mask);
            let hi1 = _mm256_and_si256(_mm256_srli_epi16(c1, 4), low_mask);

            let res_lo1_low = _mm256_shuffle_epi8(lut_low1, lo1);
            let res_lo1_high = _mm256_shuffle_epi8(lut_high1, lo1);
            let res_hi1_low = _mm256_shuffle_epi8(lut_low1, hi1);
            let res_hi1_high = _mm256_shuffle_epi8(lut_high1, hi1);

            let res_lo1_low_32 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(res_lo1_low, 0));
            let res_lo1_high_32 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(res_lo1_high, 0));

            accu1_low[0] = _mm256_add_epi32(accu1_low[0], res_lo1_low_32);
            accu1_high[0] = _mm256_add_epi32(accu1_high[0], res_lo1_high_32);
        }
    }

    // Combine high and low parts: result = low + (high << 8)
    const SHIFT_AMOUNT: i32 = 8;
    accu0_high[0] = _mm256_slli_epi32(accu0_high[0], SHIFT_AMOUNT);
    accu1_high[0] = _mm256_slli_epi32(accu1_high[0], SHIFT_AMOUNT);

    let final0 = _mm256_add_epi32(accu0_low[0], accu0_high[0]);
    let final1 = _mm256_add_epi32(accu1_low[0], accu1_high[0]);

    // Store results
    _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, final0);
    _mm256_storeu_si256(results.as_mut_ptr().add(8) as *mut __m256i, final1);

    // Zero out remaining entries
    for i in 16..FASTSCAN_BATCH_SIZE {
        results[i] = 0;
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn accumulate_batch_highacc_avx512_impl(
    packed_codes: &[u8],
    lut_low8: &[u8],
    lut_high8: &[u8],
    dim: usize,
    results: &mut [i32; FASTSCAN_BATCH_SIZE],
) {
    use std::arch::x86_64::*;

    let code_length = dim << 2;
    let low_mask = _mm512_set1_epi8(0x0f);

    // 512-bit accumulators for 32-bit values
    let mut accu_low = [_mm512_setzero_si512(); 2];
    let mut accu_high = [_mm512_setzero_si512(); 2];

    // Process 64 bytes per iteration with AVX-512
    for i in (0..code_length).step_by(64) {
        let c = _mm512_loadu_si512(packed_codes.as_ptr().add(i) as *const i32);
        let lut_low = _mm512_loadu_si512(lut_low8.as_ptr().add(i) as *const i32);
        let lut_high = _mm512_loadu_si512(lut_high8.as_ptr().add(i) as *const i32);

        let lo = _mm512_and_si512(c, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        let res_lo_low = _mm512_shuffle_epi8(lut_low, lo);
        let res_lo_high = _mm512_shuffle_epi8(lut_high, lo);
        let res_hi_low = _mm512_shuffle_epi8(lut_low, hi);
        let res_hi_high = _mm512_shuffle_epi8(lut_high, hi);

        // Convert to 32-bit and accumulate
        // Use _mm512_extracti32x4_epi32 to extract 128-bit chunks
        let res_lo_low_128 = _mm512_extracti32x4_epi32::<0>(res_lo_low);
        let res_lo_high_128 = _mm512_extracti32x4_epi32::<0>(res_lo_high);
        let res_lo_low_32 = _mm512_cvtepi8_epi32(res_lo_low_128);
        let res_lo_high_32 = _mm512_cvtepi8_epi32(res_lo_high_128);

        accu_low[0] = _mm512_add_epi32(accu_low[0], res_lo_low_32);
        accu_high[0] = _mm512_add_epi32(accu_high[0], res_lo_high_32);
    }

    // Combine high and low: result = low + (high << 8)
    accu_high[0] = _mm512_slli_epi32(accu_high[0], 8);
    let final_result = _mm512_add_epi32(accu_low[0], accu_high[0]);

    // Store all 32 results
    _mm512_storeu_si512(results.as_mut_ptr() as *mut i32, final_result);

    // Store second half if needed
    if FASTSCAN_BATCH_SIZE > 16 {
        _mm512_storeu_si512(results.as_mut_ptr().add(16) as *mut i32, accu_low[1]);
    }
}

/// Scalar fallback for high-accuracy accumulation
fn accumulate_batch_highacc_scalar(
    packed_codes: &[u8],
    lut_low8: &[u8],
    lut_high8: &[u8],
    dim: usize,
    results: &mut [i32; FASTSCAN_BATCH_SIZE],
) {
    results.fill(0);
    let code_length = dim * 4;

    for i in 0..code_length {
        let code_byte = packed_codes[i];
        let lo_nibble = (code_byte & 0x0F) as usize;
        let hi_nibble = ((code_byte >> 4) & 0x0F) as usize;

        let lut_base = (i / 32) * 16;
        let vec_in_block = i % 32;

        let val_lo_low = lut_low8[lut_base + lo_nibble] as i32;
        let val_lo_high = lut_high8[lut_base + lo_nibble] as i32;
        let val_hi_low = lut_low8[lut_base + hi_nibble] as i32;
        let val_hi_high = lut_high8[lut_base + hi_nibble] as i32;

        let val_lo = val_lo_low + (val_lo_high << 8);
        let val_hi = val_hi_low + (val_hi_high << 8);

        if vec_in_block < 16 {
            results[vec_in_block] += val_lo;
        } else {
            results[vec_in_block] += val_hi;
        }
    }
}

/// Scalar fallback for accumulate_batch when SIMD is not available
fn accumulate_batch_scalar(
    packed_codes: &[u8],
    lut: &[i8],
    dim: usize,
    results: &mut [u16; FASTSCAN_BATCH_SIZE],
) {
    results.fill(0);

    let code_length = dim * 4; // dim * 4 bytes for packed codes

    // Process each byte of packed codes
    for i in 0..code_length {
        let code_byte = packed_codes[i];
        let lo_nibble = (code_byte & 0x0F) as usize;
        let hi_nibble = ((code_byte >> 4) & 0x0F) as usize;

        // Look up values from LUT
        let lut_base = (i / 32) * 16; // Which LUT block we're in
        let vec_in_block = i % 32; // Which vector in the 32-vector block

        let val_lo = lut[lut_base + lo_nibble] as i16;
        let val_hi = lut[lut_base + hi_nibble] as i16;

        if vec_in_block < 16 {
            results[vec_in_block] = results[vec_in_block].wrapping_add(val_lo as u16);
        } else {
            results[vec_in_block] = results[vec_in_block].wrapping_add(val_hi as u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_binary_code() {
        let dim = 32;
        let binary_code: Vec<u8> = (0..dim).map(|i| (i % 2) as u8).collect();
        let mut packed = vec![0u8; (dim + 7) / 8];
        let mut unpacked = vec![0u8; dim];

        pack_binary_code(&binary_code, &mut packed, dim);
        unpack_binary_code(&packed, &mut unpacked, dim);

        assert_eq!(binary_code, unpacked);
    }

    #[test]
    fn test_pack_unpack_ex_code_2bit() {
        let dim = 32;
        let ex_code: Vec<u16> = (0..dim).map(|i| (i % 4) as u16).collect();
        let mut packed = vec![0u8; ((dim * 2) + 7) / 8];
        let mut unpacked = vec![0u16; dim];

        pack_ex_code(&ex_code, &mut packed, dim, 2);
        unpack_ex_code(&packed, &mut unpacked, dim, 2);

        assert_eq!(ex_code, unpacked);
    }

    #[test]
    fn test_pack_unpack_ex_code_4bit() {
        let dim = 32;
        let ex_code: Vec<u16> = (0..dim).map(|i| (i % 16) as u16).collect();
        let mut packed = vec![0u8; ((dim * 4) + 7) / 8];
        let mut unpacked = vec![0u16; dim];

        pack_ex_code(&ex_code, &mut packed, dim, 4);
        unpack_ex_code(&packed, &mut unpacked, dim, 4);

        assert_eq!(ex_code, unpacked);
    }
}
