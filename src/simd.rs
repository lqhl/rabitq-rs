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
            packed[i / 8] |= 1 << (i % 8);
        }
    }
}

#[inline]
fn unpack_binary_code_scalar(packed: &[u8], binary_code: &mut [u8], dim: usize) {
    for i in 0..dim {
        binary_code[i] = if (packed[i / 8] & (1 << (i % 8))) != 0 {
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
            code |= ((ex_code[offset + i] & 1) as u16) << i;
        }
        let out_offset = chunk * 2;
        packed[out_offset..out_offset + 2].copy_from_slice(&code.to_le_bytes());
    }

    // Handle remainder
    if remainder > 0 {
        let offset = chunks * 16;
        let mut code = 0u16;
        for i in 0..remainder {
            code |= ((ex_code[offset + i] & 1) as u16) << i;
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
