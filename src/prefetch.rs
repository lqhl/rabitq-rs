/// Memory prefetch utilities for optimizing cache utilization
/// These functions provide hints to the CPU to prefetch data into cache
/// before it's needed, reducing memory access latency

/// Prefetch data into L1 cache (closest to CPU, ~4 cycle latency)
/// Use this for data that will be accessed very soon
#[inline(always)]
pub fn prefetch_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::_prefetch;
        // PLDL1KEEP - prefetch for load, L1, with temporal locality
        _prefetch(ptr as *const i8, 0, 3);
    }
}

/// Prefetch data into L2 cache (larger but slower than L1, ~12 cycle latency)
/// Use this for data that will be accessed soon but not immediately
#[inline(always)]
pub fn prefetch_l2<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::_prefetch;
        // PLDL2KEEP - prefetch for load, L2, with temporal locality
        _prefetch(ptr as *const i8, 1, 3);
    }
}

/// Prefetch data into L3 cache (largest but slowest on-chip cache, ~40 cycle latency)
/// Use this for data that will be accessed later in the pipeline
#[inline(always)]
pub fn prefetch_l3<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T2};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T2);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::_prefetch;
        // PLDL3KEEP - prefetch for load, L3, with temporal locality
        _prefetch(ptr as *const i8, 2, 3);
    }
}

/// Non-temporal prefetch (bypass cache, for streaming data)
/// Use this for data that will be accessed once and then discarded
#[inline(always)]
pub fn prefetch_nta<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};
        _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::_prefetch;
        // PLDL1STRM - prefetch for load, L1, streaming (non-temporal)
        _prefetch(ptr as *const i8, 0, 1);
    }
}

/// Prefetch multiple cache lines
/// Each cache line is typically 64 bytes
#[inline(always)]
pub fn prefetch_range_l1<T>(ptr: *const T, bytes: usize) {
    const CACHE_LINE_SIZE: usize = 64;
    let num_lines = (bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;

    for i in 0..num_lines {
        unsafe {
            let offset_ptr = (ptr as *const u8).add(i * CACHE_LINE_SIZE);
            prefetch_l1(offset_ptr);
        }
    }
}

/// Prefetch the next batch of data while processing the current batch
/// This is useful in loops where we can predict the next memory access pattern
#[inline(always)]
pub fn prefetch_next_batch<T>(current_ptr: *const T, batch_size: usize, ahead: usize) {
    unsafe {
        let next_ptr = (current_ptr as *const u8).add(batch_size * ahead);
        prefetch_range_l1(next_ptr, batch_size);
    }
}