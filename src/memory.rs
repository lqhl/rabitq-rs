//! Memory management utilities with optional huge page support
//!
//! This module provides memory allocation and management functions
//! with optional huge page support for improved TLB performance.

/// Enable huge pages for a memory region (Linux only)
///
/// This function advises the kernel to use huge pages for the given memory region.
/// Huge pages reduce TLB misses and can improve performance by 5-10%.
///
/// # Safety
/// The pointer must be valid and the size must match the allocated size.
#[cfg(all(feature = "huge_pages", target_os = "linux"))]
pub unsafe fn enable_huge_pages(ptr: *mut u8, size: usize) -> std::io::Result<()> {
    use libc::{madvise, MADV_HUGEPAGE};

    let result = madvise(ptr as *mut libc::c_void, size, MADV_HUGEPAGE);

    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Enable huge pages - no-op on non-Linux or when feature is disabled
#[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
pub unsafe fn enable_huge_pages(_ptr: *mut u8, _size: usize) -> std::io::Result<()> {
    Ok(())
}

/// Allocate memory with huge page support
///
/// This allocates a Vec<T> and optionally enables huge pages for better performance.
#[cfg(feature = "huge_pages")]
pub fn allocate_aligned_vec<T: Default + Clone>(size: usize) -> Vec<T> {
    let mut vec = vec![T::default(); size];

    // Try to enable huge pages for this allocation
    unsafe {
        let ptr = vec.as_mut_ptr() as *mut u8;
        let byte_size = size * std::mem::size_of::<T>();

        // Log if huge pages couldn't be enabled (but don't fail)
        if let Err(e) = enable_huge_pages(ptr, byte_size) {
            eprintln!("Warning: Could not enable huge pages: {}", e);
        }
    }

    vec
}

/// Allocate memory without huge page support
#[cfg(not(feature = "huge_pages"))]
pub fn allocate_aligned_vec<T: Default + Clone>(size: usize) -> Vec<T> {
    vec![T::default(); size]
}

/// Helper to check if huge pages are available on the system
#[cfg(all(feature = "huge_pages", target_os = "linux"))]
pub fn check_huge_pages_available() -> bool {
    use std::fs;

    // Check if transparent huge pages are enabled
    if let Ok(content) = fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
        return content.contains("[always]") || content.contains("[madvise]");
    }

    false
}

/// Helper to check if huge pages are available - always false when not on Linux
#[cfg(not(all(feature = "huge_pages", target_os = "linux")))]
pub fn check_huge_pages_available() -> bool {
    false
}

/// Log huge page status
pub fn log_huge_page_status() {
    #[cfg(feature = "huge_pages")]
    {
        if check_huge_pages_available() {
            eprintln!("Huge pages: ENABLED (may improve performance by 5-10%)");
        } else {
            eprintln!("Huge pages: NOT AVAILABLE (enable transparent_hugepage for better performance)");
        }
    }

    #[cfg(not(feature = "huge_pages"))]
    {
        eprintln!("Huge pages: DISABLED (compile with --features huge_pages to enable)");
    }
}