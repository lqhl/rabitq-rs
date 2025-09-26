#pragma once

// Architecture selection helpers for SIMD dispatching.
//
// The macros allow manual overrides (e.g. -DRABITQ_FORCE_SCALAR) while still
// defaulting to the host architecture when no override is provided.  The
// resulting RABITQ_TARGET_* macros evaluate to 1 when the corresponding code
// path should be compiled, and 0 otherwise.

#if defined(RABITQ_FORCE_AVX512) + defined(RABITQ_FORCE_NEON) + defined(RABITQ_FORCE_SCALAR) > 1
#error "Multiple RABITQ_FORCE_* macros defined; please select a single SIMD target"
#endif

#if defined(RABITQ_FORCE_AVX512)
#define RABITQ_TARGET_AVX512 1
#define RABITQ_TARGET_NEON 0
#define RABITQ_TARGET_SCALAR 0
#elif defined(RABITQ_FORCE_NEON)
#define RABITQ_TARGET_AVX512 0
#define RABITQ_TARGET_NEON 1
#define RABITQ_TARGET_SCALAR 0
#elif defined(RABITQ_FORCE_SCALAR)
#define RABITQ_TARGET_AVX512 0
#define RABITQ_TARGET_NEON 0
#define RABITQ_TARGET_SCALAR 1
#else
#if defined(__AVX512F__)
#define RABITQ_TARGET_AVX512 1
#else
#define RABITQ_TARGET_AVX512 0
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define RABITQ_TARGET_NEON 1
#else
#define RABITQ_TARGET_NEON 0
#endif
#if RABITQ_TARGET_AVX512 || RABITQ_TARGET_NEON
#define RABITQ_TARGET_SCALAR 0
#else
#define RABITQ_TARGET_SCALAR 1
#endif
#endif

#ifndef RABITQ_TARGET_AVX512
#define RABITQ_TARGET_AVX512 0
#endif
#ifndef RABITQ_TARGET_NEON
#define RABITQ_TARGET_NEON 0
#endif
#ifndef RABITQ_TARGET_SCALAR
#define RABITQ_TARGET_SCALAR 0
#endif

namespace rabitqlib {
constexpr bool kUseAvx512 = RABITQ_TARGET_AVX512 != 0;
constexpr bool kUseNeon = RABITQ_TARGET_NEON != 0;
constexpr bool kUseScalarFallback = RABITQ_TARGET_SCALAR != 0;
}  // namespace rabitqlib
