#!/usr/bin/env python3
"""Test LUT building correctness by comparing algorithms"""

import numpy as np

def build_lut_original(chunk):
    """Original O(256*dim) algorithm"""
    lut = np.zeros(16)
    for j in range(16):
        for d in range(len(chunk)):
            if j & (1 << d):
                lut[j] += chunk[d]
    return lut

def build_lut_dp(chunk):
    """Dynamic programming O(dim) algorithm with KPOS"""
    kpos = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3]
    lut = np.zeros(16)

    for j in range(1, 16):
        pos = kpos[j]
        if pos < len(chunk):
            if j >= (1 << pos):
                base_idx = j - (1 << pos)
                lut[j] = lut[base_idx] + chunk[pos]
            else:
                # This should never happen based on KPOS values
                lut[j] = chunk[pos]

    return lut

def test_correctness():
    """Test if both algorithms produce the same results"""
    np.random.seed(42)

    print("Testing LUT building correctness...")
    errors = []

    for test_id in range(100):
        chunk = np.random.randn(4).astype(np.float32)

        lut_orig = build_lut_original(chunk)
        lut_dp = build_lut_dp(chunk)

        if not np.allclose(lut_orig, lut_dp, rtol=1e-5):
            errors.append({
                'test_id': test_id,
                'chunk': chunk,
                'lut_orig': lut_orig,
                'lut_dp': lut_dp,
                'diff': lut_orig - lut_dp
            })

    if errors:
        print(f"✗ Found {len(errors)} errors!")
        for err in errors[:5]:  # Show first 5 errors
            print(f"\nTest {err['test_id']}:")
            print(f"  Chunk: {err['chunk']}")
            print(f"  Original LUT: {err['lut_orig']}")
            print(f"  DP LUT: {err['lut_dp']}")
            print(f"  Diff: {err['diff']}")
            print(f"  Max diff: {np.max(np.abs(err['diff']))}")
    else:
        print("✓ All tests passed! Both algorithms produce identical results.")

def analyze_kpos():
    """Analyze KPOS pattern"""
    kpos = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3]

    print("\nAnalyzing KPOS pattern:")
    for j in range(16):
        binary = format(j, '04b')
        pos = kpos[j] if j > 0 else -1

        # Find the position of the rightmost 1 bit
        rightmost_one = -1
        for i in range(4):
            if j & (1 << i):
                rightmost_one = i
                break

        print(f"  j={j:2d} ({binary}): kpos={pos}, rightmost_1={rightmost_one}")

        # Check if j >= (1 << pos) for j > 0
        if j > 0 and pos < 4:
            can_use_dp = j >= (1 << pos)
            base_idx = j - (1 << pos) if can_use_dp else -1
            print(f"      Can use DP: {can_use_dp}, base_idx={base_idx}")

if __name__ == "__main__":
    test_correctness()
    analyze_kpos()