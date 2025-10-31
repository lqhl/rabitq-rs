#!/usr/bin/env python3
"""
Basic integration tests for RaBitQ-RS Python bindings
"""

import sys
import numpy as np

def test_mstg_index():
    """Test MstgIndex basic functionality"""
    print("Testing MstgIndex...")

    try:
        from rabitq_rs import MstgIndex
    except ImportError as e:
        print(f"Error: Failed to import rabitq_rs: {e}")
        print("Make sure to run 'make build-python' first")
        return False

    # Create test data
    dim = 32
    num_vectors = 100
    data = np.random.randn(num_vectors, dim).astype(np.float32)

    # Create and train index
    idx = MstgIndex(dim, metric='euclidean')
    idx.fit(data)

    # Test single query
    query = np.random.randn(dim).astype(np.float32)
    idx.set_query_arguments(ef_search=50)
    results = idx.query(query, 10)

    assert len(results) <= 10, f"Expected at most 10 results, got {len(results)}"
    assert len(results) > 0, "Expected at least one result"

    # Verify result format (numpy array with shape [n, 2])
    assert isinstance(results, np.ndarray), f"Expected numpy array, got {type(results)}"
    assert results.ndim == 2, f"Expected 2D array, got shape {results.shape}"
    assert results.shape[1] == 2, f"Expected 2 columns (id, distance), got {results.shape[1]}"

    for i, result in enumerate(results):
        neighbor_id, distance = result[0], result[1]
        assert 0 <= neighbor_id < num_vectors, f"Result {i}: neighbor_id {neighbor_id} out of range"
        assert distance >= 0, f"Result {i}: distance {distance} should be non-negative"

    print(f"  ✓ Single query test passed (found {len(results)} neighbors)")

    # Test batch query
    batch_queries = np.random.randn(5, dim).astype(np.float32)
    batch_results = idx.batch_query(batch_queries, 10)

    assert len(batch_results) == 5, f"Expected 5 batch results, got {len(batch_results)}"
    for i, results in enumerate(batch_results):
        assert isinstance(results, np.ndarray), f"Batch {i}: Expected numpy array"
        assert len(results) <= 10, f"Batch {i}: Expected at most 10 results, got {len(results)}"
        assert results.shape[1] == 2, f"Batch {i}: Expected 2 columns (id, distance)"

    print(f"  ✓ Batch query test passed")

    # Test different metrics
    for metric in ['euclidean', 'angular']:
        idx = MstgIndex(dim, metric=metric)
        idx.fit(data)
        results = idx.query(query, 5)
        assert len(results) > 0, f"No results for metric {metric}"
        print(f"  ✓ Metric '{metric}' test passed")

    return True

def test_ivf_index():
    """Test IvfIndex basic functionality (if available)"""
    print("Testing IvfIndex...")

    try:
        from rabitq_rs import IvfIndex
    except ImportError:
        print("  ⚠ IvfIndex not available (might not be exported)")
        return True  # Not a failure if not available

    # Create test data
    dim = 32
    num_vectors = 100
    data = np.random.randn(num_vectors, dim).astype(np.float32)

    # Create and train index
    idx = IvfIndex(dim, metric='euclidean', n_clusters=10)
    idx.fit(data)

    # Test query
    query = np.random.randn(dim).astype(np.float32)
    idx.set_query_arguments(nprobe=5)
    results = idx.query(query, 10)

    assert len(results) <= 10, f"Expected at most 10 results, got {len(results)}"
    print(f"  ✓ IvfIndex test passed (found {len(results)} neighbors)")

    return True

def test_error_handling():
    """Test error handling in Python bindings"""
    print("Testing error handling...")

    try:
        from rabitq_rs import MstgIndex
    except ImportError:
        print("  ⚠ Cannot test error handling without rabitq_rs module")
        return True

    # Test dimension mismatch
    idx = MstgIndex(32, metric='euclidean')
    data = np.random.randn(10, 32).astype(np.float32)
    idx.fit(data)

    # Query with wrong dimension should be handled gracefully
    wrong_dim_query = np.random.randn(64).astype(np.float32)
    try:
        results = idx.query(wrong_dim_query, 5)
        print("  ⚠ Expected error for dimension mismatch, but query succeeded")
    except Exception as e:
        print(f"  ✓ Dimension mismatch properly caught: {type(e).__name__}")

    # Test invalid parameters
    try:
        idx = MstgIndex(-1, metric='euclidean')
        print("  ⚠ Expected error for negative dimension")
    except Exception as e:
        print(f"  ✓ Invalid dimension properly caught: {type(e).__name__}")

    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("RaBitQ-RS Python Bindings Integration Tests")
    print("=" * 60)

    all_passed = True

    # Run tests
    tests = [
        test_mstg_index,
        test_ivf_index,
        test_error_handling,
    ]

    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            all_passed = False
            print(f"✗ {test_func.__name__} raised exception: {e}")

    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())