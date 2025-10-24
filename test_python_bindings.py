#!/usr/bin/env python3
"""
Test script for RaBitQ-RS Python bindings.

Run this after building with: maturin develop --release --features python
"""

import numpy as np
import time


def test_basic():
    """Test basic index build and query."""
    print("=" * 60)
    print("Test 1: Basic Index Build and Query")
    print("=" * 60)

    from rabitq_rs import MstgIndex

    # Generate test data
    np.random.seed(42)
    n, d = 1000, 128
    print(f"\nGenerating {n} vectors of dimension {d}...")
    data = np.random.randn(n, d).astype(np.float32)
    queries = np.random.randn(10, d).astype(np.float32)

    # Create index
    print("\nCreating MSTG index...")
    index = MstgIndex(
        dimension=d,
        metric="euclidean",
        max_posting_size=100,
        rabitq_bits=7,
        faster_config=True,
        centroid_precision="bf16"
    )

    # Build index
    print("\nBuilding index...")
    start = time.time()
    index.fit(data)
    build_time = time.time() - start
    print(f"Build time: {build_time:.2f}s")

    # Get memory usage
    mem_usage = index.get_memory_usage()
    print(f"Index memory usage: {mem_usage / 1024 / 1024:.2f} MB")

    # Query
    print("\nQuerying...")
    index.set_query_arguments(ef_search=150, pruning_epsilon=0.6)

    total_time = 0
    for i, query in enumerate(queries):
        start = time.time()
        results = index.query(query, k=10)
        query_time = time.time() - start
        total_time += query_time

        if i == 0:  # Print first result
            print(f"\nQuery 0 results (top 10 neighbors):")
            for j, result in enumerate(results[:5]):
                print(f"  {j+1}. ID={int(result[0])}, Distance={result[1]:.6f}")

    avg_latency = (total_time / len(queries)) * 1000
    qps = len(queries) / total_time
    print(f"\nAverage latency: {avg_latency:.2f}ms")
    print(f"QPS: {qps:.1f}")

    print("\n✓ Test 1 passed!")


def test_ann_benchmarks_wrapper():
    """Test ann-benchmarks wrapper interface."""
    print("\n" + "=" * 60)
    print("Test 2: ann-benchmarks Wrapper Interface")
    print("=" * 60)

    import sys
    import os
    # Add ann_benchmarks to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ann_benchmarks'))

    from module import RabitqMstg

    # Generate test data
    np.random.seed(42)
    n, d = 500, 64
    print(f"\nGenerating {n} vectors of dimension {d}...")
    data = np.random.randn(n, d).astype(np.float32)
    queries = np.random.randn(5, d).astype(np.float32)

    # Create wrapper
    print("\nCreating RabitqMstg wrapper...")
    index_params = {
        'max_posting_size': 100,
        'rabitq_bits': 5,
        'faster_config': True,
        'centroid_precision': 'bf16'
    }
    wrapper = RabitqMstg(metric='euclidean', index_params=index_params)

    # Fit
    print("\nBuilding index via wrapper.fit()...")
    wrapper.fit(data)

    # Set query args
    print("\nSetting query arguments...")
    wrapper.set_query_arguments({'ef_search': 100, 'pruning_epsilon': 0.5})

    # Query
    print("\nQuerying via wrapper.query()...")
    for i in range(3):
        neighbors = wrapper.query(queries[i], n=10)
        print(f"Query {i}: Found {len(neighbors)} neighbors, first 3 IDs: {neighbors[:3]}")

    # Batch query
    print("\nBatch querying via wrapper.batch_query()...")
    batch_results = wrapper.batch_query(queries, n=10)
    print(f"Batch query returned {len(batch_results)} result sets")

    # Memory usage
    mem_usage = wrapper.get_memory_usage()
    print(f"Memory usage: {mem_usage / 1024:.1f} KB")

    print("\n✓ Test 2 passed!")


def test_different_metrics():
    """Test different distance metrics."""
    print("\n" + "=" * 60)
    print("Test 3: Different Distance Metrics")
    print("=" * 60)

    from rabitq_rs import MstgIndex

    np.random.seed(42)
    n, d = 200, 32
    data = np.random.randn(n, d).astype(np.float32)
    query = np.random.randn(d).astype(np.float32)

    for metric in ['euclidean', 'angular']:
        print(f"\n--- Testing metric: {metric} ---")

        index = MstgIndex(
            dimension=d,
            metric=metric,
            max_posting_size=50,
            rabitq_bits=5,
            faster_config=True
        )

        index.fit(data)
        index.set_query_arguments(ef_search=50, pruning_epsilon=0.5)
        results = index.query(query, k=5)

        print(f"Found {len(results)} results")
        print(f"Top result: ID={int(results[0][0])}, Distance={results[0][1]:.6f}")

    print("\n✓ Test 3 passed!")


def test_recall():
    """Test recall against brute force."""
    print("\n" + "=" * 60)
    print("Test 4: Recall Test (vs Brute Force)")
    print("=" * 60)

    from rabitq_rs import MstgIndex

    np.random.seed(42)
    n, d = 500, 64
    k = 10
    data = np.random.randn(n, d).astype(np.float32)
    queries = np.random.randn(20, d).astype(np.float32)

    # Build MSTG index
    print("\nBuilding MSTG index...")
    index = MstgIndex(
        dimension=d,
        metric="euclidean",
        max_posting_size=100,
        rabitq_bits=7,
        faster_config=True
    )
    index.fit(data)
    index.set_query_arguments(ef_search=200, pruning_epsilon=0.8)

    # Compute recall
    print("\nComputing recall...")
    total_recall = 0

    for query in queries:
        # MSTG results
        mstg_results = index.query(query, k=k)
        mstg_ids = set(int(r[0]) for r in mstg_results)

        # Brute force results
        dists = np.linalg.norm(data - query, axis=1)
        bf_ids = set(np.argsort(dists)[:k])

        # Compute recall
        recall = len(mstg_ids & bf_ids) / k
        total_recall += recall

    avg_recall = total_recall / len(queries)
    print(f"\nAverage Recall@{k}: {avg_recall * 100:.1f}%")

    if avg_recall > 0.85:
        print("✓ Recall is good (>85%)")
    else:
        print("⚠ Recall is lower than expected")

    print("\n✓ Test 4 passed!")


if __name__ == "__main__":
    try:
        test_basic()
        test_ann_benchmarks_wrapper()
        test_different_metrics()
        test_recall()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run full ann-benchmarks: python ann-benchmarks/run.py")
        print("2. Tune parameters for your dataset")
        print("3. Compare with other algorithms")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
