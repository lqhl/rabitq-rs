#!/usr/bin/env python3
"""Test IvfRabitqIndex with optimizations"""

import numpy as np
import time

try:
    from rabitq_rs import IvfRabitqIndex
    print("✓ IvfRabitqIndex successfully imported")
except ImportError as e:
    print(f"✗ Failed to import IvfRabitqIndex: {e}")
    exit(1)

# Test parameters
dim = 128
num_vectors = 10000
num_queries = 100
nlist = 100

print(f"\n=== Testing IvfRabitqIndex with optimizations ===")
print(f"Dimensions: {dim}")
print(f"Vectors: {num_vectors}")
print(f"Queries: {num_queries}")
print(f"Clusters: {nlist}")

# Generate random data
np.random.seed(42)
data = np.random.randn(num_vectors, dim).astype(np.float32)
queries = np.random.randn(num_queries, dim).astype(np.float32)

# Build index
print("\nBuilding index...")
start = time.time()
index = IvfRabitqIndex(dim)
index.fit(
    data,
    nlist=nlist,
    total_bits=7,  # Use 7-bit quantization (optimized path)
    rotator_type='fht',
    seed=42,
    faster_config=False
)
build_time = time.time() - start
print(f"✓ Index built in {build_time:.3f}s")

# Search test
print("\nSearching...")
nprobe = 10
top_k = 10

# Warmup
for i in range(10):
    _ = index.query(queries[i], k=top_k, nprobe=nprobe)

# Performance measurement
start = time.time()
for i in range(num_queries):
    results = index.query(queries[i], k=top_k, nprobe=nprobe)
search_time = time.time() - start

qps = num_queries / search_time
avg_latency_ms = (search_time / num_queries) * 1000

print(f"\n=== Performance Results ===")
print(f"Total search time: {search_time:.3f}s")
print(f"QPS: {qps:.0f}")
print(f"Average latency: {avg_latency_ms:.2f}ms")
print(f"\n✓ IvfRabitqIndex with optimizations is working!")