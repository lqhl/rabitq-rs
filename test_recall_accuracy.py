#!/usr/bin/env python3
"""Test recall accuracy after fix"""

import numpy as np
from rabitq_rs import IvfRabitqIndex
from sklearn.neighbors import NearestNeighbors

# Test parameters
dim = 128
num_vectors = 5000
num_queries = 100
nlist = 50
nprobe = 10
top_k = 10

print("=== Recall Accuracy Test ===")
print(f"Dimensions: {dim}")
print(f"Vectors: {num_vectors}")
print(f"Queries: {num_queries}")
print(f"Clusters: {nlist}")
print(f"Probe: {nprobe}")
print(f"Top-K: {top_k}\n")

# Generate synthetic data
np.random.seed(42)
data = np.random.randn(num_vectors, dim).astype(np.float32)
queries = np.random.randn(num_queries, dim).astype(np.float32)

# Build ground truth using sklearn
print("Building ground truth with sklearn...")
nbrs = NearestNeighbors(n_neighbors=top_k, algorithm='brute', metric='euclidean')
nbrs.fit(data)
distances_gt, indices_gt = nbrs.kneighbors(queries)

# Build IVF-RaBitQ index
print("Building IVF-RaBitQ index...")
index = IvfRabitqIndex(dim)
index.fit(
    data,
    nlist=nlist,
    total_bits=7,
    rotator_type='random',
    seed=42,
    faster_config=False
)

# Search with IVF-RaBitQ
print(f"Searching with IVF-RaBitQ (nprobe={nprobe})...")
recalls = []
for i in range(num_queries):
    results = index.query(queries[i], k=top_k, nprobe=nprobe)
    indices_ivf = results[:, 0].astype(int)

    # Calculate recall
    recall = len(set(indices_ivf) & set(indices_gt[i])) / top_k
    recalls.append(recall)

avg_recall = np.mean(recalls)
min_recall = np.min(recalls)
max_recall = np.max(recalls)

print(f"\n=== Results ===")
print(f"Average Recall@{top_k}: {avg_recall:.3f}")
print(f"Min Recall@{top_k}: {min_recall:.3f}")
print(f"Max Recall@{top_k}: {max_recall:.3f}")

# Check if recall is reasonable (should be > 0.5 for this setup)
if avg_recall > 0.5:
    print(f"\n✓ Recall accuracy is GOOD ({avg_recall:.3f} > 0.5)")
else:
    print(f"\n✗ Recall accuracy is LOW ({avg_recall:.3f} <= 0.5) - There might be an issue!")

# Also test a very simple case with high nprobe
print(f"\n=== High nprobe test (nprobe={nlist}) ===")
results_high = index.query(queries[0], k=top_k, nprobe=nlist)
indices_high = results_high[:, 0].astype(int)
recall_high = len(set(indices_high) & set(indices_gt[0])) / top_k
print(f"Recall@{top_k} with nprobe={nlist}: {recall_high:.3f}")

if recall_high > 0.8:
    print(f"✓ High nprobe recall is excellent ({recall_high:.3f} > 0.8)")
elif recall_high > 0.6:
    print(f"✓ High nprobe recall is good ({recall_high:.3f} > 0.6)")
else:
    print(f"✗ High nprobe recall is too low ({recall_high:.3f} <= 0.6) - Serious issue!")