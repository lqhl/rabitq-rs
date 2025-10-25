"""
ann-benchmarks wrapper for RaBitQ-RS IVF index.
"""

import numpy as np
from rabitq_rs import IvfRabitqIndex


class RabitqIvf:
    """
    BaseANN implementation for RaBitQ IVF index.

    This wrapper provides the standard ann-benchmarks interface for the
    RaBitQ-RS IVF (Inverted File) index with RaBitQ quantization.
    """

    def __init__(self, metric, index_params):
        """
        Initialize the IVF-RaBitQ index.

        Args:
            metric: Distance metric ('euclidean' or 'angular')
            index_params: Dictionary of index configuration parameters
                Required keys:
                - nlist: Number of clusters (e.g., 1024, 4096)
                Optional keys:
                - total_bits: Bits for quantization (default: 7)
                - rotator_type: 'random' or 'identity' (default: 'random')
                - seed: Random seed (default: 42)
                - faster_config: Use faster config (default: True)
        """
        self.metric = metric
        self.index_params = index_params
        self.index = None
        self.dimension = None
        self.name = f"RaBitQ-IVF({self._format_params(index_params)})"

    def _format_params(self, params):
        """Format parameters for display in benchmark results."""
        formatted = []
        for key in ['nlist', 'total_bits', 'faster_config']:
            if key in params:
                formatted.append(f"{key}={params[key]}")
        return ",".join(formatted)

    def fit(self, X):
        """
        Build the index from data.

        Args:
            X: numpy array of shape (N, D) containing training vectors
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Ensure contiguous float32 array
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        n, d = X.shape
        self.dimension = d
        print(f"Building IVF-RaBitQ index for {n} vectors of dimension {d}")

        # Extract parameters
        nlist = self.index_params.get('nlist', 1024)
        total_bits = self.index_params.get('total_bits', 7)
        rotator_type = self.index_params.get('rotator_type', 'random')
        seed = self.index_params.get('seed', 42)
        faster_config = self.index_params.get('faster_config', True)

        # Create and build index
        self.index = IvfRabitqIndex(dimension=d, metric=self.metric)
        self.index.fit(
            X,
            nlist=nlist,
            total_bits=total_bits,
            rotator_type=rotator_type,
            seed=seed,
            faster_config=faster_config
        )
        print(f"Index built with {self.index.cluster_count()} clusters")

    def set_query_arguments(self, query_params):
        """
        Set query-time parameters.

        Args:
            query_params: Dictionary with required key:
                - nprobe: Number of clusters to probe during search
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call fit() first.")

        # Store nprobe for use in query methods
        self.nprobe = query_params.get('nprobe', 1)

    def query(self, v, n):
        """
        Query for n nearest neighbors of vector v.

        Args:
            v: Query vector (1D numpy array of shape (D,))
            n: Number of nearest neighbors to return

        Returns:
            numpy array of shape (n,) containing neighbor IDs
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call fit() first.")

        if not isinstance(v, np.ndarray):
            v = np.array(v)

        # Ensure contiguous float32 array
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if not v.flags['C_CONTIGUOUS']:
            v = np.ascontiguousarray(v)

        # Query returns array of shape (k, 2) with [id, distance]
        results = self.index.query(v, k=n, nprobe=self.nprobe)

        # Return only IDs (first column)
        return results[:, 0].astype(np.int32)

    def batch_query(self, X, n):
        """
        Query for n nearest neighbors for multiple queries.

        Args:
            X: Query vectors (2D numpy array of shape (N, D))
            n: Number of nearest neighbors per query

        Returns:
            List of numpy arrays, each of shape (n,) containing neighbor IDs
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call fit() first.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Ensure contiguous float32 array
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        # Batch query returns list of arrays
        results = self.index.batch_query(X, k=n, nprobe=self.nprobe)

        # Convert to list of ID arrays
        return [r[:, 0].astype(np.int32) for r in results]

    def __str__(self):
        return self.name
