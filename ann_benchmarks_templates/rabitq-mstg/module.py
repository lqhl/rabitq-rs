"""
ann-benchmarks wrapper for RaBitQ-RS MSTG index.
"""

import numpy as np
from rabitq_rs import MstgIndex


class RabitqMstg:
    """
    BaseANN implementation for RaBitQ MSTG index.

    This wrapper provides the standard ann-benchmarks interface for the
    RaBitQ-RS MSTG (Multi-Scale Tree Graph) index.
    """

    def __init__(self, metric, index_params):
        """
        Initialize the MSTG index.

        Args:
            metric: Distance metric ('euclidean' or 'angular')
            index_params: Dictionary of index configuration parameters
        """
        self.metric = metric
        self.index_params = index_params
        self.index = None
        self.name = f"MSTG-{self._format_params(index_params)}"

    def _format_params(self, params):
        """Format parameters for display in benchmark results (keep short to avoid long filenames)."""
        # Use abbreviated keys to keep filenames short
        parts = []
        if 'max_posting_size' in params:
            parts.append(f"P{params['max_posting_size']}")
        if 'rabitq_bits' in params:
            parts.append(f"B{params['rabitq_bits']}")
        if 'hnsw_m' in params:
            parts.append(f"M{params['hnsw_m']}")
        return "-".join(parts) if parts else "default"

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
        print(f"Building MSTG index for {n} vectors of dimension {d}")

        # Create index with configuration
        self.index = MstgIndex(
            dimension=d,
            metric=self.metric,
            **self.index_params
        )

        # Build the index
        self.index.fit(X)
        print(f"Index built successfully")

    def set_query_arguments(self, query_params):
        """
        Set query-time parameters.

        Args:
            query_params: Dictionary with optional keys:
                - ef_search: Number of candidates to explore in centroid graph
                - pruning_epsilon: Dynamic pruning threshold
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call fit() first.")

        self.index.set_query_arguments(
            ef_search=query_params.get('ef_search'),
            pruning_epsilon=query_params.get('pruning_epsilon')
        )

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
        results = self.index.query(v, n)

        # Return only IDs (first column), ensuring uniqueness
        ids = results[:, 0].astype(np.int32)
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for id in ids:
            if id not in seen:
                seen.add(id)
                unique_ids.append(id)
        return np.array(unique_ids, dtype=np.int32)

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
        results = self.index.batch_query(X, n)

        # Convert to list of ID arrays, ensuring uniqueness for each result
        unique_results = []
        for r in results:
            ids = r[:, 0].astype(np.int32)
            # Remove duplicates while preserving order
            seen = set()
            unique_ids = []
            for id in ids:
                if id not in seen:
                    seen.add(id)
                    unique_ids.append(id)
            unique_results.append(np.array(unique_ids, dtype=np.int32))
        return unique_results

    def get_memory_usage(self):
        """
        Get memory usage of the index in bytes.

        Returns:
            Memory usage in bytes
        """
        if self.index is None:
            return 0
        return self.index.get_memory_usage()

    def done(self):
        """
        Cleanup method called by ann-benchmarks after benchmarking.
        """
        pass

    def get_additional(self):
        """
        Get additional metrics/information about the last query.

        Returns:
            Dictionary with additional information (empty by default)
        """
        return {}

    def __str__(self):
        return self.name
