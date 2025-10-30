//! Python bindings for MSTG index using PyO3
#![allow(non_local_definitions)]

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::mstg::{MstgConfig, MstgIndex, ScalarPrecision, SearchParams};
use crate::rotation::RotatorType;
use crate::Metric;

#[cfg(feature = "python")]
#[pyclass(name = "MstgIndex")]
pub struct PyMstgIndex {
    index: Option<MstgIndex>,
    config: MstgConfig,
    dimension: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMstgIndex {
    /// Create a new MSTG index with configuration
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        dimension,
        metric="euclidean",
        max_posting_size=16,
        branching_factor=10,
        balance_weight=1.0,
        closure_epsilon=0.15,
        max_replicas=8,
        rabitq_bits=7,
        faster_config=true,
        hnsw_m=32,
        hnsw_ef_construction=400,
        centroid_precision="bf16",
        default_ef_search=150,
        pruning_epsilon=0.6
    ))]
    fn new(
        dimension: usize,
        metric: &str,
        max_posting_size: usize,
        branching_factor: usize,
        balance_weight: f32,
        closure_epsilon: f32,
        max_replicas: usize,
        rabitq_bits: usize,
        faster_config: bool,
        hnsw_m: usize,
        hnsw_ef_construction: usize,
        centroid_precision: &str,
        default_ef_search: usize,
        pruning_epsilon: f32,
    ) -> PyResult<Self> {
        let metric = match metric {
            "euclidean" | "l2" => Metric::L2,
            "angular" | "ip" | "inner_product" => Metric::InnerProduct,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid metric: {}. Use 'euclidean' or 'angular'",
                    metric
                )))
            }
        };

        let centroid_precision = match centroid_precision {
            "fp32" => ScalarPrecision::FP32,
            "bf16" => ScalarPrecision::BF16,
            "fp16" => ScalarPrecision::FP16,
            "int8" => ScalarPrecision::INT8,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid precision: {}. Use 'fp32', 'bf16', 'fp16', or 'int8'",
                    centroid_precision
                )))
            }
        };

        let config = MstgConfig {
            max_posting_size,
            branching_factor,
            balance_weight,
            closure_epsilon,
            max_replicas,
            rabitq_bits,
            faster_config,
            metric,
            hnsw_m,
            hnsw_ef_construction,
            centroid_precision,
            default_ef_search,
            pruning_epsilon,
        };

        Ok(Self {
            index: None,
            config,
            dimension,
        })
    }

    /// Build index from numpy array (N x D)
    fn fit(&mut self, data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let data = data.as_array();
        let shape = data.shape();

        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data must be 2D array (N x D)",
            ));
        }

        if shape[1] != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data dimension {} does not match expected {}",
                shape[1], self.dimension
            )));
        }

        // Convert to Vec<Vec<f32>>
        let n = shape[0];
        let mut vectors = Vec::with_capacity(n);

        for i in 0..n {
            let row = data.row(i);
            let vec: Vec<f32> = row.iter().copied().collect();
            vectors.push(vec);
        }

        // Build index
        match MstgIndex::build(&vectors, self.config.clone()) {
            Ok(index) => {
                self.index = Some(index);
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build index: {}",
                e
            ))),
        }
    }

    /// Set query-time search parameters
    fn set_query_arguments(&mut self, ef_search: Option<usize>, pruning_epsilon: Option<f32>) {
        if let Some(ef) = ef_search {
            self.config.default_ef_search = ef;
        }
        if let Some(eps) = pruning_epsilon {
            self.config.pruning_epsilon = eps;
        }
    }

    /// Query for k nearest neighbors
    /// Returns numpy array of shape (k, 2) with [id, distance]
    fn query(&self, py: Python, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<PyObject> {
        let index = self.index.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet. Call fit() first.")
        })?;

        let query = query.as_slice()?;

        if query.len() != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Query dimension {} does not match expected {}",
                query.len(),
                self.dimension
            )));
        }

        let params = SearchParams::new(
            self.config.default_ef_search,
            self.config.pruning_epsilon,
            k,
        );

        let results = index.search(query, &params);

        // Convert to numpy array
        let n = results.len();
        let mut data = Vec::with_capacity(n * 2);
        for result in &results {
            data.push(result.vector_id as f32);
            data.push(result.distance);
        }

        // Create 1D array then reshape to 2D
        let array_1d = PyArray1::<f32>::from_vec(py, data);
        let result_array = array_1d.reshape([n, 2]).unwrap();

        Ok(result_array.to_owned().into_py(py))
    }

    /// Batch query for multiple queries
    /// data: N x D array of queries
    /// k: number of neighbors per query
    /// Returns: list of N numpy arrays, each of shape (k, 2)
    fn batch_query(
        &self,
        py: Python,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<Vec<PyObject>> {
        let index = self.index.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet. Call fit() first.")
        })?;

        let queries = queries.as_array();
        let shape = queries.shape();

        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Queries must be 2D array (N x D)",
            ));
        }

        if shape[1] != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Query dimension {} does not match expected {}",
                shape[1], self.dimension
            )));
        }

        let n_queries = shape[0];
        let params = SearchParams::new(
            self.config.default_ef_search,
            self.config.pruning_epsilon,
            k,
        );

        // Convert to Vec<Vec<f32>> for parallel batch_search
        let query_vecs: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                let row = queries.row(i);
                row.iter().copied().collect()
            })
            .collect();

        // Parallel batch search (4-8x speedup)
        let all_results = index.batch_search(&query_vecs, &params);

        // Convert results to Python numpy arrays
        all_results
            .into_iter()
            .map(|results| {
                let n = results.len();
                let mut data = Vec::with_capacity(n * 2);
                for result in &results {
                    data.push(result.vector_id as f32);
                    data.push(result.distance);
                }

                let array_1d = PyArray1::<f32>::from_vec(py, data);
                let result_array = array_1d.reshape([n, 2]).unwrap();
                Ok(result_array.to_owned().into_py(py))
            })
            .collect()
    }

    /// Get memory usage in bytes
    fn get_memory_usage(&self) -> PyResult<usize> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index not built yet."))?;

        // Estimate memory usage
        let centroid_mem = index.centroid_index.memory_usage();
        let posting_mem: usize = index.posting_lists.iter().map(|p| p.memory_size()).sum();

        Ok(centroid_mem + posting_mem)
    }

    /// Get number of vectors in index
    fn __len__(&self) -> PyResult<usize> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index not built yet."))?;

        let total: usize = index.posting_lists.iter().map(|p| p.len()).sum();
        Ok(total)
    }

    fn __repr__(&self) -> String {
        format!(
            "MstgIndex(dimension={}, metric={:?}, built={})",
            self.dimension,
            self.config.metric,
            self.index.is_some()
        )
    }

    /// Save index to file
    fn save(&self, path: &str) -> PyResult<()> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index not built yet."))?;

        index.save_to_path(path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    /// Load index from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let index = MstgIndex::load_from_path(path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Load failed: {}", e))
        })?;

        let dimension = if !index.posting_lists.is_empty() {
            index.posting_lists[0].centroid.len()
        } else {
            0
        };

        let config = index.config.clone();

        Ok(Self {
            index: Some(index),
            config,
            dimension,
        })
    }
}

// ============================================================================
// IVF RaBitQ Index Python Bindings
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "IvfRabitqIndex")]
pub struct PyIvfRabitqIndex {
    index: Option<crate::ivf::IvfRabitqIndex>,
    dimension: usize,
    metric: Metric,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyIvfRabitqIndex {
    /// Create a new IVF RaBitQ index
    #[new]
    #[pyo3(signature = (dimension, metric="euclidean"))]
    fn new(dimension: usize, metric: &str) -> PyResult<Self> {
        let metric = match metric {
            "euclidean" | "l2" => Metric::L2,
            "angular" | "ip" | "inner_product" => Metric::InnerProduct,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid metric: {}. Use 'euclidean' or 'angular'",
                    metric
                )))
            }
        };

        Ok(Self {
            index: None,
            dimension,
            metric,
        })
    }

    /// Build index from numpy array (N x D) with automatic k-means clustering
    #[pyo3(signature = (data, nlist, total_bits=7, rotator_type="random", seed=42, faster_config=true))]
    fn fit(
        &mut self,
        data: PyReadonlyArray2<f32>,
        nlist: usize,
        total_bits: usize,
        rotator_type: &str,
        seed: u64,
        faster_config: bool,
    ) -> PyResult<()> {
        let data = data.as_array();
        let shape = data.shape();

        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data must be 2D array (N x D)",
            ));
        }

        if shape[1] != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data dimension {} does not match expected {}",
                shape[1], self.dimension
            )));
        }

        let rotator_type = match rotator_type {
            "fht" | "random" => RotatorType::FhtKacRotator,
            "matrix" | "identity" => RotatorType::MatrixRotator,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid rotator_type: {}. Use 'fht', 'random', 'matrix', or 'identity'",
                    rotator_type
                )))
            }
        };

        // Convert to Vec<Vec<f32>>
        let n = shape[0];
        let mut vectors = Vec::with_capacity(n);

        for i in 0..n {
            let row = data.row(i);
            let vec: Vec<f32> = row.iter().copied().collect();
            vectors.push(vec);
        }

        // Build index
        match crate::ivf::IvfRabitqIndex::train(
            &vectors,
            nlist,
            total_bits,
            self.metric,
            rotator_type,
            seed,
            faster_config,
        ) {
            Ok(index) => {
                self.index = Some(index);
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build index: {}",
                e
            ))),
        }
    }

    /// Build index from numpy array with pre-computed centroids and assignments
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, centroids, assignments, total_bits=7, rotator_type="random", seed=42, faster_config=true))]
    fn fit_with_clusters(
        &mut self,
        data: PyReadonlyArray2<f32>,
        centroids: PyReadonlyArray2<f32>,
        assignments: PyReadonlyArray1<i32>,
        total_bits: usize,
        rotator_type: &str,
        seed: u64,
        faster_config: bool,
    ) -> PyResult<()> {
        let data = data.as_array();
        let centroids = centroids.as_array();
        let assignments = assignments.as_slice()?;

        let data_shape = data.shape();
        let centroids_shape = centroids.shape();

        if data_shape.len() != 2 || centroids_shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data and centroids must be 2D arrays",
            ));
        }

        if data_shape[1] != self.dimension || centroids_shape[1] != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data/centroids dimension must match expected {}",
                self.dimension
            )));
        }

        if data_shape[0] != assignments.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data and assignments must have same length",
            ));
        }

        let rotator_type = match rotator_type {
            "fht" | "random" => RotatorType::FhtKacRotator,
            "matrix" | "identity" => RotatorType::MatrixRotator,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid rotator_type: {}. Use 'fht', 'random', 'matrix', or 'identity'",
                    rotator_type
                )))
            }
        };

        // Convert data to Vec<Vec<f32>>
        let n_data = data_shape[0];
        let mut data_vecs = Vec::with_capacity(n_data);
        for i in 0..n_data {
            let row = data.row(i);
            let vec: Vec<f32> = row.iter().copied().collect();
            data_vecs.push(vec);
        }

        // Convert centroids to Vec<Vec<f32>>
        let n_centroids = centroids_shape[0];
        let mut centroid_vecs = Vec::with_capacity(n_centroids);
        for i in 0..n_centroids {
            let row = centroids.row(i);
            let vec: Vec<f32> = row.iter().copied().collect();
            centroid_vecs.push(vec);
        }

        // Convert assignments to Vec<usize>
        let assignments_usize: Vec<usize> = assignments.iter().map(|&x| x as usize).collect();

        // Build index
        match crate::ivf::IvfRabitqIndex::train_with_clusters(
            &data_vecs,
            &centroid_vecs,
            &assignments_usize,
            total_bits,
            self.metric,
            rotator_type,
            seed,
            faster_config,
        ) {
            Ok(index) => {
                self.index = Some(index);
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build index with clusters: {}",
                e
            ))),
        }
    }

    /// Query for k nearest neighbors
    /// Returns numpy array of shape (k, 2) with [id, distance]
    #[pyo3(signature = (query, k, nprobe=1))]
    fn query(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
        nprobe: usize,
    ) -> PyResult<PyObject> {
        let index = self.index.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet. Call fit() first.")
        })?;

        let query = query.as_slice()?;

        if query.len() != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Query dimension {} does not match expected {}",
                query.len(),
                self.dimension
            )));
        }

        let params = crate::ivf::SearchParams::new(k, nprobe);
        let results = match index.search(query, params) {
            Ok(r) => r,
            Err(e) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Search failed: {}",
                    e
                )))
            }
        };

        // Convert to numpy array
        let n = results.len();
        let mut data = Vec::with_capacity(n * 2);
        for result in &results {
            data.push(result.id as f32);
            data.push(result.score);
        }

        // Create 1D array then reshape to 2D
        let array_1d = PyArray1::<f32>::from_vec(py, data);
        let result_array = array_1d.reshape([n, 2]).unwrap();

        Ok(result_array.to_owned().into_py(py))
    }

    /// Batch query for multiple queries
    /// data: N x D array of queries
    /// k: number of neighbors per query
    /// nprobe: number of clusters to probe
    /// Returns: list of N numpy arrays, each of shape (k, 2)
    #[pyo3(signature = (queries, k, nprobe=1))]
    fn batch_query(
        &self,
        py: Python,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        nprobe: usize,
    ) -> PyResult<Vec<PyObject>> {
        let index = self.index.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet. Call fit() first.")
        })?;

        let queries = queries.as_array();
        let shape = queries.shape();

        if shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Queries must be 2D array (N x D)",
            ));
        }

        if shape[1] != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Query dimension {} does not match expected {}",
                shape[1], self.dimension
            )));
        }

        let n_queries = shape[0];
        let params = crate::ivf::SearchParams::new(k, nprobe);

        let mut all_results = Vec::with_capacity(n_queries);

        for i in 0..n_queries {
            let query_row = queries.row(i);
            let query: Vec<f32> = query_row.iter().copied().collect();

            let results = match index.search(&query, params) {
                Ok(r) => r,
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Batch search failed at query {}: {}",
                        i, e
                    )))
                }
            };

            // Convert to numpy array
            let n = results.len();
            let mut data = Vec::with_capacity(n * 2);
            for result in &results {
                data.push(result.id as f32);
                data.push(result.score);
            }

            // Create 1D array then reshape to 2D
            let array_1d = PyArray1::<f32>::from_vec(py, data);
            let result_array = array_1d.reshape([n, 2]).unwrap();

            all_results.push(result_array.to_owned().into_py(py));
        }

        Ok(all_results)
    }

    /// Save index to file
    fn save(&self, path: &str) -> PyResult<()> {
        let index = self.index.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Index not built yet. Call fit() first.")
        })?;

        index.save_to_path(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to save index: {}", e))
        })
    }

    /// Load index from file
    fn load(&mut self, path: &str) -> PyResult<()> {
        let index = crate::ivf::IvfRabitqIndex::load_from_path(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load index: {}", e))
        })?;

        self.index = Some(index);
        Ok(())
    }

    /// Get number of vectors in index
    fn __len__(&self) -> PyResult<usize> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index not built yet."))?;

        Ok(index.len())
    }

    /// Get number of clusters
    fn cluster_count(&self) -> PyResult<usize> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Index not built yet."))?;

        Ok(index.cluster_count())
    }

    fn __repr__(&self) -> String {
        format!(
            "IvfRabitqIndex(dimension={}, metric={:?}, built={}, clusters={})",
            self.dimension,
            self.metric,
            self.index.is_some(),
            self.index
                .as_ref()
                .map(|idx| idx.cluster_count())
                .unwrap_or(0)
        )
    }
}

/// Python module initialization
#[cfg(feature = "python")]
#[pymodule]
fn rabitq_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMstgIndex>()?;
    m.add_class::<PyIvfRabitqIndex>()?;
    Ok(())
}
