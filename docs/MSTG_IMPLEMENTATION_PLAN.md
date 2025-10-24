# MSTG Implementation Plan

## Code Structure

```
rabitq-rs/
├── src/
│   ├── mstg/
│   │   ├── mod.rs                    # Public API and re-exports
│   │   ├── index.rs                  # MstgIndex main structure
│   │   ├── config.rs                 # Configuration structures
│   │   ├── scalar_quant.rs           # Scalar quantization (bf16, fp16, int8)
│   │   ├── posting_list.rs           # PostingList data structure
│   │   ├── metadata.rs               # Directory and metadata
│   │   ├── clustering.rs             # Hierarchical balanced clustering
│   │   ├── closure.rs                # Closure assignment with RNG
│   │   ├── hnsw.rs                   # HNSW wrapper/integration
│   │   ├── search.rs                 # Search algorithms
│   │   ├── builder.rs                # Index building pipeline
│   │   ├── io.rs                     # Disk I/O and serialization
│   │   └── tests.rs                  # Integration tests
│   ├── bin/
│   │   └── mstg.rs                   # CLI binary
│   └── lib.rs                        # Add mstg module export
```

## Milestone 1: Foundation (Week 1-2)

### Goal
Set up basic data structures and project scaffolding.

### Tasks

#### 1.1 Create Module Structure
```bash
mkdir src/mstg
touch src/mstg/{mod.rs,config.rs,scalar_quant.rs,posting_list.rs,metadata.rs}
```

#### 1.2 Define Core Types (`src/mstg/config.rs`)
```rust
use serde::{Deserialize, Serialize};
use crate::Metric;

/// Scalar quantization precision for centroid vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarPrecision {
    FP32,
    BF16,
    FP16,
    INT8,
}

impl ScalarPrecision {
    pub fn bytes_per_dim(&self) -> usize {
        match self {
            ScalarPrecision::FP32 => 4,
            ScalarPrecision::BF16 => 2,
            ScalarPrecision::FP16 => 2,
            ScalarPrecision::INT8 => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MstgConfig {
    // Clustering parameters
    pub max_posting_size: usize,
    pub branching_factor: usize,
    pub balance_weight: f32,

    // Closure assignment
    pub closure_epsilon: f32,
    pub max_replicas: usize,

    // RaBitQ parameters
    pub rabitq_bits: usize,
    pub faster_config: bool,
    pub metric: Metric,

    // HNSW parameters
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub centroid_precision: ScalarPrecision,

    // Search parameters
    pub default_ef_search: usize,
    pub pruning_epsilon: f32,
}

impl Default for MstgConfig {
    fn default() -> Self {
        Self {
            max_posting_size: 5000,
            branching_factor: 10,
            balance_weight: 1.0,
            closure_epsilon: 0.15,
            max_replicas: 8,
            rabitq_bits: 7,
            faster_config: false,
            metric: Metric::L2,
            hnsw_m: 32,
            hnsw_ef_construction: 200,
            centroid_precision: ScalarPrecision::BF16,  // Default to bf16
            default_ef_search: 150,
            pruning_epsilon: 0.6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchParams {
    pub ef_search: usize,
    pub pruning_epsilon: f32,
    pub top_k: usize,
}
```

#### 1.3 Define Posting List Structure (`src/mstg/posting_list.rs`)
```rust
use crate::{QuantizedVector, RabitqConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingList {
    pub cluster_id: u32,
    pub centroid: Vec<f32>,
    pub size: u32,
    pub rabitq_config: RabitqConfig,
    pub vectors: Vec<QuantizedVectorWithId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVectorWithId {
    pub vector_id: u64,
    pub quantized: QuantizedVector,
}

impl PostingList {
    pub fn new(cluster_id: u32, centroid: Vec<f32>) -> Self {
        Self {
            cluster_id,
            centroid,
            size: 0,
            rabitq_config: RabitqConfig::default(),
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector_id: u64, quantized: QuantizedVector) {
        self.vectors.push(QuantizedVectorWithId {
            vector_id,
            quantized,
        });
        self.size = self.vectors.len() as u32;
    }

    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.centroid.len() * std::mem::size_of::<f32>()
            + self.vectors.capacity() * std::mem::size_of::<QuantizedVectorWithId>()
    }
}
```

#### 1.4 Define Metadata Structures (`src/mstg/metadata.rs`)
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingListDirectory {
    pub entries: Vec<PostingListEntry>,
    pub centroid_to_posting: HashMap<u32, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingListEntry {
    pub cluster_id: u32,
    pub centroid_id: u32,
    pub disk_offset: u64,
    pub size_bytes: u32,
    pub num_vectors: u32,
    pub avg_norm: f32,
}

impl PostingListDirectory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            centroid_to_posting: HashMap::new(),
        }
    }

    pub fn add_entry(&mut self, entry: PostingListEntry) {
        let posting_id = self.entries.len() as u32;
        self.centroid_to_posting.insert(entry.centroid_id, posting_id);
        self.entries.push(entry);
    }

    pub fn get_posting_id(&self, centroid_id: u32) -> Option<u32> {
        self.centroid_to_posting.get(&centroid_id).copied()
    }
}
```

#### 1.5 Implement Scalar Quantization (`src/mstg/scalar_quant.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Trait for scalar-quantized vectors
pub trait QuantizedVector: Clone + Send + Sync + Sized {
    fn quantize(vector: &[f32]) -> Self;
    fn dequantize(&self) -> Vec<f32>;
    fn distance_to(&self, other: &Self) -> f32;
    fn memory_size(&self) -> usize;
}

/// BF16 (Brain Float16) quantized vector
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BF16Vector {
    data: Vec<u16>,
}

impl QuantizedVector for BF16Vector {
    fn quantize(vector: &[f32]) -> Self {
        let data = vector.iter().map(|&x| fp32_to_bf16(x)).collect();
        Self { data }
    }

    fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&x| bf16_to_fp32(x)).collect()
    }

    fn distance_to(&self, other: &Self) -> f32 {
        let mut sum = 0.0f32;
        for (a, b) in self.data.iter().zip(&other.data) {
            let diff = bf16_to_fp32(*a) - bf16_to_fp32(*b);
            sum += diff * diff;
        }
        sum
    }

    fn memory_size(&self) -> usize {
        self.data.len() * 2
    }
}

/// FP32 (no quantization)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FP32Vector {
    data: Vec<f32>,
}

impl QuantizedVector for FP32Vector {
    fn quantize(vector: &[f32]) -> Self {
        Self { data: vector.to_vec() }
    }

    fn dequantize(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn distance_to(&self, other: &Self) -> f32 {
        crate::math::squared_euclidean_distance(&self.data, &other.data)
    }

    fn memory_size(&self) -> usize {
        self.data.len() * 4
    }
}

// Conversion functions
#[inline]
pub fn fp32_to_bf16(x: f32) -> u16 {
    // BF16: sign(1) + exponent(8) + mantissa(7)
    // Truncate fp32 mantissa from 23 to 7 bits
    let bits = x.to_bits();

    // Round to nearest even (add rounding bias before truncating)
    let rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);

    (rounded >> 16) as u16
}

#[inline]
pub fn bf16_to_fp32(x: u16) -> f32 {
    // Expand bf16 back to fp32 by shifting and zero-padding mantissa
    f32::from_bits((x as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        let values = vec![1.0, -1.0, 0.5, 100.0, 0.001, std::f32::consts::PI];

        for &val in &values {
            let bf16 = fp32_to_bf16(val);
            let recovered = bf16_to_fp32(bf16);
            let error = (val - recovered).abs() / val.abs().max(1e-6);
            println!("{} -> bf16 -> {} (error: {:.6})", val, recovered, error);
            assert!(error < 0.01); // <1% error
        }
    }

    #[test]
    fn test_bf16_vector_distance() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![1.1, 2.1, 3.1, 4.1];

        let qv1 = BF16Vector::quantize(&v1);
        let qv2 = BF16Vector::quantize(&v2);

        let exact_dist = crate::math::squared_euclidean_distance(&v1, &v2);
        let approx_dist = qv1.distance_to(&qv2);

        let error = (exact_dist - approx_dist).abs() / exact_dist;
        println!("Exact: {}, Approx: {}, Error: {:.2}%", exact_dist, approx_dist, error * 100.0);
        assert!(error < 0.02); // <2% error
    }
}
```

#### 1.6 Add Dependencies to `Cargo.toml`
```toml
[dependencies]
# Existing dependencies...

# MSTG-specific
hnsw = "0.11"  # HNSW implementation
bincode = "1.3"  # Fast serialization
memmap2 = "0.9"  # Memory-mapped I/O
parking_lot = "0.12"  # Better RwLock
half = "2.3"  # FP16 support (optional, for future FP16 impl)
```

### Deliverable
- Compiling code with basic structures
- BF16/FP32 scalar quantization working
- Unit tests for data structures and quantization
- Benchmark: BF16 accuracy (<1% error) and memory savings (50%)

---

## Milestone 2: Hierarchical Balanced Clustering (Week 3)

### Goal
Implement the hierarchical balanced clustering algorithm.

### Tasks

#### 2.1 Balanced K-Means (`src/mstg/clustering.rs`)
```rust
use crate::kmeans::{KMeans, KMeansConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct BalancedKMeans {
    config: KMeansConfig,
    balance_weight: f32,
}

impl BalancedKMeans {
    pub fn new(k: usize, balance_weight: f32) -> Self {
        Self {
            config: KMeansConfig {
                k,
                max_iterations: 100,
                tolerance: 1e-4,
            },
            balance_weight,
        }
    }

    pub fn fit(&self, data: &[Vec<f32>], rng: &mut StdRng) -> BalancedClusters {
        // Implementation:
        // 1. Initialize centroids (k-means++)
        // 2. EM iterations with balance constraint
        // 3. Assignment step with capacity constraints
        // 4. Update step with weighted centroid computation
        todo!()
    }
}

#[derive(Debug)]
pub struct BalancedClusters {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub sizes: Vec<usize>,
}

impl BalancedClusters {
    pub fn max_size(&self) -> usize {
        *self.sizes.iter().max().unwrap_or(&0)
    }

    pub fn variance(&self) -> f32 {
        let mean = self.sizes.iter().sum::<usize>() as f32 / self.sizes.len() as f32;
        self.sizes
            .iter()
            .map(|&s| {
                let diff = s as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / self.sizes.len() as f32
    }
}
```

#### 2.2 Hierarchical Wrapper
```rust
pub struct HierarchicalClustering {
    max_cluster_size: usize,
    branching_factor: usize,
    balance_weight: f32,
}

impl HierarchicalClustering {
    pub fn cluster(&self, data: &[Vec<f32>]) -> Vec<Cluster> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut active_clusters = vec![Cluster::from_data(data.to_vec())];
        let mut final_clusters = Vec::new();

        while let Some(cluster) = active_clusters.pop() {
            if cluster.size() <= self.max_cluster_size {
                final_clusters.push(cluster);
            } else {
                // Split using balanced k-means
                let kmeans = BalancedKMeans::new(
                    self.branching_factor,
                    self.balance_weight,
                );
                let result = kmeans.fit(cluster.data(), &mut rng);

                // Create subclusters
                for i in 0..self.branching_factor {
                    let indices: Vec<usize> = result
                        .assignments
                        .iter()
                        .enumerate()
                        .filter(|(_, &a)| a == i)
                        .map(|(idx, _)| idx)
                        .collect();

                    let subdata: Vec<Vec<f32>> = indices
                        .iter()
                        .map(|&idx| cluster.data()[idx].clone())
                        .collect();

                    active_clusters.push(Cluster::from_data(subdata));
                }
            }
        }

        final_clusters
    }
}

#[derive(Debug, Clone)]
pub struct Cluster {
    data: Vec<Vec<f32>>,
    centroid: Vec<f32>,
}

impl Cluster {
    pub fn from_data(data: Vec<Vec<f32>>) -> Self {
        let centroid = compute_centroid(&data);
        Self { data, centroid }
    }

    pub fn data(&self) -> &[Vec<f32>] {
        &self.data
    }

    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

fn compute_centroid(data: &[Vec<f32>]) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    let dim = data[0].len();
    let mut centroid = vec![0.0; dim];

    for vec in data {
        for (i, &val) in vec.iter().enumerate() {
            centroid[i] += val;
        }
    }

    let n = data.len() as f32;
    for val in &mut centroid {
        *val /= n;
    }

    centroid
}
```

#### 2.3 Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_clustering() {
        let data = generate_test_data(10000, 128);
        let clustering = HierarchicalClustering {
            max_cluster_size: 500,
            branching_factor: 10,
            balance_weight: 1.0,
        };

        let clusters = clustering.cluster(&data);

        // Verify all clusters are within size limit
        for cluster in &clusters {
            assert!(cluster.size() <= 500);
        }

        // Verify balance (low variance)
        let sizes: Vec<usize> = clusters.iter().map(|c| c.size()).collect();
        let mean = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let variance = sizes
            .iter()
            .map(|&s| {
                let diff = s as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / sizes.len() as f32;

        println!("Mean cluster size: {}", mean);
        println!("Variance: {}", variance);
        assert!(variance < mean * mean * 0.1); // CV < 0.3
    }
}
```

### Deliverable
- Working hierarchical balanced clustering
- Tests showing balanced cluster sizes
- Benchmark comparing to flat k-means

---

## Milestone 3: Closure Assignment (Week 4)

### Goal
Implement closure assignment with RNG rule for boundary vectors.

### Tasks

#### 3.1 Closure Assignment (`src/mstg/closure.rs`)
```rust
use crate::math::{euclidean_distance, squared_euclidean_distance};

pub struct ClosureAssigner {
    epsilon: f32,
    max_replicas: usize,
}

impl ClosureAssigner {
    pub fn new(epsilon: f32, max_replicas: usize) -> Self {
        Self {
            epsilon,
            max_replicas,
        }
    }

    pub fn assign(
        &self,
        vector: &[f32],
        centroids: &[Vec<f32>],
    ) -> Vec<usize> {
        // Compute distances to all centroids
        let mut distances: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, squared_euclidean_distance(vector, c)))
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let closest_dist = distances[0].1;
        let threshold = closest_dist * (1.0 + self.epsilon);

        // Select all centroids within threshold
        let mut candidates: Vec<usize> = distances
            .iter()
            .take(self.max_replicas)
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(idx, _)| *idx)
            .collect();

        // Apply RNG rule
        candidates = self.apply_rng_rule(vector, centroids, candidates, &distances);

        candidates
    }

    fn apply_rng_rule(
        &self,
        vector: &[f32],
        centroids: &[Vec<f32>],
        candidates: Vec<usize>,
        distances: &[(usize, f32)],
    ) -> Vec<usize> {
        let mut filtered = Vec::new();

        for &i in &candidates {
            let mut should_include = true;

            // Check against previously selected centroids
            for &j in &filtered {
                let dist_i_to_v = distances.iter().find(|(idx, _)| *idx == i).unwrap().1;
                let dist_i_to_j = squared_euclidean_distance(&centroids[i], &centroids[j]);

                // RNG rule: skip i if it's farther from v than j is from i
                if dist_i_to_v > dist_i_to_j {
                    should_include = false;
                    break;
                }
            }

            if should_include {
                filtered.push(i);
            }
        }

        // Always include the closest centroid
        if !filtered.contains(&candidates[0]) {
            filtered.insert(0, candidates[0]);
        }

        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closure_assignment() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
        ];

        let assigner = ClosureAssigner::new(0.2, 8);

        // Vector close to centroid 0
        let v1 = vec![0.1, 0.0];
        let assigned = assigner.assign(&v1, &centroids);
        println!("Vector {:?} assigned to {:?}", v1, assigned);
        assert_eq!(assigned[0], 0); // Closest is 0

        // Boundary vector between 0 and 1
        let v2 = vec![0.45, 0.0];
        let assigned = assigner.assign(&v2, &centroids);
        println!("Vector {:?} assigned to {:?}", v2, assigned);
        assert!(assigned.len() >= 2); // Should be assigned to multiple
    }
}
```

### Deliverable
- Working closure assignment
- RNG rule reducing redundancy
- Visualization of assignments

---

## Milestone 4: RaBitQ Integration (Week 5-6)

### Goal
Integrate RaBitQ quantization into posting lists.

### Tasks

#### 4.1 Per-Cluster Quantization
```rust
// In src/mstg/posting_list.rs

impl PostingList {
    pub fn quantize_vectors(
        &mut self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        rabitq_bits: usize,
        metric: Metric,
        faster_config: bool,
    ) -> Result<(), String> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Compute residuals
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| crate::math::subtract(v, &self.centroid))
            .collect();

        // Train RaBitQ on residuals
        use crate::quantizer::quantize_with_centroid;
        use crate::RabitqConfig;

        // Create dummy centroid (all zeros) since we already computed residuals
        let dim = self.centroid.len();
        let zero_centroid = vec![0.0; dim];

        // Train config on residuals
        self.rabitq_config = RabitqConfig::train(
            &residuals,
            rabitq_bits,
            metric,
            faster_config,
        )?;

        // Quantize all residuals
        for (i, residual) in residuals.iter().enumerate() {
            let quantized = quantize_with_centroid(
                residual,
                &zero_centroid,
                &self.rabitq_config,
            )?;

            self.add_vector(vector_ids[i], quantized);
        }

        Ok(())
    }

    pub fn estimate_distance(&self, query: &[f32], vec_id: usize) -> f32 {
        let qvec = &self.vectors[vec_id];

        // Compute query residual
        let query_residual = crate::math::subtract(query, &self.centroid);

        // Estimate distance using RaBitQ
        self.rabitq_config.estimate_distance(&query_residual, &qvec.quantized)
    }
}
```

#### 4.2 Benchmark Quantization Accuracy
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_quantization_accuracy() {
        // Generate synthetic cluster
        let centroid = vec![0.5; 128];
        let vectors = generate_cluster_data(&centroid, 1000, 0.1);

        let mut plist = PostingList::new(0, centroid.clone());
        let ids: Vec<u64> = (0..1000).collect();

        plist.quantize_vectors(&vectors, &ids, 7, Metric::L2, false).unwrap();

        // Compute distance error
        let query = generate_random_vector(128);
        let mut errors = Vec::new();

        for i in 0..1000 {
            let exact = crate::math::squared_euclidean_distance(&query, &vectors[i]);
            let approx = plist.estimate_distance(&query, i);
            let error = (exact - approx).abs() / exact;
            errors.push(error);
        }

        let mean_error: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
        println!("Mean relative error: {:.2}%", mean_error * 100.0);
        assert!(mean_error < 0.05); // <5% error
    }
}
```

### Deliverable
- Per-cluster RaBitQ quantization working
- Distance estimation accurate within 5%
- Memory savings verified

---

## Milestone 5: HNSW Integration with Scalar Quantization (Week 7)

### Goal
Build and query HNSW index for centroids with bf16/fp32 quantization support.

### Tasks

#### 5.1 HNSW Wrapper with Scalar Quantization (`src/mstg/hnsw.rs`)
```rust
use hnsw::{Hnsw, Params};
use crate::mstg::scalar_quant::{QuantizedVector, BF16Vector, FP32Vector};
use crate::mstg::config::ScalarPrecision;

/// Centroid index that supports different scalar quantization precisions
pub struct CentroidIndex {
    precision: ScalarPrecision,
    centroid_ids: Vec<u32>,
    // Store quantized vectors based on precision
    quantized_centroids: QuantizedCentroids,
    // HNSW works on dequantized vectors
    hnsw: Hnsw<f32, Vec<f32>>,
}

enum QuantizedCentroids {
    FP32(Vec<FP32Vector>),
    BF16(Vec<BF16Vector>),
    // FP16(Vec<FP16Vector>),  // Future
    // INT8(Vec<INT8Vector>),  // Future
}

impl CentroidIndex {
    pub fn build(
        centroids: Vec<Vec<f32>>,
        centroid_ids: Vec<u32>,
        m: usize,
        ef_construction: usize,
        precision: ScalarPrecision,
    ) -> Self {
        // Quantize centroids based on precision
        let quantized_centroids = match precision {
            ScalarPrecision::FP32 => {
                let quant: Vec<FP32Vector> = centroids
                    .iter()
                    .map(|c| FP32Vector::quantize(c))
                    .collect();
                QuantizedCentroids::FP32(quant)
            }
            ScalarPrecision::BF16 => {
                let quant: Vec<BF16Vector> = centroids
                    .iter()
                    .map(|c| BF16Vector::quantize(c))
                    .collect();
                QuantizedCentroids::BF16(quant)
            }
            _ => panic!("Unsupported precision: {:?}", precision),
        };

        // Build HNSW on quantized centroids (dequantize for HNSW)
        let params = Params::new()
            .m(m)
            .ef_construction(ef_construction);

        let mut hnsw = Hnsw::new(params);

        for (i, centroid) in centroids.iter().enumerate() {
            // HNSW uses dequantized vectors for graph building
            let dequantized = Self::dequantize_at(&quantized_centroids, i);
            hnsw.insert(dequantized, i);
        }

        Self {
            precision,
            centroid_ids,
            quantized_centroids,
            hnsw,
        }
    }

    fn dequantize_at(centroids: &QuantizedCentroids, idx: usize) -> Vec<f32> {
        match centroids {
            QuantizedCentroids::FP32(vecs) => vecs[idx].dequantize(),
            QuantizedCentroids::BF16(vecs) => vecs[idx].dequantize(),
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        ef_search: usize,
        k: usize,
    ) -> Vec<(u32, f32)> {
        self.hnsw.set_ef(ef_search);
        let results = self.hnsw.search(query, k);

        results
            .into_iter()
            .map(|(idx, dist)| (self.centroid_ids[idx], dist))
            .collect()
    }

    pub fn memory_usage(&self) -> usize {
        // Quantized vector storage
        let vec_size = match &self.quantized_centroids {
            QuantizedCentroids::FP32(vecs) => {
                vecs.iter().map(|v| v.memory_size()).sum::<usize>()
            }
            QuantizedCentroids::BF16(vecs) => {
                vecs.iter().map(|v| v.memory_size()).sum::<usize>()
            }
        };

        // Graph edges
        let graph_size = self.hnsw.len() * std::mem::size_of::<u32>() * 32;

        vec_size + graph_size
    }

    pub fn get_centroid(&self, idx: usize) -> Vec<f32> {
        Self::dequantize_at(&self.quantized_centroids, idx)
    }
}
```

### Deliverable
- HNSW building and searching with scalar quantization
- Benchmark: HNSW with BF16 vs FP32 (recall, latency, memory)
- Verify <0.1% recall loss with BF16
- Confirm 50% memory savings with BF16
- Comparison: HNSW vs linear scan

---

## Milestone 6: Index Building Pipeline (Week 8)

### Goal
Integrate all components into end-to-end index builder.

### Tasks

#### 6.1 Builder (`src/mstg/builder.rs`)
```rust
use super::*;
use rayon::prelude::*;

pub struct MstgBuilder {
    config: MstgConfig,
}

impl MstgBuilder {
    pub fn new(config: MstgConfig) -> Self {
        Self { config }
    }

    pub fn build(
        &self,
        data: &[Vec<f32>],
    ) -> Result<MstgIndex, String> {
        println!("Building MSTG index for {} vectors", data.len());

        // Step 1: Hierarchical balanced clustering
        println!("Step 1: Hierarchical balanced clustering...");
        let clustering = HierarchicalClustering {
            max_cluster_size: self.config.max_posting_size,
            branching_factor: self.config.branching_factor,
            balance_weight: self.config.balance_weight,
        };
        let clusters = clustering.cluster(data);
        println!("Created {} clusters", clusters.len());

        // Step 2: Closure assignment
        println!("Step 2: Closure assignment...");
        let centroids: Vec<Vec<f32>> = clusters
            .iter()
            .map(|c| c.centroid().to_vec())
            .collect();

        let assigner = ClosureAssigner::new(
            self.config.closure_epsilon,
            self.config.max_replicas,
        );

        let mut cluster_assignments: Vec<Vec<usize>> = data
            .par_iter()
            .map(|v| assigner.assign(v, &centroids))
            .collect();

        // Step 3: Create posting lists with RaBitQ
        println!("Step 3: Quantizing posting lists...");
        let mut posting_lists: Vec<PostingList> = clusters
            .par_iter()
            .enumerate()
            .map(|(i, cluster)| {
                let mut plist = PostingList::new(i as u32, cluster.centroid().to_vec());

                // Collect vectors assigned to this cluster
                let assigned_data: Vec<Vec<f32>> = cluster_assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, assignments)| assignments.contains(&i))
                    .map(|(vec_id, _)| data[vec_id].clone())
                    .collect();

                let ids: Vec<u64> = (0..assigned_data.len() as u64).collect();

                plist
                    .quantize_vectors(
                        &assigned_data,
                        &ids,
                        self.config.rabitq_bits,
                        self.config.metric,
                        self.config.faster_config,
                    )
                    .unwrap();

                plist
            })
            .collect();

        println!("Created {} posting lists", posting_lists.len());

        // Step 4: Build HNSW on centroids
        println!("Step 4: Building HNSW index...");
        let centroid_reps: Vec<Vec<f32>> = posting_lists
            .iter()
            .map(|p| p.centroid.clone())
            .collect();
        let centroid_ids: Vec<u32> = (0..posting_lists.len() as u32).collect();

        let hnsw = CentroidIndex::build(
            centroid_reps,
            centroid_ids,
            self.config.hnsw_m,
            self.config.hnsw_ef_construction,
        );

        // Step 5: Create metadata directory
        println!("Step 5: Creating directory...");
        let directory = PostingListDirectory::new();

        Ok(MstgIndex {
            config: self.config.clone(),
            hnsw,
            posting_lists,
            directory,
        })
    }
}
```

### Deliverable
- End-to-end index building
- Progress reporting
- Memory usage tracking

---

## Milestone 7: Search Implementation (Week 9-10)

### Goal
Implement efficient search with dynamic pruning.

### Tasks

#### 7.1 Search (`src/mstg/search.rs`)
```rust
use super::*;
use rayon::prelude::*;

impl MstgIndex {
    pub fn search(
        &self,
        query: &[f32],
        params: &SearchParams,
    ) -> Vec<SearchResult> {
        // Step 1: Find candidate centroids with HNSW
        let ef_search = params.ef_search;
        let centroid_candidates = self.hnsw.search(
            query,
            ef_search,
            ef_search,
        );

        // Step 2: Dynamic pruning
        let selected_centroids = self.dynamic_prune(
            &centroid_candidates,
            params.pruning_epsilon,
        );

        // Step 3: Load posting lists (parallel)
        let posting_lists: Vec<&PostingList> = selected_centroids
            .iter()
            .map(|&cid| &self.posting_lists[cid as usize])
            .collect();

        // Step 4: Compute distances (parallel)
        let mut all_candidates: Vec<(u64, f32)> = posting_lists
            .par_iter()
            .flat_map(|plist| {
                plist
                    .vectors
                    .iter()
                    .map(|qvec| {
                        let dist = plist.rabitq_config.estimate_distance(
                            &crate::math::subtract(query, &plist.centroid),
                            &qvec.quantized,
                        );
                        (qvec.vector_id, dist)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Step 5: Top-k selection
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_candidates.truncate(params.top_k);

        all_candidates
            .into_iter()
            .map(|(id, dist)| SearchResult {
                vector_id: id as usize,
                distance: dist,
            })
            .collect()
    }

    fn dynamic_prune(
        &self,
        candidates: &[(u32, f32)],
        epsilon: f32,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let closest_dist = candidates[0].1;
        let threshold = closest_dist * (1.0 + epsilon);

        candidates
            .iter()
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(id, _)| *id)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub vector_id: usize,
    pub distance: f32,
}
```

### Deliverable
- Working search with dynamic pruning
- Benchmarks: recall vs latency curves
- Comparison with IVF-RaBitQ

---

## Milestone 8: Disk I/O (Week 11)

### Goal
Implement efficient disk serialization and loading.

### Tasks

#### 8.1 Serialization (`src/mstg/io.rs`)
```rust
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use bincode;

const MAGIC: &[u8; 4] = b"MSTG";
const VERSION: u32 = 1;

impl MstgIndex {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Header
        writer.write_all(MAGIC).map_err(|e| e.to_string())?;
        writer.write_all(&VERSION.to_le_bytes()).map_err(|e| e.to_string())?;

        // Config
        bincode::serialize_into(&mut writer, &self.config)
            .map_err(|e| e.to_string())?;

        // HNSW (serialize to bytes first)
        let hnsw_bytes = bincode::serialize(&self.hnsw)
            .map_err(|e| e.to_string())?;
        writer.write_all(&(hnsw_bytes.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer.write_all(&hnsw_bytes).map_err(|e| e.to_string())?;

        // Posting lists
        let num_lists = self.posting_lists.len() as u32;
        writer.write_all(&num_lists.to_le_bytes()).map_err(|e| e.to_string())?;

        for plist in &self.posting_lists {
            bincode::serialize_into(&mut writer, plist)
                .map_err(|e| e.to_string())?;
        }

        // Directory
        bincode::serialize_into(&mut writer, &self.directory)
            .map_err(|e| e.to_string())?;

        writer.flush().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Verify header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != MAGIC {
            return Err("Invalid file format".to_string());
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(|e| e.to_string())?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(format!("Unsupported version: {}", version));
        }

        // Load components
        let config: MstgConfig = bincode::deserialize_from(&mut reader)
            .map_err(|e| e.to_string())?;

        // Load HNSW
        let mut hnsw_len_bytes = [0u8; 8];
        reader.read_exact(&mut hnsw_len_bytes).map_err(|e| e.to_string())?;
        let hnsw_len = u64::from_le_bytes(hnsw_len_bytes) as usize;
        let mut hnsw_bytes = vec![0u8; hnsw_len];
        reader.read_exact(&mut hnsw_bytes).map_err(|e| e.to_string())?;
        let hnsw = bincode::deserialize(&hnsw_bytes).map_err(|e| e.to_string())?;

        // Load posting lists
        let mut num_lists_bytes = [0u8; 4];
        reader.read_exact(&mut num_lists_bytes).map_err(|e| e.to_string())?;
        let num_lists = u32::from_le_bytes(num_lists_bytes);

        let mut posting_lists = Vec::with_capacity(num_lists as usize);
        for _ in 0..num_lists {
            let plist = bincode::deserialize_from(&mut reader)
                .map_err(|e| e.to_string())?;
            posting_lists.push(plist);
        }

        // Load directory
        let directory = bincode::deserialize_from(&mut reader)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            config,
            hnsw,
            posting_lists,
            directory,
        })
    }
}
```

### Deliverable
- Save/load functionality
- CRC32 checksums
- Benchmark I/O performance

---

## Milestone 9: CLI and Testing (Week 12-13)

### Goal
Create user-friendly CLI and comprehensive tests.

### Tasks

#### 9.1 CLI (`src/bin/mstg.rs`)
```rust
use clap::{Parser, Subcommand};
use rabitq_rs::mstg::*;

#[derive(Parser)]
#[command(name = "mstg")]
#[command(about = "MSTG: Multi-Scale Tree Graph ANNS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build an MSTG index
    Build {
        /// Input vectors (.fvecs)
        #[arg(long)]
        input: String,

        /// Output index file
        #[arg(long)]
        output: String,

        /// Max posting list size
        #[arg(long, default_value = "5000")]
        max_posting_size: usize,

        /// RaBitQ bits
        #[arg(long, default_value = "7")]
        bits: usize,
    },

    /// Search an MSTG index
    Search {
        /// Index file
        #[arg(long)]
        index: String,

        /// Query vectors (.fvecs)
        #[arg(long)]
        queries: String,

        /// Ground truth (.ivecs)
        #[arg(long)]
        gt: Option<String>,

        /// Top-k
        #[arg(long, default_value = "100")]
        top_k: usize,

        /// ef_search
        #[arg(long, default_value = "150")]
        ef_search: usize,
    },

    /// Benchmark mode
    Benchmark {
        /// Index file
        #[arg(long)]
        index: String,

        /// Query vectors
        #[arg(long)]
        queries: String,

        /// Ground truth
        #[arg(long)]
        gt: String,
    },
}

fn main() {
    // Implementation similar to ivf_rabitq.rs
}
```

### Deliverable
- Fully functional CLI
- Unit tests for all modules
- Integration tests
- Documentation

---

## Testing Strategy

### Unit Tests
- Each module in `src/mstg/*.rs` has `#[cfg(test)]` tests
- Test data structures, algorithms individually

### Integration Tests
- `tests/mstg_integration.rs`
- End-to-end: build → save → load → search
- Small datasets (SIFT1K)

### Benchmarks
- Compare against IVF-RaBitQ
- Metrics: recall@K, latency, memory, disk I/O
- Datasets: SIFT1M, GIST1M

---

## Performance Targets

### GIST1M (1M vectors, 960 dims)
- **Memory**: <150 MB (vs ~4 GB full precision)
- **Recall@10 = 90%**: <5ms latency
- **Index build**: <10 minutes

### Improvement over IVF-RaBitQ
- **2x better QPS** at same recall (due to HNSW + dynamic pruning)
- **+5% recall** at same nprobe (due to closure assignment)

---

## Risk Mitigation

### Technical Risks

1. **HNSW library compatibility**
   - Mitigation: Evaluate multiple crates early (hnsw, instant-distance)
   - Fallback: Implement simple HNSW wrapper

2. **Memory overhead**
   - Mitigation: Profile memory usage at each stage
   - Fallback: Reduce closure replicas, HNSW parameters

3. **Disk I/O performance**
   - Mitigation: Use memory-mapped files, parallel loading
   - Fallback: Simple buffered I/O

### Schedule Risks

1. **Complexity underestimation**
   - Mitigation: Break into smaller milestones
   - Buffer: 2 weeks for unexpected issues

2. **Integration challenges**
   - Mitigation: Start integration early (Milestone 6)
   - Incremental testing at each phase

---

## Success Metrics

The implementation is successful if:

1. ✅ All unit tests pass
2. ✅ Integration tests pass on SIFT1M
3. ✅ Achieves 90% recall@10 on GIST1M in <5ms
4. ✅ 2x better QPS than IVF-RaBitQ at 90% recall
5. ✅ Memory usage <150 MB for GIST1M
6. ✅ Clean, documented code passing clippy

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Set up development branch**: `feature/mstg`
3. **Create GitHub project** with milestones
4. **Start Milestone 1**: Data structures
5. **Weekly progress reviews**

---

**Plan Version**: 1.0
**Estimated Timeline**: 13 weeks
**Author**: Claude Code
**Date**: 2025-10-20
