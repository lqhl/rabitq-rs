//! Closure assignment with RNG rule for boundary vectors
//!
//! Assigns vectors to multiple nearby clusters to improve recall

use crate::math::l2_distance_sqr;

/// Closure assignment with RNG (Relative Neighborhood Graph) rule
pub struct ClosureAssigner {
    pub epsilon: f32,
    pub max_replicas: usize,
}

impl ClosureAssigner {
    pub fn new(epsilon: f32, max_replicas: usize) -> Self {
        Self {
            epsilon,
            max_replicas,
        }
    }

    /// Assign a vector to one or more clusters
    ///
    /// Returns cluster indices that the vector should be assigned to
    pub fn assign(&self, vector: &[f32], centroids: &[Vec<f32>]) -> Vec<usize> {
        if centroids.is_empty() {
            return Vec::new();
        }

        // Compute distances to all centroids
        let mut distances: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance_sqr(vector, c)))
            .collect();

        // Sort by distance (ascending)
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

        // Apply RNG rule to reduce redundancy
        candidates = self.apply_rng_rule(vector, centroids, candidates, &distances);

        candidates
    }

    /// Apply Relative Neighborhood Graph rule to filter candidates
    ///
    /// Skip cluster j if there exists cluster i where:
    /// - dist(i, vector) < dist(i, j)
    ///
    /// This ensures geometric diversity of assigned clusters
    fn apply_rng_rule(
        &self,
        _vector: &[f32],
        centroids: &[Vec<f32>],
        candidates: Vec<usize>,
        distances: &[(usize, f32)],
    ) -> Vec<usize> {
        let mut filtered = Vec::new();

        for &candidate_idx in &candidates {
            let mut should_include = true;

            // Check against previously selected centroids
            for &selected_idx in &filtered {
                let dist_candidate_to_v = distances
                    .iter()
                    .find(|(idx, _)| *idx == candidate_idx)
                    .map(|(_, d)| *d)
                    .unwrap_or(f32::INFINITY);

                // Get centroids using iter().nth() to avoid indexing issues
                let c_selected = centroids.iter().nth(selected_idx).unwrap().as_slice();
                let c_candidate = centroids.iter().nth(candidate_idx).unwrap().as_slice();
                let dist_selected_to_candidate = l2_distance_sqr(c_selected, c_candidate);

                // RNG rule: skip candidate if it's farther from v than selected is from candidate
                if dist_candidate_to_v > dist_selected_to_candidate {
                    should_include = false;
                    break;
                }
            }

            if should_include {
                filtered.push(candidate_idx);
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
    fn test_single_cluster_assignment() {
        let centroids = vec![vec![0.0, 0.0]];
        let assigner = ClosureAssigner::new(0.2, 8);

        let v = vec![0.1, 0.0];
        let assigned = assigner.assign(&v, &centroids);

        assert_eq!(assigned.len(), 1);
        assert_eq!(assigned[0], 0);
    }

    #[test]
    fn test_boundary_vector_multiple_assignment() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
        ];

        let assigner = ClosureAssigner::new(0.5, 8);

        // Vector close to centroid 0
        let v1 = vec![0.1, 0.0];
        let assigned = assigner.assign(&v1, &centroids);
        assert_eq!(assigned[0], 0); // Closest is 0

        // Boundary vector between 0 and 1
        let v2 = vec![0.5, 0.0];
        let assigned = assigner.assign(&v2, &centroids);
        println!("Boundary vector {:?} assigned to {:?}", v2, assigned);
        assert!(assigned.len() >= 1);
        assert!(assigned.contains(&0) || assigned.contains(&1));
    }

    #[test]
    fn test_rng_rule_reduces_redundancy() {
        // Create a scenario where RNG rule should filter out some candidates
        let centroids = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.5],
        ];

        let assigner = ClosureAssigner::new(2.0, 8); // Large epsilon to get many candidates

        let v = vec![0.25, 0.0];
        let assigned = assigner.assign(&v, &centroids);

        println!("Vector {:?} assigned to {:?}", v, assigned);

        // RNG rule should have filtered out some redundant centroids
        assert!(assigned.len() <= 3);
        assert!(assigned.contains(&0) || assigned.contains(&1));
    }

    #[test]
    fn test_max_replicas_limit() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![0.3, 0.0],
            vec![0.4, 0.0],
            vec![0.5, 0.0],
        ];

        let assigner = ClosureAssigner::new(10.0, 3); // Large epsilon, but max 3 replicas

        let v = vec![0.25, 0.0];
        let assigned = assigner.assign(&v, &centroids);

        assert!(assigned.len() <= 3, "Should respect max_replicas limit");
    }
}
