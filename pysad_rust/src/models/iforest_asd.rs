// src/models/iforest_asd.rs

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rand::prelude::*;
use std::cmp;
use crate::models::base_model::BaseModel;

// Custom tree node structure
struct Node {
    split_feature: Option<usize>,
    split_value: f64,
    size: usize,
    height: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    fn new(size: usize, height: usize) -> Self {
        Node {
            split_feature: None,
            split_value: 0.0,
            size,
            height,
            left: None,
            right: None,
        }
    }

    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

#[pyclass(unsendable)]
pub struct IForestASD {
    window_size: usize,
    cur_window_x: Vec<Vec<f64>>,
    cur_window_scores: Vec<f64>,  // NEW: Track scores for current window
    reference_window_x: Vec<Vec<f64>>,
    initial_ref_window: bool,

    n_estimators: usize,
    max_samples: usize,
    height_limit: usize,
    trees: Vec<Node>,
    rng: StdRng,
    
    n_features: usize,
    anomaly_rate_threshold: f64,  // NEW: Threshold for concept drift detection
    contamination: f64,  // NEW: Used to determine anomaly cutoff
}

impl IForestASD {
    fn build_tree(data: &[Vec<f64>], height: usize, height_limit: usize, rng: &mut StdRng) -> Node {
        let size = data.len();
        let mut node = Node::new(size, height);

        // Stop criteria
        if size <= 1 || height >= height_limit {
            return node;
        }

        let n_features = data[0].len();
        
        // Randomly select split feature and value
        let split_feature = rng.gen_range(0..n_features);
        
        // Find min and max for the selected feature
        let mut min_val = data[0][split_feature];
        let mut max_val = min_val;
        for row in data.iter() {
            let val = row[split_feature];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // If min == max, this is a leaf
        if (max_val - min_val).abs() < 1e-10 {
            return node;
        }

        // Generate split value
        let split_value = rng.gen::<f64>() * (max_val - min_val) + min_val;
        
        // Split data
        let mut left_data = Vec::new();
        let mut right_data = Vec::new();
        
        for row in data {
            if row[split_feature] < split_value {
                left_data.push(row.clone());
            } else {
                right_data.push(row.clone());
            }
        }

        // Only split if we actually divided the data
        if !left_data.is_empty() && !right_data.is_empty() {
            node.split_feature = Some(split_feature);
            node.split_value = split_value;
            node.left = Some(Box::new(Self::build_tree(&left_data, height + 1, height_limit, rng)));
            node.right = Some(Box::new(Self::build_tree(&right_data, height + 1, height_limit, rng)));
        }

        node
    }

    fn path_length(node: &Node, x: &[f64], current_height: usize) -> usize {
        if node.is_leaf() {
            return current_height
        }

        if let Some(split_feature) = node.split_feature {
            if x[split_feature] < node.split_value {
                if let Some(ref left) = node.left {
                    return Self::path_length(left, x, current_height + 1);
                }
            } else if let Some(ref right) = node.right {
                return Self::path_length(right, x, current_height + 1);
            }
        }
        
        current_height
    }

    fn average_path_length(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        let h = (n as f64).ln();
        h + 0.5772156649 // Euler's constant
    }

    fn _fit_model(&mut self) {
        self.trees.clear();
        
        for _ in 0..self.n_estimators {
            // Sample data
            let sample_size = cmp::min(self.max_samples, self.reference_window_x.len());
            let mut sampled_data = Vec::with_capacity(sample_size);
            
            for _ in 0..sample_size {
                let idx = self.rng.gen_range(0..self.reference_window_x.len());
                sampled_data.push(self.reference_window_x[idx].clone());
            }

            // Build tree
            let tree = Self::build_tree(
                &sampled_data,
                0,
                self.height_limit,
                &mut self.rng
            );
            self.trees.push(tree);
        }
    }

    // NEW: Calculate anomaly rate in current window
    fn calculate_anomaly_rate(&mut self) -> f64 {
        if self.cur_window_scores.is_empty() {
            return 0.0;
        }

        // Determine cutoff score based on contamination
        let mut scores = self.cur_window_scores.clone();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff_idx = (self.contamination * scores.len() as f64).floor() as usize;
        let cutoff_score = if cutoff_idx < scores.len() { 
            scores[cutoff_idx] 
        } else { 
            0.0 
        };

        // Count anomalies (scores >= cutoff_score)
        let anomaly_count = scores.iter()
            .filter(|&&s| s >= cutoff_score)
            .count();
        
        anomaly_count as f64 / scores.len() as f64
    }
}

#[pymethods]
impl IForestASD {
    #[new]
    #[pyo3(signature = (
        initial_window_x = None,
        window_size = 2048,
        n_estimators = 100,
        max_samples = 256,
        contamination = 0.1,
        anomaly_rate_threshold = 0.2,
        random_state = None
    ))]
    fn new(
        _py: Python,
        initial_window_x: Option<&PyArray2<f64>>,
        window_size: usize,
        n_estimators: usize,
        max_samples: usize,
        contamination: f64,
        anomaly_rate_threshold: f64,  // NEW: Added parameter
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let seed = random_state.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        let mut slf = IForestASD {
            window_size,
            cur_window_x: Vec::with_capacity(window_size),
            cur_window_scores: Vec::with_capacity(window_size),  // NEW
            reference_window_x: Vec::new(),
            initial_ref_window: false,
            n_estimators,
            max_samples,
            height_limit: (max_samples as f64).log2().ceil() as usize,
            trees: Vec::new(),
            rng,
            n_features: 0,
            contamination,  // NEW
            anomaly_rate_threshold,  // NEW
        };

        if let Some(arr) = initial_window_x {
            let array = unsafe { arr.as_array() };
            slf.n_features = array.shape()[1];
            slf.reference_window_x = array.outer_iter().map(|r| r.to_vec()).collect();
            slf._fit_model();
            slf.initial_ref_window = true;
        }

        Ok(slf)
    }

    fn fit_partial(&mut self, _py: Python, x: &PyArray1<f64>, _y: Option<usize>) -> PyResult<()> {
        let xi = unsafe { x.as_slice()? };
        
        if self.n_features == 0 {
            self.n_features = xi.len();
        } else if xi.len() != self.n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Input feature count mismatch. Expected {}, got {}", 
                self.n_features, xi.len())
            ));
        }

        // Store data
        self.cur_window_x.push(xi.to_vec());
        
        // Calculate and store score for current point (before potential model update)
        if !self.trees.is_empty() {
            let score = self.score_partial(xi);
            self.cur_window_scores.push(score);
        }

        if !self.initial_ref_window {
            if self.cur_window_x.len() >= self.window_size {
                self.reference_window_x = std::mem::take(&mut self.cur_window_x);
                self.cur_window_scores.clear();  // NEW: Reset scores
                self._fit_model();
                self.initial_ref_window = true;
            }
        } else if self.cur_window_x.len() >= self.window_size {
            // NEW: Concept drift detection
            let anomaly_rate = self.calculate_anomaly_rate();
            
            if anomaly_rate >= self.anomaly_rate_threshold {
                // Retrain with current window
                self.reference_window_x = std::mem::take(&mut self.cur_window_x);
                self.cur_window_scores.clear();
                self._fit_model();
            } else {
                // Discard current window, keep existing model
                self.cur_window_x.clear();
                self.cur_window_scores.clear();
            }
        }

        Ok(())
    }

    fn score_partial(&self, x: &PyArray1<f64>) -> PyResult<f64> {
        let xi = unsafe { x.as_slice()? };
        
        if xi.len() != self.n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Scoring input feature count mismatch. Expected {}, got {}", 
                self.n_features, xi.len())
            ));
        }

        if self.trees.is_empty() {
            return Ok(0.0);
        }

        // Calculate average path length across all trees
        let mut total_path_length = 0.0;
        for tree in &self.trees {
            total_path_length += Self::path_length(tree, xi, 0) as f64;
        }
        let avg_path_length = total_path_length / self.trees.len() as f64;
        
        // Normalize score
        let expected_path_length = Self::average_path_length(self.max_samples);
        if expected_path_length > 0.0 {
            Ok(2.0f64.powf(-avg_path_length / expected_path_length))
        } else {
            Ok(1.0)
        }
    }

    fn fit_score_partial(&mut self, py: Python, x: &PyArray1<f64>, y: Option<usize>) -> PyResult<f64> {
        let xi = unsafe { x.as_slice()? };
        let score = self.score_partial(xi);
        self.fit_partial(py, x, y)?;
        Ok(score)
    }
}

// Implement BaseModel trait
impl BaseModel for IForestASD {
    fn fit_partial(&mut self, x: &[f64]) {
        Python::with_gil(|py| {
            let arr = x.to_pyarray(py);
            let _ = IForestASD::fit_partial(self, py, arr, None);
        });
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let arr = x.to_pyarray(py);
            IForestASD::score_partial(self, arr).unwrap_or(0.0)
        })
    }
}