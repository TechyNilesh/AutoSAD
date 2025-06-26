use std::collections::{HashMap, VecDeque};
use pyo3::prelude::*;
use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::models::base_model::BaseModel;

/// Implementation of RSHash algorithm for streaming anomaly detection
/// Based on the paper "Real-Time Anomaly Detection in Data Streams"
#[pyclass]
pub struct RSHash {
    decay: f64,
    components_num: usize,
    hash_num: usize,
    window: VecDeque<Array1<f64>>,
    window_size: usize,
    cmsketches: Vec<HashMap<Vec<i32>, (usize, f64)>>,
    alpha: Vec<Vec<f64>>,
    f: Array1<f64>,
    effective_s: f64,
    index: usize,
    data_min: Array1<f64>,
    data_max: Array1<f64>,
    random_state: Option<u64>,
    v: Vec<Vec<usize>>,      // Selected dimensions for each component
    r: Vec<usize>,           // Number of dimensions to sample per component
    last_score: Option<f64>, // Store last computed score
}

#[pymethods]
impl RSHash {
    #[new]
    #[pyo3(signature = (feature_mins, feature_maxes, sampling_points=1000, decay=0.015, num_components=100, num_hash_fns=1, random_state=None))]
    fn new(feature_mins: Vec<f64>, feature_maxes: Vec<f64>, sampling_points: usize, decay: f64, 
           num_components: usize, num_hash_fns: usize, random_state: Option<u64>) -> Self {
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        // Convert to Array1
        let data_min = Array1::from_vec(feature_mins.clone());
        let data_max = Array1::from_vec(feature_maxes.clone());
        let dim = data_min.len();
        
        // Calculate the effective sample size for the decay factor
        let effective_s = f64::max(1000.0, 1.0 / (1.0 - f64::powf(2.0, -decay)));
        
        // Generate random f values for each component
        let mut f_vals = Vec::with_capacity(num_components);
        let low = 1.0 / effective_s.sqrt();
        let high = 1.0 - (1.0 / effective_s.sqrt());
        
        for _ in 0..num_components {
            f_vals.push(rng.gen_range(low..high));
        }
        
        let f = Array1::from(f_vals);
        
        // Initialize CM sketches
        let mut cmsketches = Vec::with_capacity(num_hash_fns);
        for _ in 0..num_hash_fns {
            cmsketches.push(HashMap::new());
        }
        
        // Sample dimensions for each component
        let mut r = Vec::with_capacity(num_components);
        let mut v = Vec::with_capacity(num_components);
        
        // Find which features have min != max (usable features)
        let mut usable_features = Vec::new();
        for i in 0..dim {
            if (data_max[i] - data_min[i]).abs() > 1e-10 {
                usable_features.push(i);
            }
        }
        
        // Calculate r for each component and sample dimensions
        for i in 0..num_components {
            let max_term = f64::max(2.0, 1.0 / f[i]);
            let common_term = effective_s.ln() / max_term.ln();
            let low_value = 1.0 + 0.5 * common_term;
            let high_value = common_term;
            
            // Determine number of dimensions to sample
            let r_i = if (low_value.floor() - high_value.floor()).abs() < 1e-10 {
                1
            } else {
                // Convert to usize and clamp to usable features size
                let r_val = rng.gen_range(low_value as usize..high_value as usize + 1);
                std::cmp::min(r_val, usable_features.len())
            };
            
            r.push(r_i);
            
            // Sample r_i dimensions from usable features
            let mut selected_dim = Vec::new();
            let mut remaining = usable_features.clone();
            
            for _ in 0..r_i {
                if remaining.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..remaining.len());
                selected_dim.push(remaining.remove(idx));
            }
            
            v.push(selected_dim);
        }
        
        // Initialize alpha values for each component
        let mut alpha = Vec::with_capacity(num_components);
        for i in 0..num_components {
            let mut component_alpha = Vec::with_capacity(v[i].len());
            for _ in 0..v[i].len() {
                component_alpha.push(rng.gen_range(0.0..f[i]));
            }
            alpha.push(component_alpha);
        }
        
        RSHash {
            decay,
            components_num: num_components,
            hash_num: num_hash_fns,
            window: VecDeque::new(),
            window_size: sampling_points,
            cmsketches,
            alpha,
            f,
            effective_s,
            index: 1 - sampling_points as i32 as usize, // Start with correct offset
            data_min,
            data_max,
            random_state,
            v,
            r,
            last_score: None,
        }
    }
    
    /// Fit one data point to the model
    fn fit_partial(&mut self, x: Vec<f64>) {
        BaseModel::fit_partial(self, &x);
    }
    
    /// Score one data point with the model
    fn score_partial(&mut self, x: Vec<f64>) -> f64 {
        BaseModel::score_partial(self, &x)
    }
    
    /// Fit and score one data point with the model
    fn fit_score_partial(&mut self, x: Vec<f64>) -> f64 {
        BaseModel::fit_score_partial(self, &x)
    }
    
    /// Fit a batch of instances
    fn fit(&mut self, xs: Vec<Vec<f64>>) {
        BaseModel::fit(self, &xs);
    }
    
    /// Score a batch of instances
    fn score(&mut self, xs: Vec<Vec<f64>>) -> Vec<f64> {
        BaseModel::score(self, &xs)
    }
    
    /// Fit and score a batch of instances
    fn fit_score(&mut self, xs: Vec<Vec<f64>>) -> Vec<f64> {
        BaseModel::fit_score(self, &xs)
    }
}

impl BaseModel for RSHash {
    fn fit_partial(&mut self, x: &[f64]) {
        let x_array = Array1::from_vec(x.to_vec());
        let mut score_instance = 0.0;
        
        for r in 0..self.components_num {
            // Use only the selected dimensions for this component
            let mut y = vec![-1.0; self.v[r].len()];
            
            // Calculate hash values only for the selected dimensions
            for (i, &dim_idx) in self.v[r].iter().enumerate() {
                let normalized_val = if (self.data_max[dim_idx] - self.data_min[dim_idx]).abs() > 1e-10 {
                    (x_array[dim_idx] - self.data_min[dim_idx]) / (self.data_max[dim_idx] - self.data_min[dim_idx])
                } else {
                    0.0
                };
                
                y[i] = ((normalized_val + self.alpha[r][i]) / self.f[r]).floor();
            }
            
            // Create mod entry with component index
            let mut mod_entry = Vec::with_capacity(y.len() + 1);
            mod_entry.push(r as i32);
            mod_entry.extend(y.iter().map(|&val| val as i32));
            
            let mut c = Vec::with_capacity(self.hash_num);
            
            for w in 0..self.hash_num {
                // Get current count and timestamp
                let value = match self.cmsketches[w].get(&mod_entry) {
                    Some(&(tstamp, wt)) => (tstamp, wt),
                    None => (self.index, 0.0),
                };
                
                let tstamp = value.0;
                let wt = value.1;
                
                // Apply decay based on time difference
                let new_wt = wt * f64::powf(2.0, -self.decay * ((self.index as f64) - (tstamp as f64)));
                c.push(new_wt);
                
                // Update the sketch with new timestamp and incremented weight
                self.cmsketches[w].insert(mod_entry.clone(), (self.index, new_wt + 1.0));
            }
            
            // Use minimum count from all hash functions and contribute to score
            let min_c = c.iter().fold(f64::MAX, |a, &b| a.min(b));
            let c_score = (1.0 + min_c).ln();
            score_instance += c_score;
        }
        
        // Normalize by number of components
        self.last_score = Some(score_instance / self.components_num as f64);
        
        // Cleanup sketches periodically (e.g., every 1000 points)
        if self.index % 1000 == 0 {
            let threshold = 1e-6;
            
            for w in 0..self.hash_num {
                self.cmsketches[w].retain(|_, &mut (tstamp, wt)| {
                    let decayed_wt = wt * f64::powf(2.0, -self.decay * ((self.index as f64) - (tstamp as f64)));
                    decayed_wt > threshold
                });
            }
        }
        
        // Increment the index counter
        self.index += 1;
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        let x_array = Array1::from_vec(x.to_vec());
        let mut score_instance = 0.0;
        
        for r in 0..self.components_num {
            // Similar logic to fit_partial but without updating the sketches
            let mut y = vec![-1.0; self.v[r].len()];
            
            for (i, &dim_idx) in self.v[r].iter().enumerate() {
                let normalized_val = if (self.data_max[dim_idx] - self.data_min[dim_idx]).abs() > 1e-10 {
                    (x_array[dim_idx] - self.data_min[dim_idx]) / (self.data_max[dim_idx] - self.data_min[dim_idx])
                } else {
                    0.0
                };
                
                y[i] = ((normalized_val + self.alpha[r][i]) / self.f[r]).floor();
            }
            
            let mut mod_entry = Vec::with_capacity(y.len() + 1);
            mod_entry.push(r as i32);
            mod_entry.extend(y.iter().map(|&val| val as i32));
            
            let mut c = Vec::with_capacity(self.hash_num);
            
            for w in 0..self.hash_num {
                // Get current count and timestamp but don't update
                let value = match self.cmsketches[w].get(&mod_entry) {
                    Some(&(tstamp, wt)) => {
                        // Apply decay based on time difference
                        let new_wt = wt * f64::powf(2.0, -self.decay * ((self.index as f64) - (tstamp as f64)));
                        new_wt
                    },
                    None => 0.0,
                };
                
                c.push(value);
            }
            
            let min_c = c.iter().fold(f64::MAX, |a, &b| a.min(b));
            let c_score = (1.0 + min_c).ln();
            score_instance += c_score;
        }
        
        score_instance / self.components_num as f64
    }
}