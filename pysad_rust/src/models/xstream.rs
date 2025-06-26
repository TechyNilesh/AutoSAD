use std::collections::HashMap;
use std::f64::consts::LN_2;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ndarray::{Array1, Array2, Axis};
use crate::models::base_model::BaseModel;

/// Implementation of xStream algorithm for streaming anomaly detection
/// Based on the paper "xStream: Outlier Detection in Feature-Evolving Data Streams"
#[pyclass]
pub struct XStream {
    n_components: usize,
    n_chains: usize,
    depth: usize,
    projector: StreamhashProjector,
    cur_window: Vec<Array1<f64>>,
    ref_window: Vec<Array1<f64>>,
    hs_chains: HsChains,
    random_state: Option<u64>,
}

#[pymethods]
impl XStream {
    #[new]
    #[pyo3(signature = (n_components=8, n_chains=8, depth=8, random_state=None))]
    fn new(n_components: usize, n_chains: usize, depth: usize, random_state: Option<u64>) -> Self {
        let rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        let delta = vec![0.5; n_components];
        
        XStream {
            n_components,
            n_chains,
            depth,
            projector: StreamhashProjector::new(n_components, 1.0/3.0),
            cur_window: Vec::new(),
            ref_window: Vec::new(),
            hs_chains: HsChains::new(delta, n_chains, depth, random_state),
            random_state,
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
}

// Implement BaseModel trait for XStream
impl BaseModel for XStream {
    fn fit_partial(&mut self, x: &[f64]) {
        let projected_x = self.projector.transform(x);
        self.cur_window.push(projected_x.clone());
        self.hs_chains.fit(&projected_x);
        
        // Update reference window and reset current window
        if !self.cur_window.is_empty() {
            self.ref_window = self.cur_window.clone();
            self.cur_window.clear();
            
            // Calculate peak-to-peak range and update deltamax
            if !self.ref_window.is_empty() {
                let mut min_vals = self.ref_window[0].clone();
                let mut max_vals = self.ref_window[0].clone();
                
                for window in &self.ref_window[1..] {
                    for (i, &val) in window.iter().enumerate() {
                        if val < min_vals[i] {
                            min_vals[i] = val;
                        }
                        if val > max_vals[i] {
                            max_vals[i] = val;
                        }
                    }
                }
                
                let mut deltamax = Array1::zeros(self.n_components);
                for i in 0..self.n_components {
                    deltamax[i] = (max_vals[i] - min_vals[i]) / 2.0;
                    if deltamax[i].abs() <= 0.0001 {
                        deltamax[i] = 1.0;
                    }
                }
                
                self.hs_chains.set_deltamax(&deltamax);
            }
        }
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        let projected_x = self.projector.transform(x);
        -1.0 * self.hs_chains.score_chains(&projected_x)
    }
}

/// Streamed hash projection for dimensionality reduction
struct StreamhashProjector {
    keys: Vec<usize>,
    constant: f64,
    density: f64,
    n_components: usize,
}

impl StreamhashProjector {
    fn new(n_components: usize, density: f64) -> Self {
        let keys: Vec<usize> = (0..n_components).collect();
        let constant = (1.0 / density).sqrt() / (n_components as f64).sqrt();
        
        StreamhashProjector {
            keys,
            constant,
            density,
            n_components,
        }
    }
    
    fn transform(&self, x: &[f64]) -> Array1<f64> {
        let ndim = x.len();
        let feature_names: Vec<String> = (0..ndim).map(|i| i.to_string()).collect();
        
        // Create projection matrix
        let mut r = Array2::<f64>::zeros((self.n_components, ndim));
        for (i, &k) in self.keys.iter().enumerate() {
            for (j, f) in feature_names.iter().enumerate() {
                r[[i, j]] = self.hash_string(k, f);
            }
        }
        
        // Project input vector
        let x_array = Array1::from_vec(x.to_vec());
        let y = x_array.dot(&r.t());
        
        y
    }
    
    fn hash_string(&self, k: usize, s: &str) -> f64 {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        s.hash(&mut hasher);
        let hash_value = hasher.finish() as f64 / (u64::MAX as f64);
        
        let s = self.density;
        if hash_value <= s / 2.0 {
            -1.0 * self.constant
        } else if hash_value <= s {
            self.constant
        } else {
            0.0
        }
    }
}

/// Chain structure for half-space chains
struct Chain {
    depth: usize,
    deltamax: Array1<f64>,
    rand: Array1<f64>,
    rand_shift: Array1<f64>,
    cmsketch_ref: Vec<HashMap<Vec<i32>, f64>>,
    is_first_window: bool,
    fs: Vec<usize>,
}

impl Chain {
    fn new(deltamax: &[f64], depth: usize, mut rng: StdRng) -> Self {
        let deltamax_array = Array1::from_vec(deltamax.to_vec());
        let n_features = deltamax.len();
        
        // Generate random values
        let mut rand_vals = vec![0.0; n_features];
        for val in &mut rand_vals {
            *val = rng.gen::<f64>();
        }
        let rand = Array1::from_vec(rand_vals);
        
        // Calculate random shift
        let rand_shift = &rand * &deltamax_array;
        
        // Initialize the sketch for each depth
        let mut cmsketch_ref = Vec::with_capacity(depth);
        for _ in 0..depth {
            cmsketch_ref.push(HashMap::new());
        }
        
        // Generate random feature indexes for each depth
        let mut fs = Vec::with_capacity(depth);
        for _ in 0..depth {
            fs.push(rng.gen_range(0..n_features));
        }
        
        Chain {
            depth,
            deltamax: deltamax_array,
            rand,
            rand_shift,
            cmsketch_ref,
            is_first_window: true,
            fs,
        }
    }
    
    fn bincount(&self, x: &Array1<f64>) -> Vec<f64> {
        let mut scores = vec![0.0; self.depth];
        let mut prebins = Array1::<f64>::zeros(x.len());
        let mut depthcount = vec![0; self.deltamax.len()];
        
        for d in 0..self.depth {
            let f = self.fs[d];
            depthcount[f] += 1;
            
            if depthcount[f] == 1 {
                prebins[f] = x[f] + self.rand_shift[f] / self.deltamax[f];
            } else {
                prebins[f] = 2.0 * prebins[f] - self.rand_shift[f] / self.deltamax[f];
            }
            
            let cmsketch = &self.cmsketch_ref[d];
            
            // Convert to integer keys for hashmap
            let l: Vec<i32> = prebins.iter().map(|&x| x.floor() as i32).collect();
            
            if let Some(&count) = cmsketch.get(&l) {
                scores[d] = count;
            }
        }
        
        scores
    }
    
    fn score(&self, x: &Array1<f64>) -> f64 {
        let scores = self.bincount(x);
        
        let mut min_score = f64::MAX;
        for (d, &score) in scores.iter().enumerate() {
            let depth = (d + 1) as f64;
            let adjusted_score = (1.0 + score).log2() + depth;
            if adjusted_score < min_score {
                min_score = adjusted_score;
            }
        }
        
        min_score
    }
    
    fn fit(&mut self, x: &Array1<f64>) {
        let mut prebins = Array1::<f64>::zeros(x.len());
        let mut depthcount = vec![0; self.deltamax.len()];
        
        for d in 0..self.depth {
            let f = self.fs[d];
            depthcount[f] += 1;
            
            if depthcount[f] == 1 {
                prebins[f] = (x[f] + self.rand_shift[f]) / self.deltamax[f];
            } else {
                prebins[f] = 2.0 * prebins[f] - self.rand_shift[f] / self.deltamax[f];
            }
            
            // Convert to integer keys for hashmap
            let l: Vec<i32> = prebins.iter().map(|&x| x.floor() as i32).collect();
            
            let cmsketch = &mut self.cmsketch_ref[d];
            *cmsketch.entry(l).or_insert(0.0) += 1.0;
        }
        
        if self.is_first_window {
            self.is_first_window = false;
        }
    }
    
    fn set_deltamax(&mut self, deltamax: &Array1<f64>) {
        self.deltamax = deltamax.clone();
        self.rand_shift = &self.rand * deltamax;
    }
}

/// Collection of half-space chains
struct HsChains {
    chains: Vec<Chain>,
    nchains: usize,
    depth: usize,
}

impl HsChains {
    fn new(deltamax: Vec<f64>, n_chains: usize, depth: usize, random_state: Option<u64>) -> Self {
        let mut chains = Vec::with_capacity(n_chains);
        
        for i in 0..n_chains {
            let seed = random_state.unwrap_or_else(|| rand::random::<u64>()).wrapping_add(i as u64);
            let rng = StdRng::seed_from_u64(seed);
            chains.push(Chain::new(&deltamax, depth, rng));
        }
        
        HsChains {
            chains,
            nchains: n_chains,
            depth,
        }
    }
    
    fn score_chains(&self, x: &Array1<f64>) -> f64 {
        let mut score_sum = 0.0;
        
        for chain in &self.chains {
            score_sum += chain.score(x);
        }
        
        score_sum / self.nchains as f64
    }
    
    fn fit(&mut self, x: &Array1<f64>) {
        for chain in &mut self.chains {
            chain.fit(x);
        }
    }
    
    fn set_deltamax(&mut self, deltamax: &Array1<f64>) {
        for chain in &mut self.chains {
            chain.set_deltamax(deltamax);
        }
    }
}