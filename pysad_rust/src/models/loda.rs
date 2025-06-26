use crate::models::base_model::BaseModel;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::exceptions::PyTypeError;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Pure‐Rust LODA implementation
pub struct LodaImpl {
    num_bins: usize,
    num_random_cuts: usize,
    rng: StdRng,
    to_init: bool,
    num_features: usize,
    weights: Vec<f64>,
    projections: Vec<Vec<f64>>,
    histograms: Vec<Vec<f64>>,
    limits: Vec<Vec<f64>>,
}

impl LodaImpl {
    pub fn new(num_bins: usize, num_random_cuts: usize, random_state: Option<u64>) -> Self {
        let seed = random_state.unwrap_or_else(|| rand::random::<u64>());
        
        LodaImpl {
            num_bins,
            num_random_cuts,
            rng: StdRng::seed_from_u64(seed),
            to_init: true,
            num_features: 0,
            weights: Vec::new(),
            projections: Vec::new(),
            histograms: Vec::new(),
            limits: Vec::new(),
        }
    }
}

impl BaseModel for LodaImpl {
    fn fit_partial(&mut self, x: &[f64]) {
        if self.to_init {
            self.num_features = x.len();
            
            // Initialize weights
            self.weights = vec![1.0 / self.num_random_cuts as f64; self.num_random_cuts];
            
            // Initialize projections
            let n_nonzero_components = (self.num_features as f64).sqrt();
            let n_zero_components = self.num_features - n_nonzero_components as usize;
            
            self.projections = Vec::with_capacity(self.num_random_cuts);
            for _ in 0..self.num_random_cuts {
                // Create random normal vector
                let mut projection = Vec::with_capacity(self.num_features);
                for _ in 0..self.num_features {
                    projection.push(self.rng.gen_range(-1.0..1.0)); // Simple normal approximation
                }
                
                // Zero out some components
                let mut indices: Vec<usize> = (0..self.num_features).collect();
                indices.shuffle(&mut self.rng);
                for &idx in indices.iter().take(n_zero_components) {
                    projection[idx] = 0.0;
                }
                
                self.projections.push(projection);
            }
            
            // Initialize histograms and limits
            self.histograms = vec![vec![0.0; self.num_bins]; self.num_random_cuts];
            self.limits = vec![vec![0.0; self.num_bins + 1]; self.num_random_cuts];
            
            self.to_init = false;
        }
        
        // Project data and update histograms
        for i in 0..self.num_random_cuts {
            // Project data
            let mut projected_data = 0.0;
            for j in 0..self.num_features {
                projected_data += x[j] * self.projections[i][j];
            }
            
            // Find the right bin or create new bins
            if i == 0 && self.limits[i][0] == 0.0 && self.limits[i][self.num_bins] == 0.0 {
                // First time, initialize bin limits
                let min = projected_data - 0.1;
                let max = projected_data + 0.1;
                let step = (max - min) / self.num_bins as f64;
                
                for b in 0..=self.num_bins {
                    self.limits[i][b] = min + step * b as f64;
                }
            }
            
            // Find bin index
            let mut bin_idx = 0;
            while bin_idx < self.num_bins && projected_data > self.limits[i][bin_idx + 1] {
                bin_idx += 1;
            }
            
            // Ensure bin_idx is within bounds
            bin_idx = bin_idx.min(self.num_bins - 1);
            
            // Update histogram
            self.histograms[i][bin_idx] += 1.0;
            
            // Normalize histogram
            let bin_sum: f64 = self.histograms[i].iter().sum();
            if bin_sum > 0.0 {
                for b in 0..self.num_bins {
                    self.histograms[i][b] /= bin_sum;
                }
            }
            
            // Avoid zeros for log calculation
            for b in 0..self.num_bins {
                self.histograms[i][b] = self.histograms[i][b].max(1e-12);
            }
        }
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        if self.to_init {
            return 0.0;
        }
        
        let mut score = 0.0;
        
        for i in 0..self.num_random_cuts {
            // Project data
            let mut projected_data = 0.0;
            for j in 0..self.num_features {
                projected_data += x[j] * self.projections[i][j];
            }
            
            // Find bin index
            let mut bin_idx = 0;
            while bin_idx < self.num_bins && projected_data > self.limits[i][bin_idx] {
                bin_idx += 1;
            }
            
            // Ensure bin_idx is within bounds
            bin_idx = bin_idx.min(self.num_bins - 1);
            
            // Add weighted negative log probability to score
            score += -self.weights[i] * self.histograms[i][bin_idx].ln();
        }
        
        score / self.num_random_cuts as f64
    }
    
    fn fit_score_partial(&mut self, x: &[f64]) -> f64 {
        let score = self.score_partial(x);
        self.fit_partial(x);
        score
    }
}

// Helper function to get f64 slice from PyAny (handles casting)
fn get_slice_f64<'a>(x: &'a PyAny) -> PyResult<&'a PyArray1<f64>> {
    if let Ok(arr) = x.extract::<&'a PyArray1<f64>>() {
        Ok(arr)
    } else if let Ok(arr_i16) = x.extract::<&'a PyArray1<i16>>() {
        match arr_i16.cast::<f64>(false) {
            Ok(casted) => Ok(casted),
            Err(e) => Err(PyTypeError::new_err(format!("Failed to cast array to f64: {}", e)))
        }
    } else if let Ok(arr_i32) = x.extract::<&'a PyArray1<i32>>() {
        match arr_i32.cast::<f64>(false) {
            Ok(casted) => Ok(casted),
            Err(e) => Err(PyTypeError::new_err(format!("Failed to cast array to f64: {}", e)))
        }
    } else if let Ok(arr_u8) = x.extract::<&'a PyArray1<u8>>() {
        match arr_u8.cast::<f64>(false) {
            Ok(casted) => Ok(casted),
            Err(e) => Err(PyTypeError::new_err(format!("Failed to cast array to f64: {}", e)))
        }
    } else if let Ok(arr_u16) = x.extract::<&'a PyArray1<u16>>() {
        match arr_u16.cast::<f64>(false) {
            Ok(casted) => Ok(casted),
            Err(e) => Err(PyTypeError::new_err(format!("Failed to cast array to f64: {}", e)))
        }
    } else if let Ok(arr_u32) = x.extract::<&'a PyArray1<u32>>() {
        match arr_u32.cast::<f64>(false) {
            Ok(casted) => Ok(casted),
            Err(e) => Err(PyTypeError::new_err(format!("Failed to cast array to f64: {}", e)))
        }
    } else if let Ok(arr_f32) = x.extract::<&'a PyArray1<f32>>() {
        match arr_f32.cast::<f64>(false) {
            Ok(casted) => Ok(casted),
            Err(e) => Err(PyTypeError::new_err(format!("Failed to cast array to f64: {}", e)))
        }
    } else {
        Err(PyTypeError::new_err("Unsupported array dtype for input 'x': expected f64 or compatible (e.g., i16, i32, u8, u16, u32, f32)"))
    }
}

/// Python‐exposed wrapper around LodaImpl
#[pyclass]
pub struct LODA {
    inner: LodaImpl,
}

#[pymethods]
impl LODA {
    #[new]
    #[pyo3(signature = (num_bins=10, num_random_cuts=100, random_state=None))]
    fn new(num_bins: usize, num_random_cuts: usize, random_state: Option<u64>) -> Self {
        LODA { inner: LodaImpl::new(num_bins, num_random_cuts, random_state) }
    }

    fn fit_partial(&mut self, x: &PyAny) -> PyResult<()> {
        let arr = get_slice_f64(x)?;
        let slice = unsafe { arr.as_slice()? };
        self.inner.fit_partial(slice);
        Ok(())
    }

    fn score_partial(&mut self, x: &PyAny) -> PyResult<f64> {
        let arr = get_slice_f64(x)?;
        let slice = unsafe { arr.as_slice()? };
        Ok(self.inner.score_partial(slice))
    }

    fn fit_score_partial(&mut self, x: &PyAny) -> PyResult<f64> {
        let arr = get_slice_f64(x)?;
        let slice = unsafe { arr.as_slice()? };
        Ok(self.inner.fit_score_partial(slice))
    }
}
