use ndarray::{Array2, ArrayView2};
// Remove unnecessary imports
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
// Removed rayon import
use pyo3::exceptions::PyTypeError;
use crate::models::oif_tree::OnlineIsolationTree;
// Removed num_cpus import
use crate::models::base_model::BaseModel;
use std::cmp::min;
use std::collections::VecDeque;

#[pyclass]
pub struct OnlineIsolationForest {
    inner: OIFImpl,
}

// Replace the existing get_slice_f64 function with this simplified version
fn get_slice_f64<'a>(x: &'a PyAny) -> PyResult<&'a PyArray1<f64>> {
    // Since Python is already ensuring float64, just extract directly
    x.extract::<&'a PyArray1<f64>>()
        .map_err(|_| PyTypeError::new_err("Expected float64 numpy array"))
}

// Replace the existing get_vec_vec_f64 function with this simplified version
fn get_vec_vec_f64(_py: Python<'_>, xs: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    // Helper function to convert array to Vec<Vec<f64>>
    fn array_to_vec<'a>(arr: &'a PyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        let vec2d = unsafe {
            arr.as_array()
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>()
        };
        Ok(vec2d)
    }
    
    // Since Python is already ensuring float64, just extract directly
    let arr = xs.extract::<&PyArray2<f64>>()
        .map_err(|_| PyTypeError::new_err("Expected float64 numpy array"))?;
    array_to_vec(arr)
}

#[pymethods]
impl OnlineIsolationForest {
    #[new]
    fn new(
        num_trees: usize,
        max_leaf_samples: usize,
        growth_criterion: String,
        subsample: f64,
        window_size: usize,
        branching_factor: usize,
        split: String,
        // Removed n_jobs parameter
        random_state: Option<u64>,
    ) -> Self {
        let seed = random_state.unwrap_or(42);
        OnlineIsolationForest {
            inner: OIFImpl::new(
                num_trees,
                max_leaf_samples,
                growth_criterion,
                subsample,
                window_size,
                branching_factor,
                split,
                // Removed n_jobs parameter
                seed,
            ),
        }
    }

    // Modify methods to accept PyAny and handle casting
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

    fn fit(&mut self, py: Python<'_>, xs: &PyAny) -> PyResult<()> {
        let x_vec = get_vec_vec_f64(py, xs)?;
        self.inner.fit(&x_vec);
        Ok(())
    }

    fn score(&mut self, py: Python<'_>, xs: &PyAny) -> PyResult<Vec<f64>> {
        let x_vec = get_vec_vec_f64(py, xs)?;
        Ok(self.inner.score(&x_vec))
    }

    fn fit_score(&mut self, py: Python<'_>, xs: &PyAny) -> PyResult<Vec<f64>> {
        let x_vec = get_vec_vec_f64(py, xs)?;
        Ok(self.inner.fit_score(&x_vec))
    }
}

struct OIFImpl {
    num_trees: usize,
    max_leaf_samples: usize,
    growth_criterion: String,
    subsample: f64,
    window_size: usize,
    branching_factor: usize,
    split: String,
    // Removed n_jobs field
    rng: StdRng,
    to_init: bool,
    num_features: usize,
    trees: Vec<OnlineIsolationTree>,
    data_window: VecDeque<Vec<f64>>,
    data_size: usize,
    normalization_factor: f64,
}

impl OIFImpl {
    fn new(
        num_trees: usize,
        max_leaf_samples: usize,
        growth_criterion: String,
        subsample: f64,
        window_size: usize,
        branching_factor: usize,
        split: String,
        // Removed n_jobs parameter
        seed: u64,
    ) -> Self {
        OIFImpl {
            num_trees,
            max_leaf_samples,
            growth_criterion,
            subsample,
            window_size,
            branching_factor,
            split,
            // Removed n_jobs field
            rng: StdRng::seed_from_u64(seed),
            to_init: true,
            num_features: 0,
            trees: Vec::new(),
            data_window: VecDeque::new(),
            data_size: 0,
            normalization_factor: 0.0,
        }
    }

    fn fit_partial(&mut self, x: &[f64]) {
        if self.to_init {
            self._initialize(x.len());
        }
        let batch = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
        self._learn_batch(&batch);
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        if self.to_init {
            return 0.5;
        }
        let batch = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
        self._score_batch(&batch)[0]
    }

    fn fit_score_partial(&mut self, x: &[f64]) -> f64 {
        self.fit_partial(x);
        self.score_partial(x)
    }

    fn _initialize(&mut self, num_features: usize) {
        self.num_features = num_features;
        self.trees = (0..self.num_trees)
            .map(|_| {
                // Generate a new random seed for each tree
                let random_seed = self.rng.gen::<u64>();
                OnlineIsolationTree::new(
                    &self.growth_criterion,
                    self.max_leaf_samples,
                    self.subsample,
                    self.branching_factor,
                    self.split.clone(),
                    random_seed,
                )
            })
            .collect();
        self.to_init = false;
    }

    fn _learn_batch(&mut self, data: &Array2<f64>) {
        self.data_size += data.nrows();
        self.normalization_factor = OnlineIsolationTree::get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            (self.data_size as f64) * self.subsample,
        );
        
        // Always use sequential processing
        self.trees.iter_mut().for_each(|tree| {
            tree.learn(data);
        });

        if self.window_size > 0 {
            for i in 0..data.nrows() {
                let row = data.row(i);
                self.data_window.push_back(row.to_vec());
            }

            if self.data_size > self.window_size {
                let excess = self.data_size - self.window_size;
                let old_data: Vec<Vec<f64>> = (0..excess)
                    .map(|_| self.data_window.pop_front().unwrap())
                    .collect();

                if !old_data.is_empty() {
                    let old_data_shape = (old_data.len(), old_data[0].len());
                    let old_data_flat: Vec<f64> = old_data.into_iter().flatten().collect();
                    if let Ok(old_array) = Array2::from_shape_vec(old_data_shape, old_data_flat) {
                        self.data_size -= excess;
                        self.normalization_factor = OnlineIsolationTree::get_random_path_length(
                            self.branching_factor,
                            self.max_leaf_samples,
                            (self.data_size as f64) * self.subsample,
                        );
                        
                        // Always use sequential processing
                        self.trees.iter_mut().for_each(|tree| {
                            tree.unlearn(&old_array);
                        });
                    }
                }
            }
        }
    }

    fn _score_batch(&mut self, data: &Array2<f64>) -> Vec<f64> {
        let n = data.nrows();
        
        // Always use sequential processing
        let depths: Vec<Vec<f64>> = self.trees.iter().map(|t| t.predict(data)).collect();

        let mut mean_depths = vec![0.0; n];
        for d in &depths {
            for i in 0..n {
                mean_depths[i] += d[i];
            }
        }
        for v in &mut mean_depths {
            *v /= self.num_trees as f64;
        }
        mean_depths
            .iter()
            .map(|&md| 2f64.powf(-md / (self.normalization_factor + std::f64::EPSILON)))
            .collect()
    }
}

// Implement BaseModel for OIFImpl
impl BaseModel for OIFImpl {
    fn fit_partial(&mut self, x: &[f64]) {
        if self.to_init {
            self._initialize(x.len());
        }
        let batch = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
        self._learn_batch(&batch);
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        if self.to_init {
            return 0.5;
        }
        let batch = Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
        self._score_batch(&batch)[0]
    }
}
