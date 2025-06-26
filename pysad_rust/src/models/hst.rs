use crate::models::base_model::BaseModel;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::exceptions::PyTypeError;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Node structure for Half-Space Trees
#[derive(Clone)]
struct HstNode {
    left: Option<Box<HstNode>>,
    right: Option<Box<HstNode>>,
    r_mass: usize,
    l_mass: usize,
    split_att: usize,
    split_value: f64,
    k: usize,
}

/// Pure‐Rust Half-Space Trees implementation
pub struct HalfSpaceTreesImpl {
    window_size: usize,
    max_depth: usize,
    _num_trees: usize,  // Prefixed with underscore to silence the warning
    feature_mins: Vec<f64>,
    feature_maxes: Vec<f64>,
    num_dimensions: usize,
    roots: Vec<HstNode>,
    is_first_window: bool,
    step: usize,
    rng: StdRng,
}

impl HalfSpaceTreesImpl {
    pub fn new(
        feature_mins: Vec<f64>,
        feature_maxes: Vec<f64>,
        window_size: usize,
        num_trees: usize,
        max_depth: usize,
        seed: Option<u64>,
    ) -> Self {
        let num_dimensions = feature_maxes.len();
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        let mut instance = HalfSpaceTreesImpl {
            window_size,
            max_depth,
            _num_trees: num_trees,
            feature_mins,
            feature_maxes,
            num_dimensions,
            roots: Vec::with_capacity(num_trees),
            is_first_window: true,
            step: 0,
            rng,
        };
        
        // Build the trees
        for _ in 0..num_trees {
            let root = instance.build_single_hs_tree(
                instance.feature_mins.clone(),
                instance.feature_maxes.clone(),
                0
            );
            instance.roots.push(root);
        }
        
        instance
    }
    
    fn build_single_hs_tree(&mut self, mut mins: Vec<f64>, mut maxes: Vec<f64>, current_depth: usize) -> HstNode {
        if current_depth == self.max_depth {
            return HstNode {
                left: None,
                right: None,
                r_mass: 0,
                l_mass: 0,
                split_att: 0,
                split_value: 0.0,
                k: current_depth,
            };
        }
        
        let q = self.rng.gen_range(0..self.num_dimensions);
        let p = (maxes[q] + mins[q]) / 2.0;
        
        let temp = maxes[q];
        maxes[q] = p;
        let left = self.build_single_hs_tree(mins.clone(), maxes.clone(), current_depth + 1);
        maxes[q] = temp;
        mins[q] = p;
        let right = self.build_single_hs_tree(mins, maxes, current_depth + 1);
        
        HstNode {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            r_mass: 0,
            l_mass: 0,
            split_att: q,
            split_value: p,
            k: current_depth,
        }
    }
    
    fn update_mass(x: &[f64], node: &mut HstNode, ref_window: bool, max_depth: usize) {
        if ref_window {
            node.r_mass += 1;
            node.l_mass += 1; // Does not exist in original since we want it to predict while building the first window
        } else {
            node.l_mass += 1;
        }

        if node.k < max_depth {
            let target_node = if x[node.split_att] > node.split_value {
                node.right.as_mut().unwrap()
            } else {
                node.left.as_mut().unwrap()
            };

            HalfSpaceTreesImpl::update_mass(x, target_node, ref_window, max_depth);
        }
    }

    fn update_model(node: &mut HstNode) {
        node.r_mass = node.l_mass;
        node.l_mass = 0;

        if let Some(ref mut left) = node.left {
            HalfSpaceTreesImpl::update_model(left);
        }

        if let Some(ref mut right) = node.right {
            HalfSpaceTreesImpl::update_model(right);
        }
    }
    
    fn score_tree(&self, x: &[f64], node: &HstNode) -> f64 {
        if node.k == self.max_depth {
            return 0.0;
        }
        
        let target_node = if x[node.split_att] > node.split_value {
            node.right.as_ref().unwrap()
        } else {
            node.left.as_ref().unwrap()
        };
        
        (node.r_mass as f64) * (2.0_f64.powi(node.k as i32)) + self.score_tree(x, target_node)
    }
    
    fn initialize_with_window(&mut self, xs: &[Vec<f64>]) {
        for x in xs {
            self.fit_partial(x);
        }
    }
}

impl BaseModel for HalfSpaceTreesImpl {
    fn fit_partial(&mut self, x: &[f64]) {
        self.step += 1;

        let is_first_window = self.is_first_window;
        let max_depth = self.max_depth;
        for root in &mut self.roots {
            HalfSpaceTreesImpl::update_mass(x, root, is_first_window, max_depth);
        }

        if self.step % self.window_size == 0 {
            self.is_first_window = false;
            for root in &mut self.roots {
                HalfSpaceTreesImpl::update_model(root);
            }
        }
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        let mut s = 0.0;
        
        for root in &self.roots {
            s += self.score_tree(x, root);
        }
        
        -s
    }

    fn fit_score_partial(&mut self, x: &[f64]) -> f64 {
        // Get score before updating the model
        let score = self.score_partial(x);
        
        // Update the model with the current instance
        self.fit_partial(x);
        
        score
    }
}

/// Helper function to get f64 slice from PyAny (simplifying type handling)
fn get_slice_f64<'a>(x: &'a PyAny) -> PyResult<&'a PyArray1<f64>> {
    x.extract::<&'a PyArray1<f64>>().map_err(|_| 
        PyTypeError::new_err("Expected numpy array of float64 values"))
}

/// Helper function to convert PyArray to Vec<f64>
fn array_to_vec(arr: &PyArray1<f64>) -> PyResult<Vec<f64>> {
    let slice = unsafe { arr.as_slice()? };
    Ok(slice.to_vec())
}

/// Helper function to get Vec<Vec<f64>> from PyAny (simplifying type handling)
fn get_vec_vec_f64(_py: Python<'_>, xs: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    // Helper function to convert array to Vec<Vec<f64>>
    fn array2d_to_vec<'a>(arr: &'a PyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        let vec2d = unsafe {
            arr.as_array()
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>()
        };
        Ok(vec2d)
    }
    
    xs.extract::<&PyArray2<f64>>()
        .map(|arr| array2d_to_vec(arr))
        .map_err(|_| PyTypeError::new_err("Expected numpy 2D array of float64 values"))?
}

/// Python‐exposed wrapper around HalfSpaceTreesImpl
#[pyclass]
pub struct HalfSpaceTrees {
    inner: HalfSpaceTreesImpl,
}

#[pymethods]
impl HalfSpaceTrees {
    #[new]
    fn new(
        feature_mins: &PyAny, 
        feature_maxes: &PyAny,
        window_size: usize,
        num_trees: usize,
        max_depth: usize,
        initial_window_x: Option<&PyAny>,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        // Convert feature_mins and feature_maxes to Vec<f64>
        let mins_arr = get_slice_f64(feature_mins)?;
        let maxes_arr = get_slice_f64(feature_maxes)?;
        
        let mins = array_to_vec(mins_arr)?;
        let maxes = array_to_vec(maxes_arr)?;
        
        // Create the inner implementation
        let mut inner = HalfSpaceTreesImpl::new(
            mins,
            maxes,
            window_size,
            num_trees,
            max_depth,
            random_state,
        );
        
        // Handle initial window if provided
        if let Some(window) = initial_window_x {
            let py = window.py();
            let window_data = get_vec_vec_f64(py, window)?;
            inner.initialize_with_window(&window_data);
        }
        
        Ok(HalfSpaceTrees { inner })
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
