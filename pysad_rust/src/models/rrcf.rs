use pyo3::prelude::*;
use crate::models::base_model::BaseModel;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// A simplified Rust-native implementation of Robust Random Cut Forest
pub struct RRCFTree {
    points: Vec<Vec<f64>>,
    max_size: usize,
    rng: StdRng,
}

impl RRCFTree {
    pub fn new(max_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            points: Vec::new(),
            max_size,
            rng,
        }
    }

    pub fn insert_point(&mut self, point: Vec<f64>) {
        if self.points.len() >= self.max_size {
            self.points.remove(0); // Remove the oldest point
        }
        self.points.push(point);
    }

    pub fn score_point(&self, point: &[f64]) -> f64 {
        // Simplified scoring logic (e.g., based on distance to nearest neighbor)
        self.points
            .iter()
            .map(|p| euclidean_distance(p, point))
            .sum::<f64>()
            / self.points.len() as f64
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[pyclass]
pub struct RRCF {
    _num_trees: usize,  // Renamed with underscore to silence warning
    _shingle_size: usize,
    _tree_size: usize,  // Renamed with underscore to silence warning
    _random_state: Option<u64>,
    _model: Option<PyObject>,  // Renamed with underscore to silence warning
}

#[pymethods]
impl RRCF {
    /// new(num_trees=100, shingle_size=4, tree_size=256, random_state=None)
    #[new]
    #[pyo3(signature = (num_trees=100, shingle_size=4, tree_size=256, random_state=None))]
    fn new(num_trees: usize, shingle_size: usize, tree_size: usize, random_state: Option<u64>) -> Self {
        RRCF {
            _num_trees: num_trees,
            _shingle_size: shingle_size,
            _tree_size: tree_size,
            _random_state: random_state,
            _model: None,
        }
    }
}

#[pyclass]
pub struct RobustRandomCutForest {
    forest: Vec<RRCFTree>,
    index: usize,
    random_state: Option<u64>,
}

#[pymethods]
impl RobustRandomCutForest {
    #[new]
    #[pyo3(signature = (num_trees=4, shingle_size=1, tree_size=256, random_state=None))]
    fn new(num_trees: usize, shingle_size: usize, tree_size: usize, random_state: Option<u64>) -> Self {
        let mut forest = Vec::with_capacity(num_trees);

        for i in 0..num_trees {
            // Create a unique seed for each tree if random_state is provided
            let tree_seed = random_state.map(|s| s.wrapping_add(i as u64));
            forest.push(RRCFTree::new(tree_size, tree_seed));
        }

        Self { forest, index: 0, random_state }
    }

    pub fn fit_partial(&mut self, x: Vec<f64>) {
        for tree in &mut self.forest {
            tree.insert_point(x.clone());
        }
        self.index += 1;
    }

    pub fn score_partial(&self, x: Vec<f64>) -> f64 {
        self.forest
            .iter()
            .map(|tree| tree.score_point(&x))
            .sum::<f64>()
            / self.forest.len() as f64
    }

    pub fn fit_score_partial(&mut self, x: Vec<f64>) -> f64 {
        self.fit_partial(x.clone());
        self.score_partial(&x)  // Add & to borrow x
    }
}

// Implement BaseModel trait
impl BaseModel for RobustRandomCutForest {
    fn fit_partial(&mut self, x: &[f64]) {
        for tree in &mut self.forest {
            tree.insert_point(x.to_vec());
        }
        self.index += 1;
    }

    fn score_partial(&mut self, x: &[f64]) -> f64 {
        self.forest
            .iter()
            .map(|tree| tree.score_point(x))
            .sum::<f64>()
            / self.forest.len() as f64
    }
}