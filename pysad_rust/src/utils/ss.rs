use std::collections::HashMap;
use std::f64::{INFINITY, NEG_INFINITY};
use std::collections::VecDeque;
use ndarray::{Array1, ArrayView1};
use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};

/// Data statistics for the streaming data, with supporting max, min, sum, mean, sum of squares, var, std and standard scaler.
#[pyclass]
pub struct StreamStatistic {
    is_uni: bool,
    is_global: bool,
    window: VecDeque<Array1<f64>>,
    num_items: usize,
    max: HashMap<usize, f64>,
    min: HashMap<usize, f64>,
    sum: HashMap<usize, f64>,
    mean: HashMap<usize, f64>,
    sum_squares: HashMap<usize, f64>,
    var: HashMap<usize, f64>,
    std: HashMap<usize, f64>,
    store_values: bool,
    all_values: Vec<Array1<f64>>,
}

#[pymethods]
impl StreamStatistic {
    #[new]
    /// Statistics for the streaming data, with supporting max, min, sum, mean, sum of squares, var, std and standard scaler.
    ///
    /// # Arguments
    /// * `is_global` - For whole stream or a windowed stream. Defaults to true.
    /// * `window_len` - Rolling window length. Only works when is_global is false. Defaults to 10.
    /// * `store_values` - Whether to store all values for later retrieval. Defaults to false.
    pub fn new(is_global: Option<bool>, window_len: Option<usize>, store_values: Option<bool>) -> Self {
        StreamStatistic {
            is_uni: false,
            is_global: is_global.unwrap_or(true),
            window: VecDeque::with_capacity(window_len.unwrap_or(10)),
            num_items: 0,
            max: HashMap::new(),
            min: HashMap::new(),
            sum: HashMap::new(),
            mean: HashMap::new(),
            sum_squares: HashMap::new(),
            var: HashMap::new(),
            std: HashMap::new(),
            store_values: store_values.unwrap_or(false),
            all_values: Vec::new(),
        }
    }

    #[pyo3(name = "update")]
    /// Update with a new data point
    ///
    /// # Arguments
    /// * `x` - An item from StreamGenerator
    pub fn update(&mut self, x: &PyArray1<f64>) -> PyResult<()> {
        self.num_items += 1;
        
        // Convert PyArray1 to Array1
        let x_array: Array1<f64> = x.to_owned_array();
        
        let flattened: Array1<f64>;
        
        // Handle different input types and determine if uni-variate
        if x_array.len() == 1 {
            self.is_uni = true;
            flattened = x_array;
        } else {
            self.is_uni = false;
            flattened = x_array;
        }

        // Store values if enabled
        if self.store_values {
            self.all_values.push(flattened.clone());
        }

        if self.is_global {
            let mut tmp: HashMap<usize, f64> = HashMap::new();

            for (index, &item) in flattened.iter().enumerate() {
                // Update max
                let entry = self.max.entry(index).or_insert(NEG_INFINITY);
                if *entry < item {
                    *entry = item;
                }
                
                // Update min
                let entry = self.min.entry(index).or_insert(INFINITY);
                if *entry > item {
                    *entry = item;
                }
                
                // Update sum
                let sum_entry = self.sum.entry(index).or_insert(0.0);
                *sum_entry += item;
                
                // Update mean
                let old_mean = *self.mean.entry(index).or_insert(0.0);
                tmp.insert(index, item - old_mean);
                *self.mean.get_mut(&index).unwrap() = *sum_entry / self.num_items as f64;
                
                // Update sum_squares
                let sum_squares_entry = self.sum_squares.entry(index).or_insert(0.0);
                *sum_squares_entry += (item - old_mean) * (item - *self.mean.get(&index).unwrap());
                
                // Update var
                let var_entry = self.var.entry(index).or_insert(0.0);
                *var_entry = *sum_squares_entry / self.num_items as f64;
                
                // Update std
                let std_entry = self.std.entry(index).or_insert(0.0);
                *std_entry = (*var_entry).sqrt();
            }
        } else {
            // For windowed statistics, just add to window
            if self.window.len() == self.window.capacity() {
                self.window.pop_front();
            }
            self.window.push_back(flattened);
        }
        Ok(())
    }

    #[pyo3(name = "get_max")]
    /// Get max statistic.
    pub fn get_max<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let arr = if self.is_global {
            let mut indices: Vec<usize> = self.max.keys().cloned().collect();
            indices.sort_unstable();
            let result: Vec<f64> = indices.iter()
                .map(|&i| *self.max.get(&i).unwrap())
                .collect();
            if self.is_uni {
                Array1::from_vec(vec![result[0]])
            } else {
                Array1::from_vec(result)
            }
        } else {
            if self.window.is_empty() {
                Array1::from_vec(vec![])
            } else {
                let dim = self.window[0].len();
                let mut result = vec![NEG_INFINITY; dim];
                for window_item in &self.window {
                    for (i, &val) in window_item.iter().enumerate() {
                        if val > result[i] {
                            result[i] = val;
                        }
                    }
                }
                if self.is_uni {
                    Array1::from_vec(vec![result[0]])
                } else {
                    Array1::from_vec(result)
                }
            }
        };
        arr.to_pyarray(py).into_py(py)
    }

    #[pyo3(name = "get_min")]
    /// Get min statistic.
    pub fn get_min<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let arr = if self.is_global {
            let mut indices: Vec<usize> = self.min.keys().cloned().collect();
            indices.sort_unstable();
            let result: Vec<f64> = indices.iter()
                .map(|&i| *self.min.get(&i).unwrap())
                .collect();
            if self.is_uni {
                Array1::from_vec(vec![result[0]])
            } else {
                Array1::from_vec(result)
            }
        } else {
            if self.window.is_empty() {
                Array1::from_vec(vec![])
            } else {
                let dim = self.window[0].len();
                let mut result = vec![INFINITY; dim];
                for window_item in &self.window {
                    for (i, &val) in window_item.iter().enumerate() {
                        if val < result[i] {
                            result[i] = val;
                        }
                    }
                }
                if self.is_uni {
                    Array1::from_vec(vec![result[0]])
                } else {
                    Array1::from_vec(result)
                }
            }
        };
        arr.to_pyarray(py).into_py(py)
    }

    #[pyo3(name = "get_mean")]
    /// Get mean statistic.
    pub fn get_mean<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let arr = if self.is_global {
            let mut indices: Vec<usize> = self.mean.keys().cloned().collect();
            indices.sort_unstable();
            let result: Vec<f64> = indices.iter()
                .map(|&i| *self.mean.get(&i).unwrap())
                .collect();
            if self.is_uni {
                Array1::from_vec(vec![result[0]])
            } else {
                Array1::from_vec(result)
            }
        } else {
            if self.window.is_empty() {
                Array1::from_vec(vec![])
            } else {
                let dim = self.window[0].len();
                let mut result = vec![0.0; dim];
                for window_item in &self.window {
                    for (i, &val) in window_item.iter().enumerate() {
                        result[i] += val;
                    }
                }
                let window_size = self.window.len() as f64;
                for val in &mut result {
                    *val /= window_size;
                }
                if self.is_uni {
                    Array1::from_vec(vec![result[0]])
                } else {
                    Array1::from_vec(result)
                }
            }
        };
        arr.to_pyarray(py).into_py(py)
    }

    #[pyo3(name = "get_std")]
    /// Get standard deviation statistic.
    pub fn get_std<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let arr = if self.is_global {
            let mut indices: Vec<usize> = self.std.keys().cloned().collect();
            indices.sort_unstable();
            let result: Vec<f64> = indices.iter()
                .map(|&i| *self.std.get(&i).unwrap())
                .collect();
            if self.is_uni {
                Array1::from_vec(vec![result[0]])
            } else {
                Array1::from_vec(result)
            }
        } else {
            if self.window.is_empty() {
                Array1::from_vec(vec![])
            } else {
                // Calculate mean first
                let mean = {
                    let dim = self.window[0].len();
                    let mut result = vec![0.0; dim];
                    for window_item in &self.window {
                        for (i, &val) in window_item.iter().enumerate() {
                            result[i] += val;
                        }
                    }
                    let window_size = self.window.len() as f64;
                    for val in &mut result {
                        *val /= window_size;
                    }
                    result
                };
                let dim = self.window[0].len();
                let mut sum_squares = vec![0.0; dim];
                for window_item in &self.window {
                    for (i, &val) in window_item.iter().enumerate() {
                        sum_squares[i] += (val - mean[i]) * (val - mean[i]);
                    }
                }
                let window_size = self.window.len() as f64;
                for val in &mut sum_squares {
                    *val = (*val / window_size).sqrt();
                }
                if self.is_uni {
                    Array1::from_vec(vec![sum_squares[0]])
                } else {
                    Array1::from_vec(sum_squares)
                }
            }
        };
        arr.to_pyarray(py).into_py(py)
    }

    #[pyo3(name = "get_sum")]
    /// Get sum statistic.
    pub fn get_sum<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let arr = if self.is_global {
            let mut indices: Vec<usize> = self.sum.keys().cloned().collect();
            indices.sort_unstable();
            let result: Vec<f64> = indices.iter()
                .map(|&i| *self.sum.get(&i).unwrap())
                .collect();
            if self.is_uni {
                Array1::from_vec(vec![result[0]])
            } else {
                Array1::from_vec(result)
            }
        } else {
            if self.window.is_empty() {
                Array1::from_vec(vec![])
            } else {
                let dim = self.window[0].len();
                let mut result = vec![0.0; dim];
                for window_item in &self.window {
                    for (i, &val) in window_item.iter().enumerate() {
                        result[i] += val;
                    }
                }
                if self.is_uni {
                    Array1::from_vec(vec![result[0]])
                } else {
                    Array1::from_vec(result)
                }
            }
        };
        arr.to_pyarray(py).into_py(py)
    }

    #[pyo3(name = "get_var")]
    /// Get variance statistic.
    pub fn get_var<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        let arr = if self.is_global {
            let mut indices: Vec<usize> = self.var.keys().cloned().collect();
            indices.sort_unstable();
            let result: Vec<f64> = indices.iter()
                .map(|&i| *self.var.get(&i).unwrap())
                .collect();
            if self.is_uni {
                Array1::from_vec(vec![result[0]])
            } else {
                Array1::from_vec(result)
            }
        } else {
            if self.window.is_empty() {
                Array1::from_vec(vec![])
            } else {
                // Calculate mean first
                let mean = {
                    let dim = self.window[0].len();
                    let mut result = vec![0.0; dim];
                    for window_item in &self.window {
                        for (i, &val) in window_item.iter().enumerate() {
                            result[i] += val;
                        }
                    }
                    let window_size = self.window.len() as f64;
                    for val in &mut result {
                        *val /= window_size;
                    }
                    result
                };
                let dim = self.window[0].len();
                let mut sum_squares = vec![0.0; dim];
                for window_item in &self.window {
                    for (i, &val) in window_item.iter().enumerate() {
                        sum_squares[i] += (val - mean[i]) * (val - mean[i]);
                    }
                }
                let window_size = self.window.len() as f64;
                for val in &mut sum_squares {
                    *val = *val / window_size;
                }
                if self.is_uni {
                    Array1::from_vec(vec![sum_squares[0]])
                } else {
                    Array1::from_vec(sum_squares)
                }
            }
        };
        arr.to_pyarray(py).into_py(py)
    }

    #[pyo3(name = "get_count")]
    /// Get the total number of items processed
    pub fn get_count(&self) -> usize {
        self.num_items
    }

    #[pyo3(name = "get_values")]
    /// Get all stored values.
    pub fn get_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Vec<f64>>>> {
        if !self.store_values || self.all_values.is_empty() {
            return Ok(None);
        }

        // Convert each array to a Vec<f64> to maintain the original structure
        let mut result = Vec::with_capacity(self.all_values.len());
        for array in &self.all_values {
            // Convert each ndarray to a Vec<f64>
            let values: Vec<f64> = array.iter().copied().collect();
            result.push(values);
        }
        
        Ok(Some(result))
    }
}
