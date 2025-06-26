use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::{INFINITY, NEG_INFINITY};

// Node structure for the tree
#[derive(Clone)]
pub struct OnlineIsolationNode {
    data_size: usize,
    depth: usize,
    _node_index: usize,  // Prefixed with underscore to silence the warning
    min_values: Array1<f64>,
    max_values: Array1<f64>,
    projection_vector: Option<Array1<f64>>,
    split_values: Option<Array1<f64>>,
    children: Option<Vec<OnlineIsolationNode>>,
}

#[derive(Clone)]
pub struct OnlineIsolationTree {
    growth_criterion: String,
    max_leaf_samples: usize,
    subsample: f64,
    branching_factor: usize,
    split: String,
    rng: StdRng,
    data_size: usize,
    depth_limit: f64,
    root: Option<OnlineIsolationNode>,
    next_node_index: usize,
}

impl OnlineIsolationTree {
    pub fn new(
        growth_criterion: &str,
        max_leaf_samples: usize,
        subsample: f64,
        branching_factor: usize,
        split: String,
        seed: u64,
    ) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let data_size = 0;
        let depth_limit = Self::get_random_path_length(
            branching_factor,
            max_leaf_samples,
            data_size as f64,
        );

        OnlineIsolationTree {
            growth_criterion: growth_criterion.to_string(),
            max_leaf_samples,
            subsample,
            branching_factor,
            split,
            rng,
            data_size,
            depth_limit,
            root: None,
            next_node_index: 0,
        }
    }
    
    pub fn get_random_path_length(branching: usize, max_leaf: usize, n: f64) -> f64 {
        if n < max_leaf as f64 { 
            0.0 
        } else { 
            (n / max_leaf as f64).ln() / (2.0 * branching as f64).ln() 
        }
    }
    
    fn get_multiplier(&self, depth: usize) -> usize {
        if self.growth_criterion == "fixed" {
            1
        } else if self.growth_criterion == "adaptive" {
            2usize.pow(depth as u32)
        } else {
            1 // Default
        }
    }
    
    pub fn learn(&mut self, data: &Array2<f64>) {
        // Subsample data just like in Python
        let mut indices = Vec::new();
        for i in 0..data.nrows() {
            if self.rng.gen::<f64>() < self.subsample {
                indices.push(i);
            }
        }
        
        if !indices.is_empty() {
            let subsampled_data = data.select(Axis(0), &indices);
            
            // Update data size
            self.data_size += subsampled_data.nrows();
            
            // Adjust depth limit
            self.depth_limit = Self::get_random_path_length(
                self.branching_factor,
                self.max_leaf_samples,
                self.data_size as f64
            );
            
            // Update tree structure
            if self.root.is_none() {
                let (idx, node) = self.recursive_build(subsampled_data.view(), 0, 0);
                self.next_node_index = idx;
                self.root = Some(node);
            } else {
                // Fix double mutable borrow: take the root, then pass as mutable reference
                let mut root = self.root.take().unwrap();
                let (idx, node) = self.recursive_learn(
                    &mut root, 
                    subsampled_data.view(), 
                    self.next_node_index
                );
                self.next_node_index = idx;
                self.root = Some(node);
            }
        }
    }
    
    fn recursive_learn(
        &mut self, 
        node: &mut OnlineIsolationNode, 
        data: ArrayView2<f64>, 
        node_index: usize
    ) -> (usize, OnlineIsolationNode) {
        // Clone node for modification
        let mut node = node.clone();
        
        // Update data size
        node.data_size += data.nrows();
        
        // Update min_max values
        for i in 0..data.ncols() {
            let mut col_min = node.min_values[i];
            let mut col_max = node.max_values[i];
            
            for j in 0..data.nrows() {
                let val = data[[j, i]];
                if val < col_min {
                    col_min = val;
                }
                if val > col_max {
                    col_max = val;
                }
            }
            
            node.min_values[i] = col_min;
            node.max_values[i] = col_max;
        }
        
        // If leaf, check if we need to split
        if node.children.is_none() {
            let multiplier = self.get_multiplier(node.depth);
            let threshold = self.max_leaf_samples * multiplier;
            
            if node.data_size >= threshold && (node.depth as f64) < self.depth_limit {
                // Generate random samples in bounding box
                let mut sampled_data = Array2::zeros((node.data_size, data.ncols()));
                
                for i in 0..node.data_size {
                    for j in 0..data.ncols() {
                        let min_val = node.min_values[j];
                        let max_val = node.max_values[j];
                        
                        // Check if min and max are equal (or very close)
                        if (max_val - min_val).abs() < 1e-10 {
                            sampled_data[[i, j]] = min_val; // Just use the min value
                        } else {
                            sampled_data[[i, j]] = self.rng.gen_range(min_val..max_val);
                        }
                    }
                }
                
                return self.recursive_build(sampled_data.view(), node.depth, node_index);
            } else {
                return (node_index, node);
            }
        } else {
            // If not leaf, distribute data to children
            let projection = node.projection_vector.as_ref().unwrap();
            let split_vals = node.split_values.as_ref().unwrap();
            
            // Project data
            let mut projected_data = vec![0.0; data.nrows()];
            for i in 0..data.nrows() {
                for j in 0..data.ncols() {
                    projected_data[i] += data[[i, j]] * projection[j];
                }
            }
            
            // Sort data into partitions
            let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); self.branching_factor];
            
            for i in 0..data.nrows() {
                let p = projected_data[i];
                let mut branch = 0;
                
                for (j, &split_val) in split_vals.iter().enumerate() {
                    if p > split_val {
                        branch = j + 1;
                    }
                }
                
                partitions[branch].push(i);
            }
            
            // Recursively update children
            let mut children = node.children.take().unwrap();
            let mut next_idx = node_index;
            
            for (i, indices) in partitions.iter().enumerate() {
                if !indices.is_empty() {
                    let child_data = data.select(Axis(0), indices);
                    let (idx, updated_child) = self.recursive_learn(
                        &mut children[i], 
                        child_data.view(), 
                        next_idx
                    );
                    next_idx = idx;
                    children[i] = updated_child;
                }
            }
            
            node.children = Some(children);
            return (next_idx, node);
        }
    }
    
    fn recursive_build(
        &mut self, 
        data: ArrayView2<f64>, 
        depth: usize, 
        node_index: usize
    ) -> (usize, OnlineIsolationNode) {
        let multiplier = self.get_multiplier(depth);
        let threshold = self.max_leaf_samples * multiplier;
        
        // Create min_max arrays
        let mut min_values = Array1::from_elem(data.ncols(), INFINITY);
        let mut max_values = Array1::from_elem(data.ncols(), NEG_INFINITY);
        
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let val = data[[i, j]];
                if val < min_values[j] {
                    min_values[j] = val;
                }
                if val > max_values[j] {
                    max_values[j] = val;
                }
            }
        }
        
        // Check if leaf node
        if data.nrows() < threshold || (depth as f64) >= self.depth_limit {
            return (node_index + 1, OnlineIsolationNode {
                data_size: data.nrows(),
                depth,
                _node_index: node_index,  // Fixed: use _node_index instead of node_index
                min_values,
                max_values,
                projection_vector: None,
                split_values: None,
                children: None,
            });
        } else {
            // Create split node
            let mut projection_vector = Array1::zeros(data.ncols());
            
            if self.split == "axisparallel" {
                // Choose a random feature
                let feature = self.rng.gen_range(0..data.ncols());
                projection_vector[feature] = 1.0;
            } else {
                // Fallback to simple split
                let feature = self.rng.gen_range(0..data.ncols());
                projection_vector[feature] = 1.0;
            }
            
            // Project data
            let mut projected_data = vec![0.0; data.nrows()];
            for i in 0..data.nrows() {
                for j in 0..data.ncols() {
                    projected_data[i] += data[[i, j]] * projection_vector[j];
                }
            }
            
            // Find min/max of projections
            let mut min_proj = INFINITY;
            let mut max_proj = NEG_INFINITY;
            for &p in &projected_data {
                if p < min_proj { min_proj = p; }
                if p > max_proj { max_proj = p; }
            }
            
            // Generate split values
            let mut split_values = Array1::zeros(self.branching_factor - 1);
            
            // Check if min_proj and max_proj are equal (or very close)
            if (max_proj - min_proj).abs() < 1e-10 {
                // If they are equal, set all split values to the same value
                // Adding a tiny offset to ensure we don't create empty branches
                for i in 0..self.branching_factor - 1 {
                    // Distribute split points evenly around the single value point
                    split_values[i] = min_proj - 1e-10 + (i as f64 * 2e-10 / (self.branching_factor - 1) as f64);
                }
            } else {
                for i in 0..self.branching_factor - 1 {
                    split_values[i] = self.rng.gen_range(min_proj..max_proj);
                }
            }
            
            // Sort split values
            let mut split_vec: Vec<f64> = split_values.iter().map(|&v| v).collect();
            split_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 0..split_values.len() {
                split_values[i] = split_vec[i];
            }
            
            // Partition data
            let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); self.branching_factor];
            
            for i in 0..data.nrows() {
                let p = projected_data[i];
                let mut branch = 0;
                
                for (j, &split_val) in split_values.iter().enumerate() {
                    if p > split_val {
                        branch = j + 1;
                    }
                }
                
                partitions[branch].push(i);
            }
            
            // Recursively build children
            let mut children = Vec::with_capacity(self.branching_factor);
            let mut next_idx = node_index + 1;
            
            for indices in &partitions {
                if indices.is_empty() {
                    // Empty child - create leaf
                    children.push(OnlineIsolationNode {
                        data_size: 0,
                        depth: depth + 1,
                        _node_index: next_idx,  // Fixed: use _node_index instead of node_index
                        min_values: min_values.clone(),
                        max_values: max_values.clone(),
                        projection_vector: None,
                        split_values: None,
                        children: None,
                    });
                    next_idx += 1;
                } else {
                    // Build child with data
                    let child_data = data.select(Axis(0), indices);
                    let (idx, child) = self.recursive_build(child_data.view(), depth + 1, next_idx);
                    next_idx = idx;
                    children.push(child);
                }
            }
            
            return (next_idx, OnlineIsolationNode {
                data_size: data.nrows(),
                depth,
                _node_index: node_index,  // Fixed: use _node_index instead of node_index
                min_values,
                max_values,
                projection_vector: Some(projection_vector),
                split_values: Some(split_values),
                children: Some(children),
            });
        }
    }
    
    pub fn unlearn(&mut self, data: &Array2<f64>) {
        // Subsample data
        let mut indices = Vec::new();
        for i in 0..data.nrows() {
            if self.rng.gen::<f64>() < self.subsample {
                indices.push(i);
            }
        }
        
        if !indices.is_empty() {
            let subsampled_data = data.select(Axis(0), &indices);
            
            // Update data size
            self.data_size -= subsampled_data.nrows();
            
            // Adjust depth limit
            self.depth_limit = Self::get_random_path_length(
                self.branching_factor,
                self.max_leaf_samples,
                self.data_size as f64
            );
            
            // Update tree structure
            if let Some(mut root) = self.root.take() {
                self.root = Some(self.recursive_unlearn(&mut root, subsampled_data.view()));
            }
        }
    }
    
    fn recursive_unlearn(
        &mut self, 
        node: &mut OnlineIsolationNode, 
        data: ArrayView2<f64>
    ) -> OnlineIsolationNode {
        // Clone node for modification
        let mut node = node.clone();
        
        // Update data size
        node.data_size = node.data_size.saturating_sub(data.nrows());
        
        // If leaf, return it
        if node.children.is_none() {
            return node;
        }
        
        // If node doesn't have enough samples, unsplit it
        let multiplier = self.get_multiplier(node.depth);
        let threshold = self.max_leaf_samples * multiplier;
        
        if node.data_size < threshold {
            return self.recursive_unbuild(&mut node);
        }
        
        // Otherwise, distribute data to children
        let projection = node.projection_vector.as_ref().unwrap();
        let split_vals = node.split_values.as_ref().unwrap();
        
        // Project data
        let mut projected_data = vec![0.0; data.nrows()];
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                projected_data[i] += data[[i, j]] * projection[j];
            }
        }
        
        // Sort data into partitions
        let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); self.branching_factor];
        
        for i in 0..data.nrows() {
            let p = projected_data[i];
            let mut branch = 0;
            
            for (j, &split_val) in split_vals.iter().enumerate() {
                if p > split_val {
                    branch = j + 1;
                }
            }
            
            partitions[branch].push(i);
        }
        
        // Recursively update children
        let mut children = node.children.take().unwrap();
        
        for (i, indices) in partitions.iter().enumerate() {
            if !indices.is_empty() {
                let child_data = data.select(Axis(0), indices);
                children[i] = self.recursive_unlearn(&mut children[i], child_data.view());
            }
        }
        
        // Update min_max from children
        if !children.is_empty() {
            node.min_values.fill(INFINITY);
            node.max_values.fill(NEG_INFINITY);
            
            for child in &children {
                for j in 0..node.min_values.len() {
                    if child.min_values[j] < node.min_values[j] {
                        node.min_values[j] = child.min_values[j];
                    }
                    if child.max_values[j] > node.max_values[j] {
                        node.max_values[j] = child.max_values[j];
                    }
                }
            }
        }
        
        node.children = Some(children);
        node
    }
    
    fn recursive_unbuild(&mut self, node: &mut OnlineIsolationNode) -> OnlineIsolationNode {
        // Clone node for modification
        let mut node = node.clone();
        
        if let Some(children) = node.children.take() {
            // Recursively unbuild all children
            let mut unbuilt_children = Vec::new();
            
            for mut child in children {
                unbuilt_children.push(self.recursive_unbuild(&mut child));
            }
            
            // Update min_max from children
            node.min_values.fill(INFINITY);
            node.max_values.fill(NEG_INFINITY);
            
            for child in &unbuilt_children {
                for j in 0..node.min_values.len() {
                    if child.min_values[j] < node.min_values[j] {
                        node.min_values[j] = child.min_values[j];
                    }
                    if child.max_values[j] > node.max_values[j] {
                        node.max_values[j] = child.max_values[j];
                    }
                }
            }
            
            // Remove projection vector and split values
            node.projection_vector = None;
            node.split_values = None;
        }
        
        node
    }
    
    pub fn predict(&self, data: &Array2<f64>) -> Vec<f64> {
        if let Some(root) = &self.root {
            self.recursive_depth_search(root, data)
        } else {
            vec![0.0; data.nrows()]
        }
    }
    
    fn recursive_depth_search(&self, node: &OnlineIsolationNode, data: &Array2<f64>) -> Vec<f64> {
        let mut depths = vec![0.0; data.nrows()];
        
        // If leaf or empty data, return depth + normalization
        if node.children.is_none() || data.nrows() == 0 {
            let leaf_depth = node.depth as f64;
            let norm_factor = Self::get_random_path_length(
                self.branching_factor,
                self.max_leaf_samples,
                node.data_size as f64
            );
            
            for d in &mut depths {
                *d = leaf_depth + norm_factor;
            }
            
            return depths;
        }
        
        // Project data
        let projection = node.projection_vector.as_ref().unwrap();
        let split_vals = node.split_values.as_ref().unwrap();
        
        let mut projected_data = vec![0.0; data.nrows()];
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                projected_data[i] += data[[i, j]] * projection[j];
            }
        }
        
        // Sort data into partitions
        let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); self.branching_factor];
        
        for i in 0..data.nrows() {
            let p = projected_data[i];
            let mut branch = 0;
            
            for (j, &split_val) in split_vals.iter().enumerate() {
                if p > split_val {
                    branch = j + 1;
                }
            }
            
            partitions[branch].push(i);
        }
        
        // Recursively get depths for each partition
        let children = node.children.as_ref().unwrap();
        
        for (i, indices) in partitions.iter().enumerate() {
            if !indices.is_empty() {
                let child_data = data.select(Axis(0), indices);
                let child_depths = self.recursive_depth_search(&children[i], &child_data);
                
                // Copy depths back to result
                for (j, &idx) in indices.iter().enumerate() {
                    depths[idx] = child_depths[j];
                }
            }
        }
        
        depths
    }
}
