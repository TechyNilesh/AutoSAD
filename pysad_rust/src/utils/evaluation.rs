use pyo3::prelude::*;
use std::time::{Instant, Duration};
use std::collections::VecDeque;

/// Struct to track memory usage, execution time, and classification metrics
#[pyclass]
pub struct Evaluator {
    // Performance tracking
    #[pyo3(get)]
    start_time: Option<f64>,
    #[pyo3(get)]
    total_runtime: f64,
    #[pyo3(get)]
    start_memory: Option<u64>,
    #[pyo3(get)]
    memory_usage: f64,
    
    // Classification metrics storage
    #[pyo3(get)]
    true_labels: Vec<f64>,
    #[pyo3(get)]
    scores: Vec<f64>,
    
    // Window-based metrics
    window_size: usize,
    window_scores: VecDeque<f64>,
    window_labels: VecDeque<f64>,
    
    // Incremental metrics
    #[pyo3(get)]
    auroc_scores: Vec<f64>,
    #[pyo3(get)]
    precision_scores: Vec<f64>,
    #[pyo3(get)]
    recall_scores: Vec<f64>,
    #[pyo3(get)]
    f1_scores: Vec<f64>,
    #[pyo3(get)]
    processing_times: Vec<f64>,
    #[pyo3(get)]
    memory_usages: Vec<f64>,
    
    // Count processed instances
    #[pyo3(get)]
    processed_instances: usize,
    
    // Progress reporting
    progress_interval: usize,
}

#[pymethods]
impl Evaluator {
    #[new]
    fn new(window_size: Option<usize>, progress_interval: Option<usize>) -> Self {
        Evaluator {
            start_time: None,
            total_runtime: 0.0,
            start_memory: None,
            memory_usage: 0.0,
            true_labels: Vec::new(),
            scores: Vec::new(),
            window_size: window_size.unwrap_or(250),
            window_scores: VecDeque::new(),
            window_labels: VecDeque::new(),
            auroc_scores: Vec::new(),
            precision_scores: Vec::new(),
            recall_scores: Vec::new(),
            f1_scores: Vec::new(),
            processing_times: Vec::new(),
            memory_usages: Vec::new(),
            processed_instances: 0,
            progress_interval: progress_interval.unwrap_or(1000),
        }
    }
    
    /// Start performance tracking
    #[pyo3(name = "start_tracking")]
    fn start_tracking(&mut self) -> PyResult<()> {
        // Get current time in seconds since epoch
        let now = Instant::now().elapsed().as_secs_f64();
        self.start_time = Some(now);
        
        // Use a placeholder for memory since we'll use Python's psutil for actual tracking
        self.start_memory = Some(0);
        
        Ok(())
    }
    
    /// Update with a new score and label
    #[pyo3(name = "update")]
    fn update(&mut self, py: Python, true_label: f64, score: f64) -> PyResult<bool> {
        // Store values
        self.true_labels.push(true_label);
        self.scores.push(score);
        
        // Update window
        self.window_scores.push_back(score);
        self.window_labels.push_back(true_label);
        
        // Manage window size
        if self.window_scores.len() > self.window_size {
            self.window_scores.pop_front();
            self.window_labels.pop_front();
        }
        
        // Increment processed instances counter
        self.processed_instances += 1;
        
        // Check if we should calculate metrics based on progress interval
        let calculate_metrics = self.processed_instances % self.progress_interval == 0;
        
        if calculate_metrics {
            // Calculate current metrics
            self.calculate_metrics(py)?;
            
            // Update performance metrics
            if let Some(start_time) = self.start_time {
                let current_time = Instant::now().elapsed().as_secs_f64();
                self.total_runtime = current_time - start_time;
                self.processing_times.push(self.total_runtime);
            }
            
            // Memory usage would normally be tracked using Python's psutil
            // For now, we put a placeholder
            self.memory_usages.push(0.0);
        }
        
        Ok(calculate_metrics)
    }
    
    /// Calculate current metrics based on collected data
    #[pyo3(name = "calculate_metrics")]
    fn calculate_metrics(&mut self, py: Python) -> PyResult<()> {
        if self.window_labels.len() < 2 {
            return Ok(());
        }
        
        // Convert window data to vectors for processing
        let labels: Vec<_> = self.window_labels.iter().cloned().collect();
        let scores: Vec<_> = self.window_scores.iter().cloned().collect();
        
        // Calculate AUROC - using Python's sklearn since it's complex
        let auroc = calculate_auroc(py, &labels, &scores)?;
        self.auroc_scores.push(auroc);
        
        // Calculate precision, recall, F1 at a threshold of 0.5
        // In practice, you might want to determine the threshold differently
        let threshold = 0.5;
        let (precision, recall, f1) = calculate_threshold_metrics(&labels, &scores, threshold);
        
        self.precision_scores.push(precision);
        self.recall_scores.push(recall);
        self.f1_scores.push(f1);
        
        Ok(())
    }
    
    /// Set current memory usage from Python (since Rust can't easily access this)
    #[pyo3(name = "set_memory_usage")]
    fn set_memory_usage(&mut self, memory_mb: f64) -> PyResult<()> {
        self.memory_usage = memory_mb;
        
        // Update last entry in memory_usages if it exists
        if !self.memory_usages.is_empty() {
            *self.memory_usages.last_mut().unwrap() = memory_mb;
        }
        
        Ok(())
    }
    
    /// Get final evaluation results
    #[pyo3(name = "get_results")]
    fn get_results(&self, py: Python) -> PyResult<PyObject> {
        let results = pyo3::types::PyDict::new(py);
        
        // Add performance metrics
        results.set_item("total_runtime", self.total_runtime)?;
        results.set_item("memory_usage", self.memory_usage)?;
        results.set_item("processed_instances", self.processed_instances)?;
        
        // Add last calculated metrics or defaults
        let final_auroc = self.auroc_scores.last().copied().unwrap_or(0.5);
        let final_precision = self.precision_scores.last().copied().unwrap_or(0.0);
        let final_recall = self.recall_scores.last().copied().unwrap_or(0.0);
        let final_f1 = self.f1_scores.last().copied().unwrap_or(0.0);
        
        results.set_item("auroc", final_auroc)?;
        results.set_item("precision", final_precision)?;
        results.set_item("recall", final_recall)?;
        results.set_item("f1", final_f1)?;
        
        // Window metrics
        let window_metrics = pyo3::types::PyDict::new(py);
        window_metrics.set_item("window_size", self.progress_interval)?;
        window_metrics.set_item("auroc_scores", self.auroc_scores.clone())?;
        window_metrics.set_item("precision_scores", self.precision_scores.clone())?;
        window_metrics.set_item("recall_scores", self.recall_scores.clone())?;
        window_metrics.set_item("f1_scores", self.f1_scores.clone())?;
        window_metrics.set_item("runtimes", self.processing_times.clone())?;
        window_metrics.set_item("memory_usages", self.memory_usages.clone())?;
        
        results.set_item("window_metrics", window_metrics)?;
        
        Ok(results.into())
    }
}

// Helper function to calculate precision, recall, F1 score at a specific threshold
fn calculate_threshold_metrics(true_labels: &[f64], scores: &[f64], threshold: f64) -> (f64, f64, f64) {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_count = 0.0;
    
    for (label, score) in true_labels.iter().zip(scores.iter()) {
        let predicted = if *score > threshold { 1.0 } else { 0.0 };
        
        if *label > 0.0 && predicted > 0.0 {
            tp += 1.0;
        } else if *label <= 0.0 && predicted > 0.0 {
            fp += 1.0;
        } else if *label > 0.0 && predicted <= 0.0 {
            fn_count += 1.0;
        }
    }
    
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 
        2.0 * precision * recall / (precision + recall) 
    } else { 
        0.0 
    };
    
    (precision, recall, f1)
}

// Helper function to calculate AUROC using sklearn (via Python)
fn calculate_auroc(py: Python, true_labels: &[f64], scores: &[f64]) -> PyResult<f64> {
    let sklearn_metrics = py.import("sklearn.metrics")?;
    
    // Convert vectors to numpy arrays using PyArray
    let np = py.import("numpy")?;
    let y_true = np.call_method1("array", (true_labels.to_vec(),))?;
    let y_score = np.call_method1("array", (scores.to_vec(),))?;
    
    // Handle case where all labels are the same (AUROC undefined)
    let unique_labels = np.call_method1("unique", (y_true,))?;
    let label_count = unique_labels.len()?;
    
    if label_count < 2 {
        return Ok(0.5); // Default AUROC when only one class is present
    }
    
    // Calculate AUROC
    match sklearn_metrics.call_method1("roc_auc_score", (y_true, y_score)) {
        Ok(auc) => auc.extract(),
        Err(_) => Ok(0.5), // Default if calculation fails
    }
}
