# AutoSAD - An Adaptive Framework for Streaming Anomaly Detection

## Overview

AutoSAD is an adaptive framework designed for streaming anomaly detection. It provides an set of multiple anomaly detection algorithms and automatically selects the best performing models for real-time data streams.

## Features

- **Adaptive Model Selection**: Automatically selects the best performing models based on streaming data
- **Multiple Algorithms**: Supports various anomaly detection algorithms including:
  - AutoSAD (Adaptive framework)
  - Robust Random Cut Forest (RRCF)
  - Online Isolation Forest (OIF)
  - LODA (Lightweight Online Detector of Anomalies)
  - IForestASD
  - Half-Space Trees (HST)
  - RSHash
  - XStream

- **Comprehensive Evaluation**: Built-in evaluation metrics and performance tracking
- **Benchmark Datasets**: Includes 51 benchmark datasets for evaluation
- **Parallel Execution**: Support for concurrent execution of multiple experiments

## Installation

### Prerequisites

Make sure you have Python 3.7+ installed. The project supports two installation modes:

#### Option 1: Standard Installation (Python only)
```bash
pip install -r requirements.txt
```

#### Option 2: High-Performance Installation (with Rust components)
For better performance, you can use the Rust-accelerated version:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python dependencies
pip install -r requirements.txt

# Build Rust components
cd pysad_rust
cargo build --release
cd ..
```

The requirements.txt includes:
- numpy==2.3.1
- psutil==7.0.0  
- pysad==0.2.0
- scikit_learn==1.7.0
- scipy==1.16.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/AutoSAD.git
cd AutoSAD
```

2. Choose your installation option:

**Standard Installation:**
```bash
pip install -r requirements.txt
```

**High-Performance Installation:**
```bash
# Install Rust components for better performance
pip install -r requirements.txt
cd pysad_rust
cargo build --release
cd ..
```

3. Verify installation:
```bash
python -c "import AutoSAD; print('AutoSAD installed successfully')"
```

## Dataset Structure

The project includes 51 benchmark datasets in the `adbenchmark/` directory. Each dataset is stored as a `.npz` file containing:
- `X`: Feature matrix
- `y`: Labels (0 for normal, 1 for anomaly)

### Main Datasets Used in Experiments

The following table describes the key datasets used in our experiments:

| **Dataset** | **# Samples** | **# Features** | **# Anomalies** | **% Anomaly** | **Category** |
|-------------|---------------|----------------|-----------------|---------------|--------------|
| ALOI        | 49,534        | 27             | 1,508           | 3.04%         | Image        |
| Backdoor    | 95,329        | 196            | 2,329           | 2.44%         | Network      |
| Campaign    | 41,188        | 62             | 4,640           | 11.27%        | Finance      |
| Celeba      | 202,599       | 39             | 4,547           | 2.24%         | Image        |
| Census      | 299,285       | 500            | 18,568          | 6.20%         | Sociology    |
| Cover       | 286,048       | 10             | 2,747           | 0.96%         | Botany       |
| Donors      | 619,326       | 10             | 36,710          | 5.93%         | Sociology    |
| Fraud       | 284,807       | 29             | 492             | 0.17%         | Finance      |
| Http        | 567,498       | 3              | 2,211           | 0.39%         | Web          |
| Shuttle     | 49,097        | 9              | 3,511           | 7.15%         | Astronautics |
| Skin        | 245,057       | 3              | 50,859          | 20.75%        | Image        |
| Smtp        | 95,156        | 3              | 30              | 0.03%         | Web          |
| Chess       | 28,056        | 6              | 27              | 0.10%         | Games        |
| Prob        | 64,759        | 6              | 4,166           | 6.43%         | Network      |
| Bank        | 41,188        | 10             | 4,640           | 11.27%        | Finance      |
| U2R         | 60,821        | 6              | 228             | 0.37%         | Network      |

### Complete Dataset List
Available datasets include all files from `1_ALOI.npz` through `51_kddcup99_u2r.npz` covering various domains including network security, medical, image processing, financial fraud detection, and more.

## Usage

### Quick Start

#### Using the Master Script (`run_scripts.py`)

The `run_scripts.py` file provides a convenient way to run experiments with different modes:

```bash
# Run all models on a specific dataset
python run_scripts.py --mode dataset --name 5_campaign

# Run a specific model on all datasets  
python run_scripts.py --mode model --name autosad

# Run all models on all datasets
python run_scripts.py --mode all

# Run with multiple random seeds (100 runs each)
python run_scripts.py --mode random
```

#### Advanced Options

```bash
# Specify custom output directory and progress interval
python run_scripts.py --mode dataset --name 10_cover --output_dir custom_results --progress_interval 500

# Available modes:
# - dataset: Run all models on specified dataset
# - model: Run specified model on all datasets  
# - all: Run all models on all datasets
# - random: Run multiple experiments with random seeds
```

### Individual Script Usage

Each algorithm has its own script in the `scripts/` directory. You can run them individually:

#### AutoSAD
```bash
python scripts/autosad_run.py --dataset 1_ALOI --seed 42 --progress_interval 1000 --output_dir results
```

#### Robust Random Cut Forest (RRCF)
```bash
python scripts/rrcf_run.py --dataset 10_cover --seed 42 --progress_interval 1000 --output_dir results
```

#### Online Isolation Forest (OIF)
```bash
python scripts/oif_run.py --dataset 5_campaign --seed 42 --progress_interval 1000 --output_dir results
```

#### LODA
```bash
python scripts/loda_run.py --dataset 13_fraud --seed 42 --progress_interval 1000 --output_dir results
```

#### IForestASD
```bash
python scripts/ifasd_run.py --dataset 32_shuttle --seed 42 --progress_interval 1000 --output_dir results
```

#### Half-Space Trees (HST)
```bash
python scripts/hst_run.py --dataset 50_bank --seed 42 --progress_interval 1000 --output_dir results
```

#### RSHash
```bash
python scripts/rshash_run.py --dataset 48_chess --seed 42 --progress_interval 1000 --output_dir results
```

#### XStream
```bash
python scripts/xstream_run.py --dataset 49_kddcup99_prob --seed 42 --progress_interval 1000 --output_dir results
```

### Common Parameters

All scripts support the following parameters:

- `--dataset`: Name of the dataset (without .npz extension)
- `--run_count`: Run count identifier for multiple runs (optional)
- `--seed`: Random seed for reproducibility (default: 42)
- `--progress_interval`: Interval for progress updates (default: 1000)
- `--output_dir`: Directory to save results (default: "benchmark_results")
- `--verbose`: Enable verbose output (default: True)

### Multiple Runs with Different Seeds

For statistical significance, you can run experiments multiple times:

```bash
# Run AutoSAD 10 times with different seeds
for i in {1..10}; do
    python scripts/autosad_run.py --dataset 1_ALOI --run_count $i --seed $((42 + i))
done
```

## Output Format

Each experiment generates a JSON file with the following structure:

```json
{
    "dataset": "1_ALOI",
    "model": "AutoSAD",
    "auc": 0.8532,
    "total_runtime": 45.67,
    "total_memory_usage": 128.45,
    "Instances": 50000,
    "run_count": 1,
    "seed": 42,
    "window_metric": {
        "window_size": 1000,
        "auc_scores": [0.82, 0.84, 0.85, ...],
        "runtimes": [0.12, 0.13, 0.11, ...],
        "memory_usages": [45.2, 46.1, 47.3, ...]
    }
}
```

## Experiment Examples

### Basic Experiments

```bash
# Test AutoSAD on a small dataset
python scripts/autosad_run.py --dataset 15_Hepatitis

# Compare RRCF and AutoSAD on fraud detection
python scripts/rrcf_run.py --dataset 13_fraud
python scripts/autosad_run.py --dataset 13_fraud

# Run comprehensive evaluation on cover dataset
python run_scripts.py --mode dataset --name 10_cover
```

### Large-Scale Experiments

```bash
# Run all algorithms on all datasets (warning: this takes a long time)
python run_scripts.py --mode all

# Run with custom settings
python run_scripts.py --mode all --progress_interval 500 --output_dir large_scale_results
```

### Statistical Analysis

```bash
# Run multiple seeds for statistical significance
python run_scripts.py --mode random

# Or manually for specific dataset
for seed in {1..10}; do
    python scripts/autosad_run.py --dataset 1_ALOI --run_count $seed --seed $((42 + seed))
done
```

## Performance Analysis

After running experiments, you can analyze results using the included Jupyter notebook:

```bash
jupyter notebook results_table.ipynb
```

The notebook provides:
- Performance comparison tables
- Statistical significance tests
- Runtime and memory usage analysis
- Visualization of results

## Configuration

### Modifying Datasets and Models

Edit `run_scripts.py` to customize which datasets and models to run:

```python
# Modify dataset list
dataset_name_list = [
    '1_ALOI',
    '10_cover', 
    '13_fraud',
    # Add your datasets here
]

# Modify model list
models = [
    'autosad',
    'rrcf',
    'oif',
    # Add your models here
]
```

### Custom Output Directories

```bash
# Organize results by experiment type
python run_scripts.py --mode model --name autosad --output_dir experiments/autosad_full
python run_scripts.py --mode model --name rrcf --output_dir experiments/rrcf_full
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: For large datasets, reduce the progress interval or run fewer concurrent jobs
2. **Missing Dependencies**: Install required packages with `pip install numpy scipy scikit-learn psutil pysad`
3. **Rust Components**: If pysad_rust fails, ensure Rust is installed and build manually

### Installation Options

#### When to Use Standard Installation (Python only)
- Quick setup and testing
- Limited computational resources
- No Rust development environment
- Compatibility concerns

```bash
pip install -r requirements.txt
```

#### When to Use High-Performance Installation (with Rust)
- Large-scale experiments
- Performance-critical applications
- Available Rust development environment
- Long-running experiments

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install dependencies and build Rust components
pip install -r requirements.txt
cd pysad_rust
cargo build --release
cd ..
```

#### Rust Installation Troubleshooting
If Rust installation fails:
1. Check if you have a C compiler installed (Xcode on macOS, build-essential on Ubuntu)
2. Restart your terminal after installing Rust
3. Use the fallback Python-only installation if Rust continues to fail

### Performance Tips

- Use `--progress_interval 500` for more frequent updates on large datasets
- Set appropriate `--output_dir` to organize results
- Use parallel execution with `run_scripts.py` for efficiency
- Monitor memory usage during large experiments
- For very large datasets (>500k samples), consider using the Rust-accelerated version

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use AutoSAD in your research, please cite:

```bibtex
@article{autosad2024,
    title={AutoSAD: An Adaptive Framework for Streaming Anomaly Detection},
    author={Anonymous},
    journal={Anonymous},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
