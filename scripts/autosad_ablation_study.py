# Full ablation study
# python autosad_ablation_study.py --dataset dataset_name --verbose

# Specific configurations only
# python autosad_ablation_study.py --dataset dataset_name --configs baseline n_models_10 no_evolution

# Different output directory
# python autosad_ablation_study.py --dataset dataset_name --output_dir

import numpy as np
import random
import time
import psutil
import json
import os
import argparse
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.utils import ArrayStreamer
from utils import get_xy_from_npz
import traceback
from itertools import product

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AutoSAD import AutoSAD

import warnings
warnings.filterwarnings("ignore")


def run_autosad_configuration(dataset_name, config, seed=42, progress_interval=1000, output_dir="ablation_results", verbose=False):
    """Run AutoSAD with a specific configuration."""
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    config_name = config["name"]
    print(f"\nRunning AutoSAD configuration: {config_name} on {dataset_name}")
    
    try:
        # Load dataset
        X, y = get_xy_from_npz("adbenchmark/" + dataset_name + ".npz")
        X, y = shuffle(X, y, random_state=seed)
        
        # Calculate feature mins and maxes
        feature_mins = X.min(axis=0).astype(np.float64)
        feature_maxes = X.max(axis=0).astype(np.float64)
        
        PARMS = {
            "feature_mins": feature_mins,
            "feature_maxes": feature_maxes,
        }
        
        # Merge configuration parameters
        model_params = {**PARMS, **config["params"]}
        model_params["random_state"] = seed
        model_params["verbose"] = verbose
        
        # Initialize model with configuration
        model = AutoSAD(**model_params)
        
        # Setup evaluation
        processed_instances = 0
        metric = AUROCMetric()
        iterator = ArrayStreamer(shuffle=False)
        
        # Track metrics
        scores = []
        times = []
        memory_usages = []
        
        # Start timing and memory tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Process the stream
        for xi, yi in iterator.iter(X, y):
            xi = np.asarray(xi, dtype=np.float64)
            xi = np.ascontiguousarray(xi)
            
            # Fit and score the model
            score = model.fit_score_partial(xi)
            metric.update(yi, score)
            
            processed_instances += 1
            if processed_instances % progress_interval == 0:
                # Calculate AUROC incrementally
                try:
                    current_auroc = metric.get()
                    scores.append(current_auroc)
                except Exception as e:
                    if verbose:
                        print(f"Error calculating AUROC at instance {processed_instances}: {str(e)}")
                    current_auroc = 0.5
                    scores.append(current_auroc)
                
                # Update time and memory
                instance_time = time.time() - start_time
                times.append(instance_time)
                instance_memory = psutil.Process().memory_info().rss - start_memory
                memory_usage = abs(instance_memory / 1024 / 1024)  # MB
                memory_usages.append(memory_usage)
                
                if verbose:
                    print(f"{config_name}: Processed {processed_instances} instances, AUROC: {current_auroc:.4f}, Time: {instance_time:.2f}s, Memory: {memory_usage:.2f}MB")
        
        # Calculate final metrics
        final_auroc = metric.get()
        total_runtime = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss - start_memory
        memory_usage = abs(memory_usage / 1024 / 1024)  # MB
        
        # Store results
        results = {
            "dataset": dataset_name,
            "configuration": config_name,
            "config_params": config["params"],
            "auc": final_auroc,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "instances": processed_instances,
            "seed": seed,
            "window_metric": {
                "window_size": progress_interval,
                "auc_scores": scores,
                "runtimes": times,
                "memory_usages": memory_usages
            }
        }
        
        print(f"Configuration {config_name} completed - AUROC: {final_auroc:.4f}, Runtime: {total_runtime:.2f}s")
        return results
        
    except Exception as e:
        error_info = {
            'configuration': config_name,
            'dataset': dataset_name,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'line_number': traceback.extract_tb(e.__traceback__)[-1].lineno,
            'stack_trace': traceback.format_exc()
        }
        print(f"Error in configuration {config_name}: {error_info['error_message']}")
        if verbose:
            print(error_info['stack_trace'])
        return None


def generate_ablation_configurations():
    """Generate different configurations for ablation study."""
    
    configurations = []
    
    # 1. Baseline configuration
    baseline_config = {
        "name": "baseline",
        "params": {
            "n_models": 5,
            "acq_strategy": "UCB",
            "evolution_interval": 1000,
            "enable_evolution": True,
            "enable_diversity": True
        }
    }
    configurations.append(baseline_config)
    
    # 2. Number of models ablation
    n_models_values = [3, 5, 8, 10, 15, 20]
    for n_models in n_models_values:
        if n_models != 5:  # Skip baseline
            config = {
                "name": f"n_models_{n_models}",
                "params": {
                    "n_models": n_models,
                    "acq_strategy": "UCB",
                    "evolution_interval": 1000,
                    "enable_evolution": True,
                    "enable_diversity": True
                }
            }
            configurations.append(config)
    
    # 3. Acquisition function ablation
    acq_strategies = ["UCB", "EI", "PI"]
    for acq_strategy in acq_strategies:
        if acq_strategy != "UCB":  # Skip baseline
            config = {
                "name": f"acq_{acq_strategy}",
                "params": {
                    "n_models": 5,
                    "acq_strategy": acq_strategy,
                    "evolution_interval": 1000,
                    "enable_evolution": True,
                    "enable_diversity": True
                }
            }
            configurations.append(config)
    
    # 4. Evolution interval ablation
    evolution_intervals = [500, 750, 1000, 1500, 2000, 3000]
    for interval in evolution_intervals:
        if interval != 1000:  # Skip baseline
            config = {
                "name": f"evolution_interval_{interval}",
                "params": {
                    "n_models": 5,
                    "acq_strategy": "UCB",
                    "evolution_interval": interval,
                    "enable_evolution": True,
                    "enable_diversity": True
                }
            }
            configurations.append(config)
    
    # 5. Evolution enable/disable ablation
    config_no_evolution = {
        "name": "no_evolution",
        "params": {
            "n_models": 5,
            "acq_strategy": "UCB",
            "evolution_interval": 1000,
            "enable_evolution": False,
            "enable_diversity": True
        }
    }
    configurations.append(config_no_evolution)
    
    # 6. Diversity enable/disable ablation
    config_no_diversity = {
        "name": "no_diversity",
        "params": {
            "n_models": 5,
            "acq_strategy": "UCB",
            "evolution_interval": 1000,
            "enable_evolution": True,
            "enable_diversity": False
        }
    }
    configurations.append(config_no_diversity)
    
    # 7. Combined ablations - no evolution and no diversity
    config_no_evolution_no_diversity = {
        "name": "no_evolution_no_diversity",
        "params": {
            "n_models": 5,
            "acq_strategy": "UCB",
            "evolution_interval": 1000,
            "enable_evolution": False,
            "enable_diversity": False
        }
    }
    configurations.append(config_no_evolution_no_diversity)
    
    # 8. Extreme configurations
    # Minimal configuration
    config_minimal = {
        "name": "minimal",
        "params": {
            "n_models": 3,
            "acq_strategy": "UCB",
            "evolution_interval": 3000,
            "enable_evolution": False,
            "enable_diversity": False
        }
    }
    configurations.append(config_minimal)
    
    # Maximal configuration
    config_maximal = {
        "name": "maximal",
        "params": {
            "n_models": 20,
            "acq_strategy": "UCB",
            "evolution_interval": 500,
            "enable_evolution": True,
            "enable_diversity": True
        }
    }
    configurations.append(config_maximal)
    
    return configurations


def run_ablation_study(dataset_name, seed=42, progress_interval=1000, output_dir="ablation_results", verbose=False, specific_configs=None):
    """Run complete ablation study on a dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== AutoSAD Ablation Study on {dataset_name} ===")
    
    # Generate configurations
    if specific_configs is None:
        configurations = generate_ablation_configurations()
    else:
        all_configs = generate_ablation_configurations()
        configurations = [config for config in all_configs if config["name"] in specific_configs]
    
    print(f"Total configurations to test: {len(configurations)}")
    
    # Store all results
    all_results = []
    
    # Run each configuration
    for i, config in enumerate(configurations, 1):
        print(f"\nConfiguration {i}/{len(configurations)}: {config['name']}")
        
        result = run_autosad_configuration(
            dataset_name=dataset_name,
            config=config,
            seed=seed,
            progress_interval=progress_interval,
            output_dir=output_dir,
            verbose=verbose
        )
        
        if result is not None:
            all_results.append(result)
            
            # Save individual result
            config_file = f"ablation_{dataset_name}_{config['name']}_seed{seed}.json"
            config_path = os.path.join(output_dir, config_file)
            with open(config_path, 'w') as f:
                json.dump(result, f, indent=4)
                
        print(f"Completed {i}/{len(configurations)} configurations")
    
    # Save summary results
    summary_results = {
        "dataset": dataset_name,
        "seed": seed,
        "total_configurations": len(configurations),
        "successful_runs": len(all_results),
        "results": all_results
    }
    
    summary_file = f"ablation_summary_{dataset_name}_seed{seed}.json"
    summary_path = os.path.join(output_dir, summary_file)
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    
    # Generate analysis report
    generate_analysis_report(all_results, dataset_name, output_dir, seed)
    
    print(f"\nAblation study completed!")
    print(f"Results saved to {output_dir}")
    print(f"Summary: {summary_path}")
    
    return all_results


def generate_analysis_report(results, dataset_name, output_dir, seed):
    """Generate a human-readable analysis report."""
    
    if not results:
        print("No results to analyze")
        return
    
    # Sort results by AUROC for ranking
    sorted_results = sorted(results, key=lambda x: x["auc"], reverse=True)
    
    report_lines = []
    report_lines.append(f"AutoSAD Ablation Study Analysis Report")
    report_lines.append(f"Dataset: {dataset_name}")
    report_lines.append(f"Seed: {seed}")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Performance ranking
    report_lines.append("PERFORMANCE RANKING (by AUROC):")
    report_lines.append("-" * 40)
    for i, result in enumerate(sorted_results, 1):
        config_name = result["configuration"]
        auc = result["auc"]
        runtime = result["total_runtime"]
        memory = result["total_memory_usage"]
        report_lines.append(f"{i:2d}. {config_name:25s} - AUROC: {auc:.4f}, Time: {runtime:6.1f}s, Memory: {memory:6.1f}MB")
    
    report_lines.append("")
    
    # Component analysis
    report_lines.append("COMPONENT ANALYSIS:")
    report_lines.append("-" * 40)
    
    # Find baseline for comparison
    baseline_result = next((r for r in results if r["configuration"] == "baseline"), None)
    if baseline_result:
        baseline_auc = baseline_result["auc"]
        report_lines.append(f"Baseline AUROC: {baseline_auc:.4f}")
        report_lines.append("")
        
        # Analyze different components
        component_groups = {
            "Number of Models": [r for r in results if r["configuration"].startswith("n_models_")],
            "Acquisition Function": [r for r in results if r["configuration"].startswith("acq_")],
            "Evolution Interval": [r for r in results if r["configuration"].startswith("evolution_interval_")],
            "Evolution Disabled": [r for r in results if "no_evolution" in r["configuration"]],
            "Diversity Disabled": [r for r in results if "no_diversity" in r["configuration"]],
        }
        
        for group_name, group_results in component_groups.items():
            if group_results:
                report_lines.append(f"{group_name}:")
                for result in sorted(group_results, key=lambda x: x["auc"], reverse=True):
                    config_name = result["configuration"]
                    auc = result["auc"]
                    auc_diff = auc - baseline_auc
                    sign = "+" if auc_diff >= 0 else ""
                    report_lines.append(f"  {config_name:25s} - AUROC: {auc:.4f} ({sign}{auc_diff:+.4f})")
                report_lines.append("")
    
    # Key findings
    report_lines.append("KEY FINDINGS:")
    report_lines.append("-" * 40)
    
    # Best configuration
    best_config = sorted_results[0]
    report_lines.append(f"Best configuration: {best_config['configuration']} (AUROC: {best_config['auc']:.4f})")
    
    # Worst configuration
    worst_config = sorted_results[-1]
    report_lines.append(f"Worst configuration: {worst_config['configuration']} (AUROC: {worst_config['auc']:.4f})")
    
    # Performance range
    auc_range = best_config['auc'] - worst_config['auc']
    report_lines.append(f"Performance range: {auc_range:.4f}")
    
    # Runtime analysis
    fastest_config = min(results, key=lambda x: x["total_runtime"])
    slowest_config = max(results, key=lambda x: x["total_runtime"])
    report_lines.append(f"Fastest configuration: {fastest_config['configuration']} ({fastest_config['total_runtime']:.1f}s)")
    report_lines.append(f"Slowest configuration: {slowest_config['configuration']} ({slowest_config['total_runtime']:.1f}s)")
    
    # Memory analysis
    lowest_memory = min(results, key=lambda x: x["total_memory_usage"])
    highest_memory = max(results, key=lambda x: x["total_memory_usage"])
    report_lines.append(f"Lowest memory usage: {lowest_memory['configuration']} ({lowest_memory['total_memory_usage']:.1f}MB)")
    report_lines.append(f"Highest memory usage: {highest_memory['configuration']} ({highest_memory['total_memory_usage']:.1f}MB)")
    
    # Save report
    report_file = f"ablation_analysis_{dataset_name}_seed{seed}.txt"
    report_path = os.path.join(output_dir, report_file)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print("\n" + '\n'.join(report_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AutoSAD ablation study')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Interval for progress updates')
    parser.add_argument('--output_dir', type=str, default="ablation_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--configs', nargs='+', help='Specific configurations to run (optional)')
    
    args = parser.parse_args()
    
    run_ablation_study(
        dataset_name=args.dataset,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        specific_configs=args.configs
    )
