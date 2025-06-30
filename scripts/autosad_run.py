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

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AutoSAD import AutoSAD

import warnings
warnings.filterwarnings("ignore")


def run_autosad(dataset_name, run_count=None, seed=42, progress_interval=1000, output_dir="benchmark_results", verbose=False):
    """Run AutoSAD on a dataset."""
    
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n===Running AutoSAD on {dataset_name}===")
    
    try:
        # Load dataset
        X, y = get_xy_from_npz("adbenchmark/" + dataset_name + ".npz")
        #X, y = shuffle(X, y, random_state=seed)
        
        # Calculate feature mins and maxes
        feature_mins = X.min(axis=0).astype(np.float64)
        feature_maxes = X.max(axis=0).astype(np.float64)
        
        PARMS  = {
            "feature_mins": feature_mins,
            "feature_maxes": feature_maxes,
        }
        
        # Initialize model
        model = AutoSAD(
            # window_size=1000,
            # n_models=10,
            random_state=seed,
            verbose=verbose,
            **PARMS
        )
        
        # Setup evaluation
        processed_instances = 0
        metric = AUROCMetric()
        iterator = ArrayStreamer(shuffle=False)
        
        # Track metrics
        scores = []
        times = []
        memory_usages = []
        
        # Start timing
        start_time = time.time()
        
        # start memory usage
        start_memory = psutil.Process().memory_info().rss
        
        # Process the stream test the train (preqeuential)
        for xi, yi in iterator.iter(X, y):
            
            xi = np.asarray(xi, dtype=np.float64)  # Ensure correct dtype
            xi = np.ascontiguousarray(xi)          # Ensure contiguous memory
            
            # fit and score the model
            score = model.fit_score_partial(xi)
            metric.update(yi, score)
            
            # record the instance
            processed_instances += 1
            if processed_instances % progress_interval == 0:
                
                # Calculate AUROC Incrementally
                try:
                    current_auroc = metric.get()
                    scores.append(current_auroc)
                except Exception as e:
                    if verbose:
                        print(f"Error calculating AUROC at instance {processed_instances}: {str(e)}", flush=True)
                    current_auroc = 0.5  # Default AUROC if error occurs
                    scores.append(current_auroc)  # Default AUROC if error occurs
                
                # Incremental Time and Memory Usage update
                isinstance_time = time.time() - start_time # seconds
                times.append(isinstance_time)
                isinstance_memory = psutil.Process().memory_info().rss - start_memory
                memory_usage = abs(isinstance_memory / 1024 / 1024)  # MB
                memory_usages.append(memory_usage)
                
                if verbose:
                    if run_count is not None:
                        print(f"AutoSAD on {dataset_name} (Run {run_count}): Processed {processed_instances} instances, AUROC: {current_auroc:.4f}, Time: {isinstance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)
                    else:
                        print(f"AutoSAD on {dataset_name}: Processed {processed_instances} instances, AUROC: {current_auroc:.4f}, Time: {isinstance_time:.2f} seconds, Memory: {memory_usage:.2f} MB", flush=True)        
        
        # Calculate final metrics
        final_auroc = metric.get()
        total_runtime = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss - start_memory
        memory_usage = abs(memory_usage / 1024 / 1024)  # MB
        
        # Store results
        results = {
            "dataset": dataset_name,
            "model": "AutoSAD", 
            "auc": final_auroc,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "Instances": processed_instances,
            "window_metric": {
                "window_size": progress_interval,
                "auc_scores": scores,
                "runtimes": times,
                "memory_usages": memory_usages
            }
        }
        
        # Generate file name based on run count
        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"{results['model']}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"{results['model']}_{dataset_name}.json"
        
        # Save results to file
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"AutoSAD completed on {dataset_name}")
        print(f"Final AUROC: {final_auroc:.4f}")
        print(f"Runtime: {total_runtime:.2f} seconds")
        print(f"Memory: {memory_usage:.2f} MB")
        print(f"Processed {processed_instances} instances")
        print(f"Results saved to {output_file}")
        
        return results
        
    except Exception as e:
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'line_number': traceback.extract_tb(e.__traceback__)[-1].lineno,
            'stack_trace': traceback.format_exc()
        }
        print(f"\nError Details:")
        print(f"Type: {error_info['error_type']}")
        print(f"Message: {error_info['error_message']}")
        print(f"Line Number: {error_info['line_number']}")
        print("\nFull Stack Trace:")
        print(error_info['stack_trace'])
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AutoSAD on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='name of the dataset to run HST on')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Interval for progress updates')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output',default=False)
    
    args = parser.parse_args()
    
    run_autosad(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose
    )