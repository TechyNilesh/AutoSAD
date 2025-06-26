import numpy as np
import random
from abc import ABC
from pysad_rust import (
    HalfSpaceTrees,
    IForestASD,
    RobustRandomCutForest,
    LODA,
    OnlineIsolationForest
)
from pysad_rust import StreamStatistic

class AutoSAD(ABC):
    def __init__(
        self,
        n_models: int = 10,
        random_seed: int = 52,
        **kwargs
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.n_models = n_models

        # Will be initialized on first data
        self.feature_mins = kwargs.get('feature_mins', None)
        self.feature_maxes = kwargs.get('feature_maxes', None)
        self.models = None
        self.scores = None
        self.step = 0
        self._best_model_idx = None
        self._init_pool()

    def _init_pool(self):
        """Initialize a diverse pool of anomaly detectors"""
        types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
        
        pool = []
        model_params = []
        # Start with one of each type
        params = {
            "num_trees": 32, "max_depth": 15, "window_size": 250,
            "feature_mins": self.feature_mins,
            "feature_maxes": self.feature_maxes
        }
        pool.append(HalfSpaceTrees(**params))
        model_params.append(params.copy())
        params = {
            "window_size": 512, "n_estimators": 32, "max_samples": 256
        }
        pool.append(IForestASD(**params))
        model_params.append(params.copy())
        params = {
            "num_trees": 32, "tree_size": 256
        }
        pool.append(RobustRandomCutForest(**params))
        model_params.append(params.copy())
        params = {
            "num_bins": 100, "num_random_cuts": 10
        }
        pool.append(LODA(**params))
        model_params.append(params.copy())
        params = {
            "branching_factor": 2, "split": "axisparallel",
            "num_trees": 32, "max_leaf_samples": 32,
            "window_size": 2048, "growth_criterion": "adaptive",
            "subsample": 1.0
        }
        pool.append(OnlineIsolationForest(**params))
        model_params.append(params.copy())

        # Fill remaining slots with random types
        while len(pool) < self.n_models:
            cls = np.random.choice(types)
            params = {}
            if cls is HalfSpaceTrees:
                params = {
                    "num_trees": np.random.randint(5,50),
                    "max_depth": np.random.randint(5,25),
                    "window_size": np.random.randint(50,500),
                    "feature_mins": self.feature_mins,
                    "feature_maxes": self.feature_maxes
                }
            elif cls is IForestASD:
                params = {
                    "window_size": np.random.randint(128,1024),
                    "n_estimators": np.random.randint(16,128),
                    "max_samples": np.random.randint(64,512)
                }
            elif cls is RobustRandomCutForest:
                params = {
                    "num_trees": np.random.randint(16,128),
                    "tree_size": np.random.randint(64,512)
                }
            elif cls is LODA:
                params = {
                    "num_bins": np.random.randint(50,200),
                    "num_random_cuts": np.random.randint(5,50)
                }
            else:  # OnlineIsolationForest
                params = {
                    "branching_factor": np.random.choice([2, 3, 4]),
                    "split": "axisparallel",
                    "num_trees": np.random.randint(16, 128),
                    "max_leaf_samples": np.random.randint(16, 64),
                    "window_size": np.random.randint(512, 4096),
                    "growth_criterion": "adaptive",
                    "subsample": np.random.uniform(0.5, 1.0)
                }
            pool.append(cls(**params))
            model_params.append(params.copy())

        self.models = pool
        self.model_params = model_params
        self._best_model_idx = np.random.randint(0, self.n_models)
        self.scores = [StreamStatistic() for _ in range(self.n_models)]
        
    def mutate(self, model, params):
        """Create a mutated version of a model using its params dict with normal distribution"""
        new_params = params.copy()
        
        def sample_normal(current_value, min_val, max_val, std_dev_ratio=0.1):
            """Sample from normal distribution centered at current value and bounded by min/max"""
            # Standard deviation as a percentage of the parameter range
            std_dev = (max_val - min_val) * std_dev_ratio
            # Sample and clip to ensure values stay in valid range
            new_val = np.random.normal(current_value, std_dev)
            return int(np.clip(new_val, min_val, max_val))
        
        if isinstance(model, HalfSpaceTrees):
            new_params["num_trees"] = sample_normal(params["num_trees"], 5, 50)
            new_params["max_depth"] = sample_normal(params["max_depth"], 5, 25)
            new_params["window_size"] = sample_normal(params["window_size"], 50, 500)
            # feature_mins/maxes remain unchanged
            return HalfSpaceTrees(**new_params), new_params
        
        elif isinstance(model, IForestASD):
            new_params["window_size"] = sample_normal(params["window_size"], 128, 1024)
            new_params["n_estimators"] = sample_normal(params["n_estimators"], 16, 128)
            new_params["max_samples"] = sample_normal(params["max_samples"], 64, 512)
            return IForestASD(**new_params), new_params
        
        elif isinstance(model, RobustRandomCutForest):
            new_params["num_trees"] = sample_normal(params["num_trees"], 16, 128)
            new_params["tree_size"] = sample_normal(params["tree_size"], 64, 512)
            return RobustRandomCutForest(**new_params), new_params
        
        elif isinstance(model, LODA):
            new_params["num_bins"] = sample_normal(params["num_bins"], 50, 200)
            new_params["num_random_cuts"] = sample_normal(params["num_random_cuts"], 5, 50)
            return LODA(**new_params), new_params
        
        elif isinstance(model, OnlineIsolationForest):
            new_params["branching_factor"] = np.random.choice([2, 3, 4])  # Discrete parameter
            new_params["num_trees"] = sample_normal(params["num_trees"], 16, 128)
            new_params["max_leaf_samples"] = sample_normal(params["max_leaf_samples"], 16, 64)
            new_params["window_size"] = sample_normal(params["window_size"], 512, 4096)
            new_params["subsample"] = np.clip(np.random.normal(params["subsample"], 0.1), 0.1, 1.0)
            # Keep split and growth_criterion unchanged as they are string parameters
            return OnlineIsolationForest(**new_params), new_params
        
        return model, params

    def fit_score_partial(self, X):
        current_scores = []
        for i, model in enumerate(self.models):
            score = model.fit_score_partial(X)
            self.scores[i].update(np.array([score], dtype=np.float64))
            current_scores.append(score)

        self.step += 1

        # Every 1000 steps, evolve and update best model index for next window
        if self.step % 1000 == 0:
            print(f"Step {self.step}: Evolving models")
            
            # Get performance of all models
            mean_scores = np.array([s.get_mean()[0] for s in self.scores])
            
            # claculate  normalized scores higher is better
            normalized_scores = mean_scores / np.max(mean_scores)
    

            best_idx = np.argmax(normalized_scores)  # Lower score is better

            # Update best model index for next 1000 iterations
            self._best_model_idx = best_idx
            print(f"Best model for next window: {type(self.models[best_idx]).__name__} (index: {best_idx})")
            
            best_model = self.models[best_idx]
            best_params = self.model_params[best_idx]
            
            # Sort indices by score (ascending order - worst first)
            sorted_indices = np.argsort(normalized_scores)
            
            # Keep the best model, replace others
            models_to_replace = sorted_indices[:-1]  # All except best
            np.random.shuffle(models_to_replace)  # Randomize replacement order
            
            # Calculate how many models to replace with mutations vs new models
            n_models_to_replace = len(models_to_replace)
            n_mutations = n_models_to_replace // 2  # Half mutations
            n_new = n_models_to_replace - n_mutations  # Half new models
            
            # Replace with mutations of the best
            for i in range(n_mutations):
                idx = models_to_replace[i]
                new_model, new_params = self.mutate(best_model, best_params)
                self.models[idx] = new_model
                self.model_params[idx] = new_params
                self.scores[idx] = StreamStatistic()  # Reset statistics
            
            # Replace with completely new models
            types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
            for i in range(n_mutations, n_models_to_replace):
                idx = models_to_replace[i]
                
                # Create a new model with default parameters
                cls = np.random.choice(types)
                params = {}
                
                if cls is HalfSpaceTrees:
                    params = {
                        "num_trees": 32, "max_depth": 15, "window_size": 250,
                        "feature_mins": self.feature_mins,
                        "feature_maxes": self.feature_maxes
                    }
                elif cls is IForestASD:
                    params = {
                        "window_size": 512, "n_estimators": 32, "max_samples": 256
                    }
                elif cls is RobustRandomCutForest:
                    params = {
                        "num_trees": 32, "tree_size": 256
                    }
                elif cls is LODA:
                    params = {
                        "num_bins": 100, "num_random_cuts": 10
                    }
                else:  # OnlineIsolationForest
                    params = {
                        "branching_factor": 2, "split": "axisparallel",
                        "num_trees": 32, "max_leaf_samples": 32,
                        "window_size": 2048, "growth_criterion": "adaptive",
                        "subsample": 1.0
                    }
                    
                self.models[idx] = cls(**params)
                self.model_params[idx] = params
                self.scores[idx] = StreamStatistic()  # Reset statistics
                                
            print(f"Replaced {n_mutations} models with mutations and {n_new} with new models")

        # Always use the best model from the previous window for scoring
        return current_scores[self._best_model_idx]