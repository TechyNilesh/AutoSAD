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

class PostProcessor:
    def __init__(self):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._count = 0

    def process(self, score: float) -> float:
        """
        Update running min/max and sum/count with `score`,
        then return the normalized value in [0,1].
        """
        # update min/max
        if score < self._min:
            self._min = score
        if score > self._max:
            self._max = score

        # update sum/count
        self._sum += score
        self._count += 1

        # normalize this score
        span = self._max - self._min
        if span == 0:
            return 0.0
        return (score - self._min) / span

    def get_average(self) -> float:
        """
        Return the **normalized** average of all seen scores,
        i.e. (mean - min) / (max - min), in [0,1].
        If no or constant data, returns 0.0.
        """
        if self._count == 0 or self._max == self._min:
            return 0.0

        mean = self._sum / self._count
        return (mean - self._min) / (self._max - self._min)

class AutoSAD(ABC):
    def __init__(
        self,
        n_models: int = 5,
        random_seed: int = 42,
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

    def random_model_with_params(self):
        """Return a random model and its randomly sampled hyperparameters."""
        types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
        cls = np.random.choice(types)
        params = {}
        if cls is HalfSpaceTrees:
            params = {
                "num_trees": np.random.randint(8, 32),      # Number of trees in ensemble
                "max_depth": np.random.randint(5, 15),      # Maximum depth of each tree
                "window_size": np.random.randint(50, 250),  # Size of sliding window
                "feature_mins": self.feature_mins,          # Min values for feature scaling
                "feature_maxes": self.feature_maxes         # Max values for feature scaling
            }
        elif cls is IForestASD:
            params = {
                "window_size": np.random.randint(128, 1024),
                "n_estimators": np.random.randint(16, 128),
                "max_samples": np.random.randint(64, 512)
            }
        elif cls is RobustRandomCutForest:
            params = {
                "num_trees": np.random.randint(16, 128),
                "tree_size": np.random.randint(64, 512)
            }
        elif cls is LODA:
            params = {
                "num_bins": np.random.randint(50, 200),
                "num_random_cuts": np.random.randint(5, 50)
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
        return cls(**params), params

    def _init_pool(self):
        """Initialize a diverse pool of anomaly detectors"""
        #types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
        
        pool = []
        model_params = []
        # Start with one of each type
        params = {
            "num_trees": 32,     # Default number of trees for balanced performance
            "max_depth": 15,     # Default tree depth for good feature space partitioning
            "window_size": 250,  # Default window size for temporal adaptation
            "feature_mins": self.feature_mins,    # Feature scaling bounds
            "feature_maxes": self.feature_maxes   # Feature scaling bounds
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
            model, params = self.random_model_with_params()
            pool.append(model)
            model_params.append(params.copy())

        self.models = pool
        self.model_params = model_params
        self._best_model_idx = np.random.randint(0, self.n_models)
        self.scores = [PostProcessor() for _ in range(self.n_models)]
        
    def mutate(self, model, params):
        """Create a mutated version of a model using its params dict with normal distribution"""
        new_params = params.copy()
        
        def sample_normal(current_value, min_val, max_val, std_dev_ratio=0.5):
            """Sample from normal distribution centered at current value and bounded by min/max"""
            # Standard deviation as a percentage of the parameter range
            std_dev = (max_val - min_val) * std_dev_ratio
            # Sample and clip to ensure values stay in valid range
            new_val = np.random.normal(current_value, std_dev)
            return int(np.clip(new_val, min_val, max_val))
        
        if isinstance(model, HalfSpaceTrees):
            new_params["num_trees"] = sample_normal(params["num_trees"], 8, 32)
            new_params["max_depth"] = sample_normal(params["max_depth"], 5, 15)
            new_params["window_size"] = sample_normal(params["window_size"], 50, 250)
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
            score = self.scores[i].process(score)
            current_scores.append(score)

        self.step += 1

        # Every 1000 steps, evolve and update best model index for next window
        if self.step % 1000 == 0:
            print(f"Step {self.step}: Evolving models")

            scores = [s.get_average() for s in self.scores]

            print(f"Scores: {scores}")
            # Find the index of the best model (lowest score)
            
            best_idx = np.argmax(scores)  # Lower score is better

            # Update best model index for next 1000 iterations
            self._best_model_idx = best_idx
            print(f"Best model for next window: {type(self.models[best_idx]).__name__} (index: {best_idx})")
            
            best_model = self.models[best_idx]
            best_params = self.model_params[best_idx]
            self.scores[best_idx] =  PostProcessor() # Reset statistics
            
            # Sort indices by score (ascending order - worst first)
            sorted_indices = np.argsort(scores)

            print(f"Sorted Scores: {sorted_indices}")
            
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
                self.scores[idx] = PostProcessor()  # Reset statistics
            
            # Replace with completely new models
            for i in range(n_mutations, n_models_to_replace):
                idx = models_to_replace[i]
                new_model, new_params = self.random_model_with_params()
                self.models[idx] = new_model
                self.model_params[idx] = new_params
                self.scores[idx] = PostProcessor() # Reset statistics
                                
            print(f"Replaced {n_mutations} models with mutations and {n_new} with new models")

        # Always use the best model from the previous window for scoring
        #return current_scores[self._best_model_idx]
        return max(current_scores)  # Return the best score from the current models