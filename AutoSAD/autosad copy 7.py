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
from scipy.stats import truncnorm

class PostProcessor:
    def __init__(self):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._count = 0
        self._sum_sq = 0.0  # Added to store sum of squares

    def process(self, score: float) -> float:
        """
        Update running min/max, sum/count, and sum_sq with `score`,
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
        self._sum_sq += score**2 # Added to update sum of squares

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

    def get_variance(self) -> float:
        """
        Return the variance of all seen (raw) scores.
        If count < 2, returns 0.0.
        """
        if self._count < 2:
            return 0.0

        mean = self._sum / self._count
        variance = (self._sum_sq / self._count) - (mean**2)
        return variance

class AutoSAD(ABC):
    def __init__(
        self,
        n_models: int = 5,
        random_state: int = 52,
        verbose: bool = False,
        **kwargs
    ):
        self.n_models = n_models
        self.random_state = random_state  # Store the random seed
        self.verbose = verbose  # Control print statements

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Will be initialized on first data
        self.feature_mins = kwargs.get('feature_mins', None)
        self.feature_maxes = kwargs.get('feature_maxes', None)
        self.models = None
        self.scores = None
        self.step = 0
        self._best_model_idx = None
        self.exploration_std = 1.0  # Initial exploration standard deviation
        self._init_pool()

    def _get_hyperparameter_options(self):
        """Return a dictionary with model-specific hyperparameter options"""
        return {
            "HalfSpaceTrees": {
                "num_trees": [8, 16, 32, 64, 128],
                "max_depth": [5, 8, 10, 12, 15],
                "window_size": [50, 100, 150, 200, 250],
                # feature_mins and feature_maxes are set at instantiation
            },
            "IForestASD": {
                "window_size": [512, 1024, 2048, 4096],
                "n_estimators": [16, 32, 64, 128],
                "max_samples": [64, 128, 256, 512],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "anomaly_rate_threshold": [0.1, 0.2, 0.3, 0.4]
            },
            "RobustRandomCutForest": {
                "num_trees": [8, 16, 32, 64, 128],
                "tree_size": [64, 128, 256, 512],
                "shingle_size": [1, 2, 3]
            },
            "LODA": {
                "num_bins": [50, 100, 150, 200],
                "num_random_cuts": [5, 10, 25, 50]
            },
            "OnlineIsolationForest": {
                "branching_factor": [2, 3, 4],
                "split": ["axisparallel", "hyperplane"],
                "num_trees": [8, 16, 32, 64, 128],
                "max_leaf_samples": [16, 32, 64],
                "window_size": [512, 1024, 2048, 4096],
                "growth_criterion": ["adaptive", "depth", "size"],
                "subsample": [0.1, 0.25, 0.5, 0.75, 1.0]
            }
        }

    def _pwhs(self, current_value, options, sigma=1.0):
        """
        Probabilistic weighted hyperparameter selection
        Uses truncated normal distribution with adaptive standard deviation for numeric values 
        and weighted choice for categorical
        """
        if isinstance(current_value, (int, float)):
            lower_bound, upper_bound = min(options), max(options)
            
            mu = current_value
            
            # Use the exploration standard deviation parameter
            std_options = np.std(options) * sigma

            # Truncate the normal distribution to stay within bounds
            a_norm, b_norm = (lower_bound - mu) / std_options, (upper_bound - mu) / std_options

            try:
                # Sample a value from the truncated normal distribution
                new_value = truncnorm.rvs(a_norm, b_norm, loc=mu, scale=std_options)
                # Find the closest value in the options to the sampled value
                new_value = min(options, key=lambda x: abs(x - new_value))
            except ValueError:
                if self.verbose:
                    print(f"**ValueError: {a_norm}, {b_norm}, {mu}, {std_options}**")
                    print("**Assigning random value**")
                new_value = np.random.choice(options)
            
            if isinstance(current_value, int):
                new_value = int(round(new_value))
        
        else: # Categorical or boolean hyperparameter and other types
            # Adaptively adjust probability based on exploration standard deviation
            p_current = max(0.5, 1.0 - 0.5 * sigma)  # Higher sigma means lower probability of keeping current value
            
            try:
                probabilities = [p_current if val == current_value else (1-p_current)/(len(options)-1) for val in options]
                new_value = np.random.choice(options, p=probabilities)
            except:
                if self.verbose:
                    print(f"**Error: {current_value}, {options}, {sigma}**")
                    print("**Assigning random value**")
                new_value = np.random.choice(options)
        
        return new_value

    def random_model_with_params(self):
        """Return a random model and its randomly sampled hyperparameters."""
        types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
        cls = np.random.choice(types)
        
        # Get model name to retrieve appropriate hyperparameter options
        model_name = cls.__name__
        options = self._get_hyperparameter_options()[model_name]
        
        params = {"random_state": self.random_state}
        
        # Select random parameter values from options
        for param, values in options.items():
            if param not in ["feature_mins", "feature_maxes"]:  # Skip these special parameters
                params[param] = np.random.choice(values)
                
        # Add special parameters for specific models
        if cls is HalfSpaceTrees:
            params["feature_mins"] = self.feature_mins
            params["feature_maxes"] = self.feature_maxes
            
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
            "feature_maxes": self.feature_maxes,   # Feature scaling bounds
            "random_state": self.random_state     # Reproducible randomness
        }
        pool.append(HalfSpaceTrees(**params))
        model_params.append(params.copy())
        
        params = {
            "window_size": 2048,
            "n_estimators": 32,
            "max_samples": 256,
            'contamination':0.1,
            'anomaly_rate_threshold': 0.3,
            "random_state": self.random_state
        }
        pool.append(IForestASD(**params))
        model_params.append(params.copy())
        
        params = {
            "num_trees": 32,
            'shingle_size': 1,
            "tree_size": 256,
            "random_state": self.random_state
        }
        pool.append(RobustRandomCutForest(**params))
        model_params.append(params.copy())
        
        params = {
            "num_bins": 100,
            "num_random_cuts": 32,
            "random_state": self.random_state
        }
        pool.append(LODA(**params))
        model_params.append(params.copy())
        
        params = {
            "branching_factor": 2,
            "split": "axisparallel",
            "num_trees": 32,
            "max_leaf_samples": 32,
            "window_size": 2048,
            "growth_criterion": "adaptive",
            "subsample": 1.0,
            "random_state": self.random_state
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
        """Create a mutated version of a model using probabilistic parameter selection"""
        model_name = type(model).__name__
        options = self._get_hyperparameter_options()[model_name]
        new_params = params.copy()
        
        # Generate a new random state for the mutated model
        new_params["random_state"] = self.random_state
        
        # Use exploration standard deviation for mutation
        exploration_sigma = self.exploration_std
        
        # Mutate each parameter using the _pwhs function with the current exploration std
        for param, values in options.items():
            if param in params and param not in ["feature_mins", "feature_maxes"]:
                new_params[param] = self._pwhs(params[param], values, exploration_sigma)
        
        # Create and return new model instance with mutated parameters
        if model_name == "HalfSpaceTrees":
            new_params["feature_mins"] = self.feature_mins
            new_params["feature_maxes"] = self.feature_maxes
            return HalfSpaceTrees(**new_params), new_params
        elif model_name == "IForestASD":
            return IForestASD(**new_params), new_params
        elif model_name == "RobustRandomCutForest":
            return RobustRandomCutForest(**new_params), new_params
        elif model_name == "LODA":
            return LODA(**new_params), new_params
        elif model_name == "OnlineIsolationForest":
            return OnlineIsolationForest(**new_params), new_params
        
        return model, params

    def fit_score_partial(self, X):

        current_scores = []
        for i, model in enumerate(self.models):
            score = model.fit_score_partial(X)
            score = self.scores[i].process(score)
            current_scores.append(score)

        self.step += 1

        # Every 1000 steps, evolve models
        if self.step % 1000 == 0:
            if self.verbose:
                print(f"Step {self.step}: Evolving models")

            scores = [s.get_average() for s in self.scores]
            if self.verbose:
                print(f"Current scores: {scores}")

            # Update exploration standard deviation using the formula: σ_{t+1} = σ_t × 2^{-λ}
            # We'll use average score to determine λ
            avg_score = np.mean(scores)
            lambda_decay = avg_score  # Higher scores mean faster decay
            self.exploration_std *= 2**(-lambda_decay)
            
            # Keep exploration within reasonable bounds
            self.exploration_std = max(0.1, min(self.exploration_std, 2.0))
            
            if self.verbose:
                print(f"Updated exploration std: {self.exploration_std:.4f} (decay factor: {lambda_decay:.4f})")

            # Sort indices by score (ascending order)
            sorted_indices = np.argsort(scores)
            # Calculate number of top models to keep (half of total)
            n_models = len(self.models)
            n_top = n_models // 2
            
            # Get indices of top half models
            top_indices = sorted_indices[-n_top:]  # Get best performing half
            
            # Get indices of bottom half to replace
            bottom_indices = sorted_indices[:n_top]  
            
            if self.verbose:
                print(f"Top {n_top} models will be kept and mutated")
            
            # Replace bottom half with mutations of top half (one-to-one mutation)
            for i in range(n_top):
                source_idx = top_indices[i]  # Index of model to mutate
                target_idx = bottom_indices[i]  # Index where to store mutation
                
                # Get source model and its params
                source_model = self.models[source_idx]
                source_params = self.model_params[source_idx]
                
                # Create mutation and store it
                new_model, new_params = self.mutate(source_model, source_params)
                self.models[target_idx] = new_model
                self.model_params[target_idx] = new_params
                self.scores[target_idx] = PostProcessor()
                
                if self.verbose:
                    print(f"Mutated {type(source_model).__name__} (score: {scores[source_idx]:.4f}) -> position {target_idx}")
                    print(f"Old params: {source_params}")
                    print(f"New params: {new_params}")
            
            # Diversity check: if more than 70% of models are of the same type
            model_types = [type(m).__name__ for m in self.models]
            from collections import Counter
            type_counts = Counter(model_types)

            for model_type, count in type_counts.items():
                if count / n_models > 0.7:
                    if self.verbose:
                        print(f"Diversity alert: {model_type} constitutes {count/n_models*100:.2f}% of models.")
                    # Identify indices of this dominant model type
                    dominant_indices = [i for i, t in enumerate(model_types) if t == model_type]
                    
                    # Get scores for these dominant models
                    dominant_scores = [(self.scores[i].get_average(), i) for i in dominant_indices]
                    
                    # Sort them by score (descending) to find the best one
                    dominant_scores.sort(key=lambda x: x[0], reverse=True)
                    
                    # Keep the best one, mark others for replacement
                    indices_to_replace = [idx for score, idx in dominant_scores[1:]] # All except the first (best)
                    
                    for idx_to_replace in indices_to_replace:
                        new_model, new_params = self.random_model_with_params()
                        self.models[idx_to_replace] = new_model
                        self.model_params[idx_to_replace] = new_params
                        self.scores[idx_to_replace] = PostProcessor()
                        if self.verbose:
                            print(f"Replaced model at index {idx_to_replace} with a new random {type(new_model).__name__}")
        
        # Calculate various score aggregations
        current_scores_np = np.array(current_scores)
        # Handle potential division by zero or log(0) in geometric mean
        geometric_mean = np.exp(np.mean(np.log(current_scores_np + 1e-10)))
        
        # Handle potential division by zero in harmonic mean
        with np.errstate(divide='ignore', invalid='ignore'):
            harmonic_mean = len(current_scores) / np.sum(1.0 / (current_scores_np + 1e-10))
        
        # Return dictionary with different aggregations
        return {
            'geometric_mean': float(geometric_mean),
            'harmonic_mean': float(harmonic_mean),
            'median': float(np.median(current_scores_np)),
            'mean': float(np.mean(current_scores_np)),
            'max': float(np.max(current_scores_np)),
            'min': float(np.min(current_scores_np)),
            #'exploration_std': float(self.exploration_std) if self.step % 1000 == 0 else None
        }