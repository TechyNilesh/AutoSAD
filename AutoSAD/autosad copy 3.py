import numpy as np
import random
from abc import ABC
from sklearn.metrics import roc_auc_score
from collections import deque
from pysad_rust import (
    HalfSpaceTrees,
    IForestASD,
    RobustRandomCutForest,
    LODA,
    OnlineIsolationForest  # Add OIF import
)
from scipy.stats import rankdata

class AutoSAD(ABC):
    def __init__(
        self,
        n_models: int = 10,
        random_seed: int = 42,
        window_size: int = 1000,
        **kwargs
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.n_models = n_models
        self.window_size = window_size
        self.min_window_size = kwargs.get('min_window_size', 100)
        self.score_history = []
        
        # Store original data ranges for normalization
        self.feature_mins = kwargs.get('feature_mins', None)
        self.feature_maxes = kwargs.get('feature_maxes', None)
        
        # Store last n anomaly scores for better evaluation
        self.history_size = 5  # Number of evaluation cycles to store for stability
        self.historical_aucs = []
        
        self.models = []
        self.model_params = []
        self.scores = []
        self.auc_scores = [0.0] * n_models
        self.step = 0
        self._best_model_idx = 0
        
        # Track model performance history for weighted ensemble
        self.model_weights = np.ones(n_models) / n_models
        # Keep track of model types for diversity assessment
        self.model_types = []

        self._init_pool()

    def _init_pool(self):
        """Initialize a diverse pool of anomaly detectors with complementary strengths"""
        types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]

        # 1) one of each type with default params
        defaults = [
            (HalfSpaceTrees, {

                "num_trees": 50, "max_depth": 15, "window_size": 300,
                "feature_mins": self.feature_mins,
                "feature_maxes": self.feature_maxes
            }),
            (IForestASD, {
                "window_size": 768, "n_estimators": 100, "max_samples": 384
            }),
            (RobustRandomCutForest, {
                "num_trees": 100, "tree_size": 384
            }),
            (LODA, {
                "num_bins": 175, "num_random_cuts": 25
            }),
            (OnlineIsolationForest, {
                "num_trees": 100,
                "window_size": 1536,
                "max_leaf_samples": 48,
                "growth_criterion": "adaptive",
                "subsample": 0.9,
                "branching_factor": 3,
                "split": "axisparallel",
                "random_state": 42
            })
        ]

        for cls, params in defaults:
            self.models.append(cls(**params))
            self.model_params.append(params.copy())
            self.model_types.append(cls.__name__)

        # 2) fill up to n_models with random variants
        while len(self.models) < self.n_models:
            cls = random.choice(types)
            if cls is HalfSpaceTrees:
                params = {
                    "num_trees": random.randint(25, 100),
                    "max_depth": random.randint(10, 20),
                    "window_size": random.randint(200, 500),
                    "feature_mins": self.feature_mins,
                    "feature_maxes": self.feature_maxes
                }
            elif cls is IForestASD:
                params = {
                    "window_size": random.randint(256, 2048),
                    "n_estimators": random.randint(50, 150),
                    "max_samples": random.randint(256, 512)
                }
            elif cls is RobustRandomCutForest:
                params = {
                    "num_trees": random.randint(50, 200),
                    "tree_size": random.randint(256, 1024)
                }
            elif cls is OnlineIsolationForest:
                params = {
                    "num_trees": random.randint(50, 150),
                    "window_size": random.randint(512, 2048),
                    "max_leaf_samples": random.randint(32, 64),
                    "growth_criterion": random.choice(["fixed", "adaptive"]), 
                    "subsample": random.uniform(0.8, 1.0),
                    "branching_factor": random.choice([2, 3, 4]),
                    "split": random.choice(["axisparallel", "random"]),
                    "random_state": random.randint(0, 1000)
                }
            else:  # LODA
                params = {
                    "num_bins": random.randint(100, 250),
                    "num_random_cuts": random.randint(15, 75)
                }
            self.models.append(cls(**params))
            self.model_params.append(params.copy())
            self.model_types.append(cls.__name__)

        # 3) Prepare sliding windows for each model's scores
        self.scores = [
            deque(maxlen=self.window_size)
            for _ in range(self.n_models)
        ]
        # Pick an initial best model at random
        self._best_model_idx = random.randrange(self.n_models)

    def mutate(self, model, params):
        """Mutate hyperparams around current settings."""
        new_params = params.copy()

        def sample_norm(val, lo, hi, ratio=0.1):
            std = (hi - lo) * ratio
            return int(np.clip(np.random.normal(val, std), lo, hi))

        if isinstance(model, HalfSpaceTrees):
            new_params["num_trees"] = sample_norm(params["num_trees"], 25, 100)
            new_params["max_depth"] = sample_norm(params["max_depth"], 10, 20)
            new_params["window_size"] = sample_norm(params["window_size"], 200, 500)
            return HalfSpaceTrees(**new_params), new_params

        if isinstance(model, IForestASD):
            new_params["window_size"] = sample_norm(params["window_size"], 256, 2048)
            new_params["n_estimators"] = sample_norm(params["n_estimators"], 50, 150)
            new_params["max_samples"] = sample_norm(params["max_samples"], 256, 512)
            return IForestASD(**new_params), new_params

        if isinstance(model, RobustRandomCutForest):
            new_params["num_trees"] = sample_norm(params["num_trees"], 50, 200)
            new_params["tree_size"] = sample_norm(params["tree_size"], 256, 1024)
            return RobustRandomCutForest(**new_params), new_params

        if isinstance(model, OnlineIsolationForest):
            new_params["num_trees"] = sample_norm(params["num_trees"], 50, 150)
            new_params["window_size"] = sample_norm(params["window_size"], 512, 2048)
            new_params["max_leaf_samples"] = sample_norm(params["max_leaf_samples"], 32, 64)
            # For categorical parameters, occasionally change them
            if random.random() < 0.2:
                new_params["growth_criterion"] = random.choice(["fixed", "adaptive"])
            if random.random() < 0.2:
                new_params["subsample"] = round(random.uniform(0.8, 1.0), 2)
            if random.random() < 0.2:
                new_params["branching_factor"] = random.choice([2, 3, 4])
            if random.random() < 0.2:
                new_params["split"] = random.choice(["axisparallel", "random"])
            # Generate a new random state
            new_params["random_state"] = random.randint(0, 1000)
            return OnlineIsolationForest(**new_params), new_params

        # LODA
        new_params["num_bins"] = sample_norm(params["num_bins"], 100, 250)
        new_params["num_random_cuts"] = sample_norm(params["num_random_cuts"], 15, 75)
        return LODA(**new_params), new_params

    def fit_score_partial(self, X):
        """
        Simplified window-based evolution:
        - Collect raw anomaly scores
        - When window is full, mutate all models
        - Return the raw score from the current best model
        """
        current_scores = []
        for i, model in enumerate(self.models):
            raw = model.fit_score_partial(X)
            try:
                s = float(raw)
            except:
                s = 0.0
            current_scores.append(s)
            self.scores[i].append(s)

        self.step += 1
        # Perform mutation when window is full
        if self.step % self.window_size == 0:
            if all(len(w) >= self.window_size for w in self.scores):
                # Keep the best performing model
                losses = []
                for w in self.scores:
                    arr = np.array(w)
                    losses.append(np.std(arr))  # Use standard deviation as loss
                
                best_idx = int(np.argmin(losses))
                # Mutate all models except the best one
                for idx in range(self.n_models):
                    if idx != best_idx:
                        m_new, params_new = self.mutate(
                            self.models[best_idx],
                            self.model_params[best_idx]
                        )
                        self.models[idx] = m_new
                        self.model_params[idx] = params_new
                        self.scores[idx].clear()
                self._best_model_idx = best_idx

        return current_scores[self._best_model_idx]