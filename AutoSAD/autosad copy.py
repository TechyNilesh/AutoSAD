import concurrent
import numpy as np
from pysad.core.base_model import BaseModel
from sklearn.metrics import roc_auc_score
import os
from pysad_rust import HalfSpaceTrees, IForestASD, RobustRandomCutForest, OnlineIsolationForest, LODA


class AutoSAD(BaseModel):
    def __init__(
        self, random_seed=1, window_size=1000, n_models=5, contamination_factors=None, **kwargs
    ):
        """AutoSAD: Automatic model selection for streaming anomaly detection.
        
        Args:
            schema: Optional metadata about the input data (default=None).
            random_seed (int): Random seed for reproducibility (default=1).
            window_size (int): Number of instances between model evaluations (default=2000).
            n_models (int): Number of candidate models to evaluate (default=2).
            contamination_factors (list): List of contamination factors to try (default=None).
            feature_mins (list): Minimum values for each feature (default=None).
            feature_maxes (list): Maximum values for each feature (default=None).
            max_workers (int): Maximum number of processes to use for parallel processing (default=None).
        """
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        self.window_size = window_size
        self.n_models = n_models
        self.selected_model = None
        self.feature_mins = kwargs.get('feature_mins')
        self.feature_maxes = kwargs.get('feature_maxes')
        self.candidate_models = self._initialize_models(first_run=True)
        self.window_scores = {model: np.zeros(window_size) for model in self.candidate_models}
        self.score_indices = {model: 0 for model in self.candidate_models}
        self.contamination_factors = contamination_factors if contamination_factors else [0.05, 0.2]
        self.instance_count = 0
        self.max_workers = os.cpu_count()
        self.model_performance = {model: [] for model in self.candidate_models}

    def _random_hyperparameters(self, model_type):
        """Generate random hyperparameters for a given model type."""
        if model_type == "HalfSpaceTrees":
            return {
                "num_trees": np.random.choice([10, 25]),
                "max_depth": np.random.choice([10, 15]),
                "window_size": np.random.choice([50, 100]),
                "feature_mins": self.feature_mins,
                "feature_maxes": self.feature_maxes,
            }
        elif model_type == "RobustRandomCutForest":
            return {
                "num_trees": np.random.choice([8, 64]),
                "tree_size": np.random.choice([50, 500]),
            }
        elif model_type == "OnlineIsolationForest":
            return {
                "num_trees": np.random.choice([10, 20, 30]),
                "window_size": np.random.choice([512, 2048]),
                "max_leaf_samples": np.random.choice([32, 64, 128]),
                "growth_criterion": np.random.choice(["depth", "leaves"]),
                "subsample": np.random.choice([256, 512, 1024]),
                "branching_factor": np.random.choice([2, 4, 8]),
                "split": np.random.choice(["random", "best"]),
                "n_jobs": 1,
            }
        elif model_type == "LODA":
            return {
                "num_bins": np.random.choice([50, 150, 100]),
                "num_random_cuts": np.random.choice([5, 10]),
                "seed": self.random_seed,
            }

    def _get_model_name(self, model):
        """Get the name of a model for display purposes."""
        return model.__class__.__name__

    def _initialize_models(self, first_run=False):
        """Initialize candidate models for evaluation."""
        if first_run or not hasattr(self, 'best_model'):
            model_types = ["HalfSpaceTrees", "IForestASD", "RobustRandomCutForest", "OnlineIsolationForest", "LODA"]
            models = []
            for _ in range(self.n_models):
                model_type = np.random.choice(model_types)
                params = self._random_hyperparameters(model_type)
                if model_type == "HalfSpaceTrees":
                    models.append(HalfSpaceTrees(**params))
                elif model_type == "RobustRandomCutForest":
                    models.append(RobustRandomCutForest(**params))
                elif model_type == "OnlineIsolationForest":
                    models.append(OnlineIsolationForest(**params))
                elif model_type == "LODA":
                    models.append(LODA(**params))
            return models
        else:
            model_types = ["HalfSpaceTrees", "IForestASD", "RobustRandomCutForest", "OnlineIsolationForest", "LODA"]
            models = [self.best_model]
            for _ in range(self.n_models - 1):
                model_type = np.random.choice(model_types)
                params = self._random_hyperparameters(model_type)
                if model_type == "HalfSpaceTrees":
                    models.append(HalfSpaceTrees(**params))
                elif model_type == "RobustRandomCutForest":
                    models.append(RobustRandomCutForest(**params))
                elif model_type == "OnlineIsolationForest":
                    models.append(OnlineIsolationForest(**params))
                elif model_type == "LODA":
                    models.append(LODA(**params))
            return models

    def _process_model_score(self, model, X):
        """Process a single model's score for an instance."""
        score = model.fit_score_partial(X)
        return model, float(score) if score is not None and not np.isnan(score) else 0.5

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: Returns the self.
        """
        # Sequential processing of models
        for model in self.candidate_models:
            model_, score = self._process_model_score(model, X)
            idx = self.score_indices[model_]
            self.window_scores[model_][idx] = score
            self.score_indices[model_] = idx + 1

        self.instance_count += 1
        if self.instance_count % self.window_size == 0:
            print(f"Window {self.instance_count // self.window_size}: Scores collected - {self.window_size} instances")
            self._n1_experts_evaluation()
            self.candidate_models = self._initialize_models(first_run=False)
            self.window_scores = {model: np.zeros(self.window_size) for model in self.candidate_models}
            self.score_indices = {model: 0 for model in self.candidate_models}
            print(f"Window {self.instance_count // self.window_size}: Model selection and reset completed.")

        return self

    def _pseudo_label(self, scores, contamination_factor):
        """Generate pseudo-labels based on anomaly scores and contamination factor."""
        n_samples = len(scores)
        if n_samples == 0:
            return []
        n_anomalies = max(1, int(n_samples * contamination_factor))
        threshold = np.sort(scores)[-n_anomalies] if n_anomalies > 0 else np.inf
        labels = [1 if score >= threshold else 0 for score in scores]
        print(f"Pseudo-labels: {sum(labels)} anomalies out of {n_samples}")
        return labels

    def _evaluate_single_model(self, m, other_models):
        """Evaluate a single model against other models."""
        scores_m = self.window_scores[m][:self.score_indices[m]]
        if len(scores_m) == 0:
            return m, 0.5
        
        model_aucs = []
        for m_prime in other_models:
            scores_m_prime = self.window_scores[m_prime][:self.score_indices[m_prime]]
            if len(scores_m_prime) == 0:
                continue
            for c in self.contamination_factors:
                pseudo_labels = self._pseudo_label(scores_m_prime, c)
                if len(set(pseudo_labels)) < 2:
                    model_aucs.append(0.5)
                    print(f"Warning: Uniform pseudo-labels for {m_prime} at c={c}")
                else:
                    try:
                        auc = roc_auc_score(pseudo_labels, scores_m)
                        model_aucs.append(auc)
                    except ValueError as e:
                        model_aucs.append(0.5)
                        print(f"AUC error for {m} vs {m_prime}: {e}")
        
        avg_score = np.mean(model_aucs) if model_aucs else 0.5
        if np.isnan(avg_score):
            avg_score = 0.5
        return m, avg_score

    def _n1_experts_evaluation(self):
        """Evaluate models using N-1 experts methodology sequentially."""
        expert_scores = {}

        for m in self.candidate_models:
            other_models = [x for x in self.candidate_models if x != m]
            model, score = self._evaluate_single_model(m, other_models)
            expert_scores[model] = score
            self.model_performance.setdefault(model, []).append(score)

            # Early pruning: remove models with avg AUC < 0.6 over last 3 windows
            if len(self.model_performance[model]) > 3 and np.mean(self.model_performance[model][-3:]) < 0.6:
                self.candidate_models.remove(model)
                del self.window_scores[model]
                del self.score_indices[model]
                del self.model_performance[model]
                print(f"Pruned model: {self._get_model_name(model)}")

        if expert_scores:
            self.selected_model = max(expert_scores, key=expert_scores.get)
            self.best_model = self.selected_model
            print(f"Selected Model: {self._get_model_name(self.selected_model)}, Expert Scores: {dict((self._get_model_name(model), score) for model, score in expert_scores.items())}")

        # Add new models if needed to maintain n_models
        while len(self.candidate_models) < self.n_models:
            new_models = self._initialize_models(first_run=True)[:self.n_models - len(self.candidate_models)]
            self.candidate_models.extend(new_models)
            self.window_scores.update({m: np.zeros(self.window_size) for m in new_models})
            self.score_indices.update({m: 0 for m in new_models})
            self.model_performance.update({m: [] for m in new_models})

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. 
            Higher scores represent more anomalous instances whereas lower scores 
            correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.selected_model:
            return self.selected_model.score_partial(X)
        # If no model is selected yet, use the average score from all candidate models
        if self.candidate_models:
            scores = []
            for model in self.candidate_models:
                score = model.score_partial(X)
                if score is not None and not np.isnan(score):
                    scores.append(score)
            if scores:
                return np.mean(scores)
        return np.random.uniform(0, 1)
    

import numpy as np
import time
from pysad.models import HalfSpaceTrees, IForestASD, RobustRandomCutForest

class CountMinSketch:
    def __init__(self, width=1000, depth=5, seed=1):
        np.random.seed(seed)
        self.width = width
        self.depth = depth
        self.hash_a = np.random.randint(1, width, size=depth)
        self.hash_b = np.random.randint(0, width, size=depth)
        self.table = np.zeros((depth, width), dtype=int)

    def _hash(self, key):
        return [(self.hash_a[i] * key + self.hash_b[i]) % self.width for i in range(self.depth)]

    def update(self, key, count=1):
        bins = self._hash(key)
        for i, idx in enumerate(bins):
            self.table[i, idx] += count

    def estimate(self, key):
        bins = self._hash(key)
        return min(self.table[i, idx] for i, idx in enumerate(bins))

class FlashSelect(BaseModel):
    def __init__(self, n_models=5, window_size=1000, decay_rate=0.9, cms_width=1000, cms_depth=5, max_workers=None, **kwargs):
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.max_workers = max_workers
        
        self.feature_mins = kwargs.get('feature_mins')
        self.feature_maxes = kwargs.get('feature_maxes')
        
        self.models = [
            IForestASD(), 
            RobustRandomCutForest(),
            HalfSpaceTrees(feature_mins=self.feature_mins, feature_maxes=self.feature_maxes)
        ]
        
        while len(self.models) < n_models:
            model_class = np.random.choice([HalfSpaceTrees, IForestASD, RobustRandomCutForest])
            if model_class == HalfSpaceTrees:
                model = HalfSpaceTrees(feature_mins=self.feature_mins, feature_maxes=self.feature_maxes)
            else:
                model = model_class()
            self.models.append(model)
            
        self.current_model = self.models[0]
        self.cms = CountMinSketch(width=cms_width, depth=cms_depth)
        self.counter = 0

    def _process_model_score(self, model_idx, model, X):
        score = model.fit_score_partial(X)
        if score is None or np.isnan(score):
            score = 0.5
        return model_idx, model, float(score)

    def fit_partial(self, X, y=None):
        scores = []

        # Sequential processing of models
        for i, m in enumerate(self.models):
            model_idx, model, score = self._process_model_score(i, m, X)
            scores.append(score)
            bucket = int(score * 100)
            self.cms.update(hash((model_idx, bucket)))

        self.counter += 1
        if self.counter % self.window_size == 0:
            votes = []
            for i in range(len(self.models)):
                total = sum(
                    self.cms.estimate(hash((i, b))) for b in range(101)
                )
                votes.append(total * (self.decay_rate ** (self.counter // self.window_size - 1)))
            best_idx = int(np.argmax(votes))
            self.current_model = self.models[best_idx]
            print(f"[Window {self.counter // self.window_size}] Selected model: {self.current_model.__class__.__name__}")
        return self

    def fit_score_partial(self, X, y=None):
        self.fit_partial(X, y)
        return self.score_partial(X)

    def score_partial(self, X):
        return self.current_model.score_partial(X)