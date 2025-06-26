from pysad.utils import Window
import numpy as np
from scipy.stats import entropy, kendalltau
from sklearn.metrics.pairwise import linear_kernel
from collections import deque
from pysad_rust import (
    HalfSpaceTrees,
    IForestASD,
    RobustRandomCutForest,
    LODA
)

class AutoSAD():
    def __init__(self,
                 window_size=1000,
                 ensemble_size=5,
                 init_budget=10,
                 alpha=0.5,
                 lam=0.1,
                 random_seed=52, **kwargs):
        
        np.random.seed(random_seed)
        self.window = Window(window_size)
        self.window_size = window_size
        self.window_full = False
        self.ensemble = []
        self.candidate_pool = []
        self.history = deque(maxlen=10)
        self.ensemble_size = ensemble_size
        
        self.feature_mins = kwargs.get('feature_mins', None)
        self.feature_maxes = kwargs.get('feature_maxes', None)
        
        # Model types
        self.model_types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA]
        self.model_params = []
        
        # Adaptive parameters
        self.alpha = alpha
        self.lam = lam
        self.budget = init_budget
        
        # Metrics tracker
        self.metrics = {
            'entropy': [],
            'consistency': [],
            'agreement': []
        }
        
        self._init_pool()

    def _init_pool(self):
        """Initialize model pool with diverse architectures"""
        # Initialize with one of each type
        params = {
            "num_trees": 32, 
            "max_depth": 15, 
            "window_size": 250,
            "feature_mins": self.feature_mins,
            "feature_maxes": self.feature_maxes
        }
        self.candidate_pool.append(HalfSpaceTrees(**params))
        self.model_params.append(params)

        params = {
            "window_size": 512, "n_estimators": 32, "max_samples": 256
        }
        self.candidate_pool.append(IForestASD(**params))
        self.model_params.append(params)

        params = {
            "num_trees": 32, "tree_size": 256
        }
        self.candidate_pool.append(RobustRandomCutForest(**params))
        self.model_params.append(params)

        params = {
            "num_bins": 100, "num_random_cuts": 10
        }
        self.candidate_pool.append(LODA(**params))
        self.model_params.append(params)

        # Add initial model to ensemble
        self._update_ensemble(self.candidate_pool[0])

    def _unsupervised_score(self, model):
        """Composite metric for model ranking"""
        window_data = self._convert_to_2d_array(self.window.get())
        scores = model.score(window_data)
        
        # 1. Score distribution quality
        hist = np.histogram(scores, bins=50)[0]
        ent = entropy(hist + 1e-9)  # Avoid zero-division
        
        # 2. Temporal consistency
        consistency = np.mean([np.std(scores[i:i+100]) 
                             for i in range(0, len(scores), 100)])
        
        # 3. Ensemble agreement
        if len(self.ensemble) > 1:
            tau = np.mean([kendalltau(scores, m.score(window_data))[0]
                         for m in self.ensemble])
        else:
            tau = 0.5
            
        return 0.4*(1/(1+ent)) + 0.3*(1/(1+consistency)) + 0.3*abs(tau)

    def _detect_drift(self):
        """Hybrid drift detection using scores and representations"""
        # Score-based detection
        window_data = self._convert_to_2d_array(self.window.get())
        curr_scores = [m.score(window_data) for m in self.ensemble]
        prev_scores = [h['scores'] for h in self.history]
        
        mean_drift = np.mean([self._hoeffding_test(p, c) 
                            for p,c in zip(prev_scores, curr_scores)])
        
        # Representation-based validation
        if mean_drift < 0.95:
            z_old = self.ensemble[0].get_latent(self._convert_to_2d_array(self.history[0]['data']))
            z_new = self.ensemble[0].get_latent(window_data)
            return linear_CKA(z_old, z_new) < 0.8
            
        return False

    def _hoeffding_test(self, scores1, scores2):
        """Hoeffding-based reliability estimation"""
        mu1, mu2 = np.mean(scores1), np.mean(scores2)
        s_range = max(np.max(scores2) - np.min(scores1), 1e-5)
        return np.exp(-2*(mu1-mu2)**2/(s_range**2*(1/len(scores1))))

    def _update_ensemble(self, model):
        """Diversity-preserving ensemble management"""
        if len(self.ensemble) < 5:
            self.ensemble.append(model)
        else:
            # Replace least diverse member
            window_data = self._convert_to_2d_array(self.window.get())
            similarities = [linear_CKA(model.get_latent(window_data),
                                      m.get_latent(window_data))
                           for m in self.ensemble]
            self.ensemble.pop(np.argmax(similarities))
            self.ensemble.append(model)

    def _generate_candidates(self, best_model):
        """Generate new model candidates using normal distribution"""
        new_params = {}
        
        if isinstance(best_model, HalfSpaceTrees):
            new_params = {
                "num_trees": int(np.clip(np.random.normal(32, 5), 5, 50)),
                "max_depth": int(np.clip(np.random.normal(15, 3), 5, 25)),
                "window_size": int(np.clip(np.random.normal(250, 50), 50, 500)),
                "feature_mins": self.feature_mins,
                "feature_maxes": self.feature_maxes
            }
            return HalfSpaceTrees(**new_params)
            
        elif isinstance(best_model, IForestASD):
            new_params = {
                "window_size": int(np.clip(np.random.normal(512, 100), 128, 1024)),
                "n_estimators": int(np.clip(np.random.normal(32, 8), 16, 128)),
                "max_samples": int(np.clip(np.random.normal(256, 32), 64, 512))
            }
            return IForestASD(**new_params)
            
        elif isinstance(best_model, RobustRandomCutForest):
            new_params = {
                "num_trees": int(np.clip(np.random.normal(32, 8), 16, 128)),
                "tree_size": int(np.clip(np.random.normal(256, 32), 64, 512))
            }
            return RobustRandomCutForest(**new_params)
            
        else:  # LODA
            new_params = {
                "num_bins": int(np.clip(np.random.normal(100, 20), 50, 200)),
                "num_random_cuts": int(np.clip(np.random.normal(10, 3), 5, 50))
            }
            return LODA(**new_params)

    def _convert_to_2d_array(self, X):
        """Convert input to 2D numpy array of float64"""
        if isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim > 2:
            raise ValueError("Input array should be 1D or 2D")
            
        return X

    def fit_partial(self, X, y=None):
        X = self._convert_to_2d_array(X)
        self.window.update(X)
        self.window_full = len(self.window.get()) >= self.window_size
        
        if self.window_full:
            if self._detect_drift():
                # Generate new candidates
                best_model = max(self.ensemble, 
                               key=lambda m: self._unsupervised_score(m))
                new_models = [self._generate_candidates(best_model)
                             for _ in range(self.budget//2)]
                
                # Update pool and ensemble
                self.candidate_pool.extend(new_models)
                self._update_ensemble(max(new_models, 
                                        key=self._unsupervised_score))
                
                # Adjust search parameters
                self.alpha = min(0.9, self.alpha*1.1)
                self.lam = max(0.01, self.lam*0.9)
            else:
                # Fine-tune existing models
                for model in self.ensemble:
                    model.fit_partial(X)
                
                # Adjust search parameters
                self.alpha = max(0.1, self.alpha*0.9)
                self.lam = min(1.0, self.lam*1.1)
            
            # Store history
            self.history.append({
                'scores': [m.score_partial(self._convert_to_2d_array(X)) for m in self.ensemble],
                'data': X.copy()
            })
            
        return self

    def score_partial(self, X):
        if not self.ensemble:
            return 0.0
        
        X = self._convert_to_2d_array(X)
        
        # Weighted ensemble scoring
        weights = [self._unsupervised_score(m) for m in self.ensemble]
        weights = np.array(weights) / sum(weights)
        
        # Ensure each model receives properly converted data
        scores = [m.score_partial(X) for m in self.ensemble]
        return np.dot(weights, scores)

    def fit_score_partial(self, X, y=None):
        """Combines fit_partial and score_partial into a single method.
        
        Args:
            X: Input samples
            y: Not used, present for API consistency
            
        Returns:
            float: Anomaly score for the input sample
        """
        self.fit_partial(X)
        score = self.score_partial(X)
        return score

# Utility Functions
def linear_CKA(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)
    return np.trace(YTX.dot(X.T.dot(Y))) / np.sqrt(
        np.trace(XTX.dot(XTX)) * np.trace(YTY.dot(YTY)))