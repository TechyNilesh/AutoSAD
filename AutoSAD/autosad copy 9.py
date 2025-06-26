import numpy as np
import random
from abc import ABC, abstractmethod
from collections import Counter
from scipy.stats import truncnorm
from scipy.stats import norm      # added for EI/PI

from pysad_rust import (
    HalfSpaceTrees,
    IForestASD,
    RobustRandomCutForest,
    LODA,
    OnlineIsolationForest,
)


# ──────────────────────────  helpers  ──────────────────────────
class PostProcessor:
    """
    Tracks min/max/mean/variance and returns a score normalised to [0,1].
    """

    def __init__(self, max_history: int = 1000):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
        self.normalized_scores = []  # Store normalized scores for correlation
        self.max_history = max_history

    def process(self, score: float) -> float:
        if score < self._min:
            self._min = score
        if score > self._max:
            self._max = score

        self._count += 1
        self._sum += score
        self._sum_sq += score ** 2

        span = self._max - self._min
        normalized = 0.0 if span == 0 else (score - self._min) / span
        
        # Store normalized score
        self.normalized_scores.append(normalized)
        if len(self.normalized_scores) > self.max_history:
            self.normalized_scores.pop(0)
            
        return normalized

    # Convenience metrics (optional)
    def mean(self):
        return 0.0 if self._count == 0 else self._sum / self._count

    def variance(self):
        if self._count < 2:
            return 0.0
        mean = self.mean()
        return (self._sum_sq / self._count) - mean**2


class SimpleAnomalyRewardCalculator:
    """
    Simple reward calculator that uses anomaly scores directly.
    Higher anomaly scores indicate better anomaly detection performance.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.recent_scores = []  # List of lists, one per model
        
    def update_scores(self, normalized_scores: list):
        """
        Update normalized scores for all models.
        normalized_scores: list of normalized scores from each model at current time step
        """
        # Initialize if first call
        if not self.recent_scores:
            self.recent_scores = [[] for _ in range(len(normalized_scores))]
            
        # Add scores for each model
        for i, score in enumerate(normalized_scores):
            self.recent_scores[i].append(score)
            # Keep only recent scores
            if len(self.recent_scores[i]) > self.window_size:
                self.recent_scores[i].pop(0)
                
    def calculate_rewards(self) -> np.ndarray:
        """
        Calculate rewards based on recent anomaly scores.
        Returns rewards for each model (higher is better for anomaly detection).
        """
        if not self.recent_scores or len(self.recent_scores[0]) < 10:
            return np.zeros(len(self.recent_scores))
            
        rewards = []
        for model_scores in self.recent_scores:
            if len(model_scores) < 10:
                rewards.append(0.0)
                continue
                
            # Use mean of recent scores as reward
            # Higher anomaly scores are better for anomaly detection
            mean_score = np.mean(model_scores[-min(100, len(model_scores)):])
            rewards.append(mean_score)
                
        return np.array(rewards)
    
class ConsensusRewardCalculator:
    """
    Truly incremental reward calculator based on consensus between models.
    Uses exponentially weighted moving correlations between model outputs.

    Diversity Matters: Consensus works best when your model pool is diverse (different algorithms, parameters, etc.), reducing the risk of all models making the same mistake.
    """
    
    def __init__(self, window_size: int = 1000, decay_factor: float = 0.005):
        self.n_models = None
        self.decay = decay_factor  # Controls how quickly old data is forgotten
        
        # Statistics for incremental correlation calculation
        self.means = None           # Exponentially weighted means
        self.variances = None       # Exponentially weighted variances
        self.covariances = None     # Matrix of covariances between models
        self.corr_matrix = None     # Current correlation matrix
        self.n_updates = 0          # Count of updates processed
        self.rewards = None         # Current reward values
        
    def update_scores(self, normalized_scores: list):
        """
        Update correlation statistics incrementally with new scores.
        """
        scores = np.array(normalized_scores)
        
        # Initialize on first call
        if self.n_models is None:
            self.n_models = len(scores)
            self.means = np.zeros(self.n_models)
            self.variances = np.zeros(self.n_models)
            self.covariances = np.zeros((self.n_models, self.n_models))
            self.corr_matrix = np.zeros((self.n_models, self.n_models))
            self.rewards = np.zeros(self.n_models)
        
        self.n_updates += 1
        
        # Adjust decay for initial values (more aggressive updating at start)
        effective_decay = min(1.0, self.decay * 10) if self.n_updates < 100 else self.decay
        
        # Update means (EWMA)
        delta = scores - self.means
        self.means += effective_decay * delta
        
        # Update variances (EWMV)
        self.variances = (1 - effective_decay) * (self.variances + effective_decay * delta * delta)
        
        # Early exit if we don't have enough data or variance
        if self.n_updates < 10 or np.any(self.variances < 1e-10):
            # Use normalized scores directly as fallback rewards
            self.rewards = scores
            return
            
        # Update covariances between all pairs of models
        for i in range(self.n_models):
            for j in range(i+1, self.n_models):  # Only upper triangular
                delta_j = scores[j] - self.means[j]
                cov_update = delta[i] * delta_j
                self.covariances[i,j] = (1 - effective_decay) * self.covariances[i,j] + effective_decay * cov_update
                self.covariances[j,i] = self.covariances[i,j]  # Symmetric
        
        # Update correlation matrix
        for i in range(self.n_models):
            self.corr_matrix[i,i] = 1.0  # Diagonal is always 1
            std_i = np.sqrt(self.variances[i])
            
            if std_i > 1e-10:  # Avoid division by zero
                for j in range(i+1, self.n_models):
                    std_j = np.sqrt(self.variances[j])
                    
                    if std_j > 1e-10:
                        corr = self.covariances[i,j] / (std_i * std_j)
                        # Ensure correlation is in [-1, 1]
                        corr = max(min(corr, 1.0), -1.0)
                        self.corr_matrix[i,j] = corr
                        self.corr_matrix[j,i] = corr  # Symmetric
        
        # Zero out diagonal for reward calculation (ignore self-correlation)
        reward_matrix = self.corr_matrix.copy()
        np.fill_diagonal(reward_matrix, 0)
        
        # Reward = average correlation with other models
        self.rewards = np.abs(np.mean(reward_matrix, axis=1))
                
    def calculate_rewards(self) -> np.ndarray:
        """Return current rewards without recalculation"""
        if self.rewards is None:
            return np.zeros(self.n_models) if self.n_models else np.array([])
        return self.rewards
    
class MedianCorrelationRewardCalculator:
    """
    Reward calculator based on correlation with median scores.
    Models that correlate more strongly with the median prediction are considered better.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.model_scores = []  # List of lists, one per model
        self.median_scores = []  # Median scores at each timestep
        self.n_models = None
        
    def update_scores(self, normalized_scores: list):
        """
        Update scores for all models and calculate the median score.
        normalized_scores: list of normalized scores from each model at current time step
        """
        # Initialize if first call
        if not self.model_scores:
            self.n_models = len(normalized_scores)
            self.model_scores = [[] for _ in range(self.n_models)]
            
        # Calculate median of all model scores at this timestep
        median_score = np.median(normalized_scores)
        self.median_scores.append(median_score)
        
        # Store scores for each model
        for i, score in enumerate(normalized_scores):
            self.model_scores[i].append(score)
            
        # Keep only recent window of scores
        if len(self.median_scores) > self.window_size:
            self.median_scores.pop(0)
            for i in range(self.n_models):
                self.model_scores[i].pop(0)
                
    def calculate_rewards(self) -> np.ndarray:
        """
        Calculate rewards based on correlation with median scores.
        Returns correlation values for each model (higher is better).
        """
        if not self.model_scores or len(self.model_scores[0]) < 100:  # Need enough data for meaningful correlation
            return np.zeros(self.n_models) if self.n_models else np.array([])
            
        rewards = []
        for i in range(self.n_models):
            # Calculate correlation between model scores and median scores
            if np.std(self.model_scores[i]) > 0 and np.std(self.median_scores) > 0:
                corr = np.corrcoef(self.model_scores[i], self.median_scores)[0, 1]
                # Handle NaN correlation (can happen with constant values)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
                
            rewards.append(abs(corr))
                
        return np.array(rewards)

class BanditGate:
    """
    Simple bandit-based model selector with multiple acquisition functions.
    """

    def __init__(self, n_arms: int, decay: float = 0.20):
        self.n = np.zeros(n_arms, dtype=int)      # pulls per arm
        self.mean = np.zeros(n_arms, dtype=float) # EWMA reward per arm
        self.t = 0
        self.decay = decay

    def update(self, arm_idx: int, reward: float):
        self.t += 1
        self.n[arm_idx] += 1
        # EWMA to forget stale performance
        self.mean[arm_idx] = (1 - self.decay) * reward + self.decay * self.mean[arm_idx]

    def ucb(self):
        """Upper Confidence Bound acquisition function."""
        conf = np.sqrt(2 * np.log(max(1, self.t)) / np.maximum(1, self.n))
        return self.mean + conf      # lower is better

    def ei(self):
        """Expected Improvement acquisition function."""
        conf = np.sqrt(2 * np.log(max(1, self.t)) / np.maximum(1, self.n))
        mu = self.mean
        best = mu.min()
        imp = best - mu
        z = np.zeros_like(mu)
        mask = conf > 0
        z[mask] = imp[mask] / conf[mask]
        return -(imp * norm.cdf(z) + conf * norm.pdf(z))  # negative for minimization

    def pi(self):
        """Probability of Improvement acquisition function."""
        conf = np.sqrt(2 * np.log(max(1, self.t)) / np.maximum(1, self.n))
        mu = self.mean
        best = mu.min()
        imp = best - mu
        z = np.zeros_like(mu)
        mask = conf > 0
        z[mask] = imp[mask] / conf[mask]
        return -norm.cdf(z)  # negative for minimization

    def acq(self, strategy: str = "UCB"):
        """Select appropriate acquisition function based on strategy."""
        strat = strategy.upper()
        if strat == "UCB":
            return self.ucb()
        elif strat == "EI":
            return self.ei()
        elif strat == "PI":
            return self.pi()
        raise ValueError(f"Unknown acquisition strategy: {strategy}")


# ──────────────────────────  main class  ──────────────────────────
class AutoSAD(ABC):
    def __init__(
        self,
        n_models: int = 5,
        random_state: int = 52,
        verbose: bool = False,
        acq_strategy: str = "UCB",
        **kwargs,
    ):
        self.n_models = n_models
        self.random_state = random_state
        self.verbose = verbose
        self.acq_strategy = acq_strategy

        random.seed(random_state)
        np.random.seed(random_state)

        self.feature_mins = kwargs.get("feature_mins")
        self.feature_maxes = kwargs.get("feature_maxes")

        self.models = []
        self.model_params = []
        self.scores = []
        self.step = 0
        self.exploration_std = 1.0

        self._init_pool()
        self.bandit = BanditGate(self.n_models)
        self.reward_calculator = MedianCorrelationRewardCalculator()

    # ───────────────────── hyperparameter grids ─────────────────────
    def _get_hyperparameter_options(self):
        return {
            "HalfSpaceTrees": {
                "num_trees": [8, 16, 32, 64, 128],
                "max_depth": [5, 8, 10, 12, 15],
                "window_size": [50, 100, 150, 200, 250],
            },
            "IForestASD": {
                "window_size": [512, 1024, 2048, 4096],
                "n_estimators": [16, 32, 64, 128],
                "max_samples": [64, 128, 256, 512],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "anomaly_rate_threshold": [0.1, 0.2, 0.3, 0.4],
            },
            "RobustRandomCutForest": {
                "num_trees": [8, 16, 32, 64, 128],
                "tree_size": [64, 128, 256, 512],
                "shingle_size": [1, 2, 3],
            },
            "LODA": {
                "num_bins": [50, 100, 150, 200],
                "num_random_cuts": [5, 10, 25, 50],
            },
            "OnlineIsolationForest": {
                "branching_factor": [2, 3, 4],
                "split": ["axisparallel", "hyperplane"],
                "num_trees": [8, 16, 32, 64, 128],
                "max_leaf_samples": [16, 32, 64],
                "window_size": [512, 1024, 2048, 4096],
                "growth_criterion": ["adaptive", "depth", "size"],
                "subsample": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
        }

    # ───────────────────── weighted HP sampler ─────────────────────
    def _pwhs(self, current, candidates, sigma=1.0):
        if isinstance(current, (int, float)):
            lo, hi = min(candidates), max(candidates)
            std = np.std(candidates) * sigma
            a, b = (lo - current) / std, (hi - current) / std
            try:
                new = truncnorm.rvs(a, b, loc=current, scale=std)
                new = min(candidates, key=lambda x: abs(x - new))
                return int(round(new)) if isinstance(current, int) else new
            except ValueError:
                return np.random.choice(candidates)
        # categorical
        p_keep = max(0.5, 1 - 0.5 * sigma)
        probs = [p_keep if v == current else (1 - p_keep) / (len(candidates) - 1)
                 for v in candidates]
        return np.random.choice(candidates, p=probs)

    # ───────────────────── pool initialisation ─────────────────────
    def random_model_with_params(self):
        types = [HalfSpaceTrees, IForestASD, RobustRandomCutForest, LODA, OnlineIsolationForest]
        cls = np.random.choice(types)
        opts = self._get_hyperparameter_options()[cls.__name__]
        params = {"random_state": self.random_state}
        for p, vals in opts.items():
            if p not in ("feature_mins", "feature_maxes"):
                params[p] = np.random.choice(vals)
        if cls is HalfSpaceTrees:
            params["feature_mins"] = self.feature_mins
            params["feature_maxes"] = self.feature_maxes
        return cls(**params), params

    def _init_pool(self):
        while len(self.models) < self.n_models:
            m, p = self.random_model_with_params()
            self.models.append(m)
            self.model_params.append(p)
            self.scores.append(PostProcessor())

    # ───────────────────── mutation ─────────────────────
    def mutate(self, model, params):
        name = type(model).__name__
        opts = self._get_hyperparameter_options()[name]
        new_params = params.copy()
        new_params["random_state"] = self.random_state
        for p, vals in opts.items():
            if p in params and p not in ("feature_mins", "feature_maxes"):
                new_params[p] = self._pwhs(params[p], vals, self.exploration_std)
        if name == "HalfSpaceTrees":
            new_params["feature_mins"] = self.feature_mins
            new_params["feature_maxes"] = self.feature_maxes
            return HalfSpaceTrees(**new_params), new_params
        elif name == "IForestASD":
            return IForestASD(**new_params), new_params
        elif name == "RobustRandomCutForest":
            return RobustRandomCutForest(**new_params), new_params
        elif name == "LODA":
            return LODA(**new_params), new_params
        elif name == "OnlineIsolationForest":
            return OnlineIsolationForest(**new_params), new_params
        return model, params

    # ───────────────────── online step ─────────────────────
    def fit_score_partial(self, X):
        # Get normalized scores from all models
        normalized_scores = []
        
        for i, model in enumerate(self.models):
            raw_score = model.fit_score_partial(X)
            normalized_score = self.scores[i].process(raw_score)
            normalized_scores.append(normalized_score)
        
        # Update reward calculator with normalized scores
        self.reward_calculator.update_scores(normalized_scores)
        
        # Calculate rewards based on anomaly scores
        rewards = self.reward_calculator.calculate_rewards()
        
        # Update bandit with anomaly score-based rewards
        # Convert to loss for bandit (lower is better for bandit)
        for i, reward in enumerate(rewards):
            loss = 1.0 - reward  # Convert reward to loss
            self.bandit.update(i, loss)
        
        # Select best arm based on acquisition function
        arm = np.argmin(self.bandit.acq(self.acq_strategy))
        
        self.step += 1

        # evolution every 1000 points
        if self.step % 1000 == 0:
            self._evolve()

        return normalized_scores[arm]

    # ───────────────────── evolution ─────────────────────
    def _evolve(self):
        if self.verbose:
            print(f"\n=== evolve @ step {self.step} ===")
            
            # Print reward information
            rewards = self.reward_calculator.calculate_rewards()
            for i, (reward, model) in enumerate(zip(rewards, self.models)):
                print(f"Model {i} ({type(model).__name__}): reward = {reward:.3f}")
        
        # update exploration-decay
        avg = np.mean([s.mean() for s in self.scores])
        self.exploration_std = np.clip(self.exploration_std * 2 ** (-avg), 0.1, 2.0)

        self.bandit.decay = max(0.01, self.bandit.decay * 2 ** (-avg))

        # rank by bandit acquisition function
        ranking = np.argsort(self.bandit.acq(self.acq_strategy))
        n_top = self.n_models // 2
        top_idx = ranking[:n_top]
        bot_idx = ranking[-n_top:]

        # mutate
        for src, tgt in zip(top_idx, bot_idx):
            new_m, new_p = self.mutate(self.models[src], self.model_params[src])
            self.models[tgt] = new_m
            self.model_params[tgt] = new_p
            self.scores[tgt] = PostProcessor()
            self.bandit.mean[tgt] = 0.0
            self.bandit.n[tgt] = 0

        # diversity guard
        types = [type(m).__name__ for m in self.models]
        for t, cnt in Counter(types).items():
            if cnt / self.n_models > 0.7:
                idx_to_replace = [i for i, tt in enumerate(types) if tt == t][1:]
                for idx in idx_to_replace:
                    new_m, new_p = self.random_model_with_params()
                    self.models[idx] = new_m
                    self.model_params[idx] = new_p
                    self.scores[idx] = PostProcessor()
                    self.bandit.mean[idx] = 0.0
                    self.bandit.n[idx] = 0

        if self.verbose:
            print("pool types:", Counter([type(m).__name__ for m in self.models]))
            print(f"exploration σ = {self.exploration_std:.3f}")
            print(f"decay = {self.bandit.decay:.3f}")

            # print best model and params
            print(f"best model: {type(self.models[ranking[0]]).__name__}")
            print(f"params: {self.model_params[ranking[0]]}")
            print(f"score: {self.scores[ranking[0]].mean():.3f}")

            rewards = self.reward_calculator.calculate_rewards()
            if len(rewards) > 0:
                print(f"reward: {rewards[ranking[0]]:.3f}")

            # print all models and params
            # for i, m in enumerate(self.models):
            #     print(f"model {i}: {type(m).__name__}")
            #     print(f"params: {self.model_params[i]}")
            #     print(f"score: {self.scores[i].mean():.3f}")
            #     print(f"reward: {self.bandit.mean[i]:.3f}")
            #     print(f"pulls: {self.bandit.n[i]}")
            #     print("-" * 40)