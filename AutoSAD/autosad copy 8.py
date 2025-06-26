"""
AutoSAD with a bandit-based model selector.
Works with pysad-rust ≥ 0.8.  Copy-paste and run.
"""

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

    def __init__(self):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def process(self, score: float) -> float:
        if score < self._min:
            self._min = score
        if score > self._max:
            self._max = score

        self._count += 1
        self._sum += score
        self._sum_sq += score ** 2

        span = self._max - self._min
        return 0.0 if span == 0 else (score - self._min) / span

    # Convenience metrics (optional)
    def mean(self):
        return 0.0 if self._count == 0 else self._sum / self._count

    def variance(self):
        if self._count < 2:
            return 0.0
        mean = self.mean()
        return (self._sum_sq / self._count) - mean**2


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
        acq_strategy: str = "UCB",     # added
        **kwargs,
    ):
        self.n_models = n_models
        self.random_state = random_state
        self.verbose = verbose
        self.acq_strategy = acq_strategy   # added

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
        self.bandit = BanditGate(self.n_models)   # ← bandit layer

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
        # pick arm via selected acquisition (lower is better)
        arm = np.argmin(self.bandit.acq(self.acq_strategy))   # modified
        raw_score = self.models[arm].fit_score_partial(X)
        reward = self.scores[arm].process(raw_score)  # reward == normalised loss
        self.bandit.update(arm, reward)
        self.step += 1

        # evolution every 1000 points
        if self.step % 1000 == 0:
            self._evolve()

        return reward  # could also return min/avg reward

    # ───────────────────── evolution ─────────────────────
    def _evolve(self):
        if self.verbose:
            print(f"\n=== evolve @ step {self.step} ===")
        
        # update exploration-decay
        avg = np.mean([s.mean() for s in self.scores])
        self.exploration_std = np.clip(self.exploration_std * 2 ** (-avg), 0.1, 2.0)

        self.bandit.decay = max(0.01, self.bandit.decay * 2 ** (-avg))

        # rank by selected acquisition
        ranking = np.argsort(self.bandit.acq(self.acq_strategy))   # modified
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

            # print all models and params
            # for i, m in enumerate(self.models):
            #     print(f"model {i}: {type(m).__name__}")
            #     print(f"params: {self.model_params[i]}")
            #     print(f"score: {self.scores[i].mean():.3f}")
            #     print(f"reward: {self.bandit.mean[i]:.3f}")
            #     print(f"pulls: {self.bandit.n[i]}")
            #     print("-" * 40)
