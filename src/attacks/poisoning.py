"""
================================================================================
DATA POISONING ATTACKS
================================================================================
Attacks that corrupt training / calibration data.
================================================================================
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from abc import ABC, abstractmethod

@dataclass
class PoisonConfig:
    fraction: float = 0.05
    random_state: int = 42
    target_class: int = 0
    trigger_value: float = 999.0
    trigger_features: List[int] = field(default_factory=list)

class BasePoisoning(ABC):
    def __init__(self, config: PoisonConfig | None = None):
        self.config = config or PoisonConfig()
        self.rng = np.random.RandomState(self.config.random_state)

    @abstractmethod
    def poison(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

    def _select_indices(self, n: int) -> np.ndarray:
        k = int(n * self.config.fraction)
        return self.rng.choice(n, size=k, replace=False)

class LabelFlipPoisoning(BasePoisoning):
    """Random / targeted / boundary-proximate label flipping."""
    def __init__(self, config=None, *, mode: str = "random"):
        super().__init__(config)
        self.mode = mode

    def poison(self, X, y):
        X_o, y_o = X.copy(), y.copy()
        if self.mode == "targeted":
            t_idx = np.where(y == self.config.target_class)[0]
            k = min(int(len(t_idx) * self.config.fraction), len(t_idx))
            flip = self.rng.choice(t_idx, size=k, replace=False)
            y_o[flip] = 1 - y_o[flip]
        elif self.mode == "boundary":
            for cls in [0, 1]:
                c_idx = np.where(y == cls)[0]
                o_idx = np.where(y == 1 - cls)[0]
                if len(o_idx) == 0: continue
                centroid = X[o_idx].mean(axis=0)
                dists = np.linalg.norm(X[c_idx] - centroid, axis=1)
                k = int(len(c_idx) * self.config.fraction / 2)
                closest = c_idx[np.argsort(dists)[:k]]
                y_o[closest] = 1 - y_o[closest]
        else:
            idx = self._select_indices(len(y))
            y_o[idx] = 1 - y_o[idx]
        return X_o, y_o

class BackdoorPoisoning(BasePoisoning):
    """Injects trigger pattern + forces target label."""
    def poison(self, X, y):
        X_o, y_o = X.copy(), y.copy()
        idx = self._select_indices(len(y))
        feats = self.config.trigger_features or [0, 1]
        for f in feats:
            if f < X.shape[1]:
                X_o[idx, f] = self.config.trigger_value
        y_o[idx] = self.config.target_class
        return X_o, y_o

    def apply_trigger(self, X):
        X_t = X.copy()
        feats = self.config.trigger_features or [0, 1]
        for f in feats:
            if f < X.shape[1]:
                X_t[:, f] = self.config.trigger_value
        return X_t

class CleanLabelPoisoning(BasePoisoning):
    """Moves target-class features towards source centroid without label change."""
    def __init__(self, config=None, *, perturbation_strength: float = 0.5):
        super().__init__(config)
        self.strength = perturbation_strength

    def poison(self, X, y):
        X_o, y_o = X.copy(), y.copy()
        tc, sc = self.config.target_class, 1 - self.config.target_class
        t_idx = np.where(y == tc)[0]
        s_idx = np.where(y == sc)[0]
        if len(s_idx) == 0 or len(t_idx) == 0:
            return X_o, y_o
        centroid = X[s_idx].mean(axis=0)
        dists = np.linalg.norm(X[t_idx] - centroid, axis=1)
        k = min(int(len(t_idx) * self.config.fraction), len(t_idx))
        nearest = t_idx[np.argsort(dists)[:k]]
        for i in nearest:
            X_o[i] += self.strength * (centroid - X_o[i])
        return X_o, y_o

class CalibrationPoisoning(BasePoisoning):
    """Targets conformal prediction calibration (inflate/deflate q_hat)."""
    def __init__(self, config=None, *, mode: str = "inflate", score_delta: float = 0.3):
        super().__init__(config)
        self.mode = mode
        self.score_delta = score_delta

    def poison(self, X, y):
        X_o, y_o = X.copy(), y.copy()
        idx = self._select_indices(len(y))
        if self.mode == "inflate":
            for i in idx:
                X_o[i] += self.rng.randn(X.shape[1]) * self.score_delta
        else:
            centroids = {}
            for c in [0, 1]:
                m = y == c
                if m.any(): centroids[c] = X[m].mean(axis=0)
            for i in idx:
                if y[i] in centroids:
                    X_o[i] += self.score_delta * (centroids[y[i]] - X_o[i])
        return X_o, y_o

    def poison_scores(self, scores):
        s = scores.copy()
        idx = self._select_indices(len(scores))
        if self.mode == "inflate":
            s[idx] += self.score_delta
        else:
            s[idx] = np.maximum(s[idx] - self.score_delta, 0.0)
        return s
