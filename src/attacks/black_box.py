"""
================================================================================
BLACK-BOX ADVERSARIAL ATTACKS
================================================================================
Decision-based and transfer-based attacks requiring only query access.
Implements Boundary Attack, HopSkipJump, and Transfer Attack.
================================================================================
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod

from src.attacks.white_box import AttackConfig, BaseAttack


# ═══════════════════════════════════════════════════════════════
# BOUNDARY ATTACK — Brendel et al. 2018
# ═══════════════════════════════════════════════════════════════

class BoundaryAttack(BaseAttack):
    """
    Decision-based Boundary Attack.

    Starts from a misclassified point and walks along the decision
    boundary towards the original, minimising L2 distance while
    maintaining misclassification.
    """

    def __init__(
        self,
        config: AttackConfig | None = None,
        *,
        n_init_samples: int = 100,
        step_adapt: float = 0.5,
        delta_init: float = 0.1,
        epsilon_init: float = 0.1,
    ):
        super().__init__(config)
        self.n_init_samples = n_init_samples
        self.step_adapt = step_adapt
        self.delta = delta_init
        self.eps_step = epsilon_init

    def generate(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        X_adv = X.copy()
        for i in range(len(X)):
            X_adv[i] = self._attack_single(model, X[i], y[i])
        return X_adv.astype(np.float32)

    def _attack_single(
        self, model, x: np.ndarray, y_true: int
    ) -> np.ndarray:
        cfg = self.config
        dim = x.shape[0]

        # ── Initialise: find a misclassified starting point ─────
        x_adv = self._find_adversarial_start(model, x, y_true)
        if x_adv is None:
            return x  # couldn't find an adversarial start

        # ── Walk towards original while staying adversarial ─────
        for step in range(cfg.max_iter):
            # Orthogonal perturbation (stay on boundary sphere)
            perturbation = np.random.randn(dim)
            perturbation -= np.dot(perturbation, x_adv - x) / (
                np.linalg.norm(x_adv - x) ** 2 + 1e-12
            ) * (x_adv - x)
            perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-12)

            d = np.linalg.norm(x_adv - x)
            perturbation = perturbation * d * self.delta

            # Step towards original
            direction = x - x_adv
            direction = direction / (np.linalg.norm(direction) + 1e-12)

            candidate = x_adv + perturbation + self.eps_step * d * direction
            candidate = np.clip(candidate, cfg.clip_min, cfg.clip_max)

            # Accept if still adversarial and closer
            pred = int(model.predict_proba(candidate.reshape(1, -1))[0, 1] > 0.5)
            if pred != y_true:
                new_dist = np.linalg.norm(candidate - x)
                if new_dist < np.linalg.norm(x_adv - x):
                    x_adv = candidate

            # Adaptive step size
            ratio = np.linalg.norm(x_adv - x) / (d + 1e-12)
            if ratio < 1.0:
                self.delta *= (1.0 + self.step_adapt)
                self.eps_step *= (1.0 + self.step_adapt)
            else:
                self.delta *= (1.0 - self.step_adapt)
                self.eps_step *= (1.0 - self.step_adapt)

        return self._apply_feature_mask(
            x_adv.reshape(1, -1), x.reshape(1, -1)
        ).flatten()

    def _find_adversarial_start(
        self, model, x: np.ndarray, y_true: int
    ) -> Optional[np.ndarray]:
        """Sample random directions until we find a misclassified point."""
        for _ in range(self.n_init_samples):
            noise = np.random.randn(*x.shape) * self.config.epsilon * 5
            candidate = x + noise
            candidate = np.clip(
                candidate, self.config.clip_min, self.config.clip_max
            )
            pred = int(
                model.predict_proba(candidate.reshape(1, -1))[0, 1] > 0.5
            )
            if pred != y_true:
                return candidate
        return None


# ═══════════════════════════════════════════════════════════════
# HOPSIPJUMP — Chen et al. 2020
# ═══════════════════════════════════════════════════════════════

class HopSkipJumpAttack(BaseAttack):
    """
    HopSkipJump Attack.

    Estimates the gradient direction at the decision boundary using
    binary search and Monte Carlo sampling, then steps along it.
    """

    def __init__(
        self,
        config: AttackConfig | None = None,
        *,
        n_gradient_samples: int = 50,
        binary_search_steps: int = 15,
        step_schedule: str = "geometric",
    ):
        super().__init__(config)
        self.n_gradient_samples = n_gradient_samples
        self.binary_search_steps = binary_search_steps
        self.step_schedule = step_schedule

    def generate(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        X_adv = X.copy()
        for i in range(len(X)):
            X_adv[i] = self._attack_single(model, X[i], y[i])
        return X_adv.astype(np.float32)

    def _attack_single(
        self, model, x: np.ndarray, y_true: int
    ) -> np.ndarray:
        cfg = self.config
        dim = x.shape[0]

        # Initialise — find adversarial start
        boundary = BoundaryAttack(cfg)
        x_adv = boundary._find_adversarial_start(model, x, y_true)
        if x_adv is None:
            return x

        for step in range(cfg.max_iter):
            # ── Binary search to find boundary point ────────────
            x_boundary = self._binary_search(model, x, x_adv, y_true)

            # ── Estimate gradient at boundary ───────────────────
            grad_est = self._estimate_gradient(
                model, x_boundary, y_true, dim
            )

            # ── Step along estimated gradient ───────────────────
            if self.step_schedule == "geometric":
                eta = cfg.epsilon / (2.0 ** (step + 1))
            else:
                eta = cfg.step_size

            x_adv = x_boundary + eta * grad_est
            x_adv = np.clip(x_adv, cfg.clip_min, cfg.clip_max)

            # Verify still adversarial
            pred = int(
                model.predict_proba(x_adv.reshape(1, -1))[0, 1] > 0.5
            )
            if pred == y_true:
                x_adv = x_boundary  # revert

        return self._apply_feature_mask(
            x_adv.reshape(1, -1), x.reshape(1, -1)
        ).flatten()

    def _binary_search(
        self, model, x_orig: np.ndarray, x_adv: np.ndarray, y_true: int
    ) -> np.ndarray:
        lo, hi = 0.0, 1.0
        for _ in range(self.binary_search_steps):
            mid = (lo + hi) / 2.0
            x_mid = (1 - mid) * x_orig + mid * x_adv
            pred = int(
                model.predict_proba(x_mid.reshape(1, -1))[0, 1] > 0.5
            )
            if pred != y_true:
                hi = mid  # closer to original
            else:
                lo = mid  # closer to adversarial
        return (1 - hi) * x_orig + hi * x_adv

    def _estimate_gradient(
        self, model, x_boundary: np.ndarray, y_true: int, dim: int
    ) -> np.ndarray:
        """Monte Carlo gradient estimation at boundary."""
        grad = np.zeros(dim)
        delta = np.linalg.norm(x_boundary) * 0.01 + 1e-6

        for _ in range(self.n_gradient_samples):
            noise = np.random.randn(dim)
            noise = noise / (np.linalg.norm(noise) + 1e-12) * delta

            x_plus = x_boundary + noise
            pred = int(
                model.predict_proba(x_plus.reshape(1, -1))[0, 1] > 0.5
            )
            if pred != y_true:
                grad += noise
            else:
                grad -= noise

        grad = grad / (np.linalg.norm(grad) + 1e-12)
        return grad


# ═══════════════════════════════════════════════════════════════
# TRANSFER ATTACK
# ═══════════════════════════════════════════════════════════════

class TransferAttack(BaseAttack):
    """
    Transfer Attack: craft adversarial examples on a surrogate model
    and evaluate transferability to the target.

    Uses a lightweight sklearn surrogate (GradientBoosting) that's
    fast to train and yields transferable perturbations.
    """

    def __init__(
        self,
        config: AttackConfig | None = None,
        *,
        surrogate_type: str = "gradient_boosting",
        n_surrogate_estimators: int = 100,
    ):
        super().__init__(config)
        self.surrogate_type = surrogate_type
        self.n_estimators = n_surrogate_estimators
        self.surrogate = None

    def fit_surrogate(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train surrogate model on available data."""
        from sklearn.ensemble import GradientBoostingClassifier

        self.surrogate = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=4,
            random_state=42,
        )
        self.surrogate.fit(X_train, y_train)

    def generate(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        if self.surrogate is None:
            # Auto-train surrogate using the target's predictions
            y_pseudo = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
            self.fit_surrogate(X, y_pseudo)

        cfg = self.config
        X_adv = X.copy()

        # Use numerical gradient on surrogate
        for _ in range(cfg.max_iter):
            grad = self._numerical_gradient(X_adv, y)

            if cfg.norm == "l_inf":
                X_adv = X_adv + cfg.step_size * np.sign(grad)
            else:
                grad_norm = np.linalg.norm(
                    grad, axis=1, keepdims=True
                ) + 1e-12
                X_adv = X_adv + cfg.step_size * (grad / grad_norm)

            X_adv = self._project(X_adv, X)
            X_adv = self._apply_feature_mask(X_adv, X)

        return X_adv.astype(np.float32)

    def _numerical_gradient(
        self, X: np.ndarray, y: np.ndarray, h: float = 1e-4
    ) -> np.ndarray:
        """Finite-difference gradient on surrogate probabilities."""
        n, d = X.shape
        grad = np.zeros_like(X)

        base_probs = self.surrogate.predict_proba(X)[:, 1]

        for j in range(d):
            X_plus = X.copy()
            X_plus[:, j] += h
            probs_plus = self.surrogate.predict_proba(X_plus)[:, 1]
            grad[:, j] = (probs_plus - base_probs) / h

        # Flip sign for untargeted (maximise loss)
        signs = np.where(y == 1, 1.0, -1.0).reshape(-1, 1)
        return grad * signs
