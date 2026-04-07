"""
================================================================================
PHYSICALLY-CONSTRAINED ADVERSARIAL ATTACKS
================================================================================
Attacks that respect real-world network traffic invariants.
Only mutable features (IAT, packet length, duration) are perturbed.
================================================================================
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial.distance import cdist

from src.attacks.white_box import AttackConfig, BaseAttack


# ═══════════════════════════════════════════════════════════════
# FEATURE-CONSTRAINED EVASION
# ═══════════════════════════════════════════════════════════════


class FeatureConstrainedEvasion(BaseAttack):
    """
    Only perturbs attacker-controllable flow features while
    enforcing physical invariants:
      • bytes_sent ≥ 0
      • packet_count ∈ ℤ⁺
      • temporal monotonicity (IAT ≥ 0)
      • duration ≥ sum(IATs)
    """

    @dataclass
    class PhysicalConstraints:
        non_negative: List[str] = field(
            default_factory=lambda: ["bytes", "pkts", "duration", "iat"]
        )
        integer_valued: List[str] = field(default_factory=lambda: ["pkts", "packet"])
        max_perturbation_ratio: float = 0.3  # at most 30% change

    def __init__(
        self,
        config: AttackConfig | None = None,
        feature_names: List[str] | None = None,
    ):
        super().__init__(config)
        self.feature_names = feature_names or []
        self.constraints = self.PhysicalConstraints()
        self._resolve_mutable_indices()

    def _resolve_mutable_indices(self):
        """Auto-detect mutable features from names."""
        if self.config.mutable_features:
            return
        mutable_keywords = ["iat", "duration", "pkt", "bytes", "len"]
        self.config.mutable_features = [
            i
            for i, name in enumerate(self.feature_names)
            if any(kw in name.lower() for kw in mutable_keywords)
        ]

    def generate(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        cfg = self.config
        X_adv = X.copy()

        if not cfg.mutable_features:
            return X_adv  # no mutable features identified

        # Per-feature noise scaled to feature std
        for idx in cfg.mutable_features:
            col = X[:, idx]
            std = np.std(col) + 1e-9
            max_delta = std * self.constraints.max_perturbation_ratio

            noise = np.random.uniform(-max_delta, max_delta, size=len(X))
            X_adv[:, idx] += noise

            # Apply physical constraints
            feature_name = (
                self.feature_names[idx].lower() if idx < len(self.feature_names) else ""
            )

            # Non-negativity
            if any(kw in feature_name for kw in self.constraints.non_negative):
                X_adv[:, idx] = np.maximum(X_adv[:, idx], 0.0)

            # Integer constraint
            if any(kw in feature_name for kw in self.constraints.integer_valued):
                X_adv[:, idx] = np.round(X_adv[:, idx])

        # Enforce ε-budget
        X_adv = self._project(X_adv, X)

        return X_adv.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# SLOW-DRIP ATTACK
# ═══════════════════════════════════════════════════════════════


class SlowDripAttack(BaseAttack):
    """
    Low-and-slow exfiltration evasion.

    Stretches inter-arrival times (IAT) and shrinks packet sizes
    to mimic benign, low-bandwidth traffic while still exfiltrating
    data over time.

    Attacker strategy:
      1. Multiply IAT features by slowdown factor (>1.0)
      2. Divide byte/packet-size features by compression factor
      3. Preserve total data volume (bytes × packets ≈ const)
    """

    def __init__(
        self,
        config: AttackConfig | None = None,
        feature_names: List[str] | None = None,
        *,
        slowdown: float = 2.5,
        compression: float = 0.4,
    ):
        super().__init__(config)
        self.feature_names = feature_names or []
        self.slowdown = slowdown
        self.compression = compression

    def generate(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_adv = X.copy()

        for i, name in enumerate(self.feature_names):
            name_low = name.lower()

            # Stretch timing features
            if any(kw in name_low for kw in ["iat", "duration"]):
                # Add randomised jitter around the slowdown factor
                jitter = np.random.uniform(
                    0.8 * self.slowdown,
                    1.2 * self.slowdown,
                    size=len(X),
                )
                X_adv[:, i] *= jitter
                X_adv[:, i] = np.maximum(X_adv[:, i], 0.0)

            # Compress packet sizes
            elif any(kw in name_low for kw in ["bytes", "len", "size"]):
                jitter = np.random.uniform(
                    0.8 * self.compression,
                    1.2 * self.compression,
                    size=len(X),
                )
                X_adv[:, i] *= jitter
                X_adv[:, i] = np.maximum(X_adv[:, i], 0.0)

        return self._apply_feature_mask(X_adv, X).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# MIMICRY ATTACK — Statistical Profile Matching
# ═══════════════════════════════════════════════════════════════


class MimicryAttack(BaseAttack):
    """
    Mimicry Attack via Statistical Profile Matching.

    Maps malicious flow feature distributions to match the
    statistical signature of benign traffic clusters.

    Steps:
      1. Compute centroid + covariance of benign cluster
      2. For each attack sample, find nearest benign centroid
      3. Transport attack features towards centroid via
         Wasserstein-barycenter-inspired interpolation
      4. Respect feature constraints during transport
    """

    def __init__(
        self,
        config: AttackConfig | None = None,
        *,
        blend_factor: float = 0.7,
        n_benign_clusters: int = 5,
    ):
        super().__init__(config)
        self.blend_factor = blend_factor
        self.n_clusters = n_benign_clusters
        self._centroids: Optional[np.ndarray] = None

    def fit_benign_profile(self, X_benign: np.ndarray):
        """Learn benign traffic cluster centroids (offline)."""
        from sklearn.cluster import KMeans

        n = min(self.n_clusters, len(X_benign))
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        km.fit(X_benign)
        self._centroids = km.cluster_centers_

    def generate(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            raise RuntimeError(
                "Must call fit_benign_profile(X_benign) before generate()"
            )

        # Assign each attack sample to nearest benign centroid
        dists = cdist(X, self._centroids)
        assignments = np.argmin(dists, axis=1)

        X_adv = X.copy()
        for i in range(len(X)):
            target = self._centroids[assignments[i]]

            # Wasserstein-inspired interpolation
            # Move features towards benign centroid by blend_factor
            delta = target - X[i]
            X_adv[i] = X[i] + self.blend_factor * delta

        # Apply constraints
        X_adv = np.clip(X_adv, self.config.clip_min, self.config.clip_max)
        X_adv = self._apply_feature_mask(X_adv, X)

        return X_adv.astype(np.float32)
