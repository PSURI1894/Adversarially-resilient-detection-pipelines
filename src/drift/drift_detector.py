"""
================================================================================
CONCEPT DRIFT DETECTION ENGINE
================================================================================
Implements multiple drift detection algorithms with multi-signal consensus:
- ADWIN (Adaptive Windowing) — scanning multiple split points
- Page-Hinkley — sequential change-point detection on error rate
- Kolmogorov-Smirnov (KS) Test — per-feature distributional shift
- Maximum Mean Discrepancy (MMD) — RBF kernel two-sample test

Consensus: retraining triggers only when ≥ 2 detectors agree.
================================================================================
"""

import numpy as np
from scipy import stats
from collections import deque
from typing import List, Dict, Optional


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) for drift detection.
    Detects shifts in the mean of a stream by scanning multiple split points.
    """

    def __init__(self, delta: float = 0.002, window_size: int = 1000):
        self.delta = delta
        self.window_size = window_size
        self.stream: deque = deque(maxlen=window_size)

    def update(self, v: float) -> bool:
        """Add a value and test for drift. Returns True if drift detected."""
        self.stream.append(v)
        if len(self.stream) < 20:
            return False

        data = list(self.stream)
        n = len(data)

        # Scan multiple split points (every 10% of window)
        step = max(1, n // 10)
        for cut in range(step, n - step + 1, step):
            w1 = data[:cut]
            w2 = data[cut:]
            n1, n2 = len(w1), len(w2)
            mu1, mu2 = np.mean(w1), np.mean(w2)

            m = 1.0 / (1.0 / n1 + 1.0 / n2)
            epsilon = np.sqrt(0.5 / m * np.log(4.0 / self.delta))

            if abs(mu1 - mu2) > epsilon:
                # Detected drift — shrink window to post-change portion
                for _ in range(cut):
                    if self.stream:
                        self.stream.popleft()
                return True

        return False

    def reset(self):
        self.stream.clear()


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting change points in a stream.
    Useful for monitoring error rates.
    """

    def __init__(
        self, delta: float = 0.005, lambda_threshold: float = 50, alpha: float = 0.9999
    ):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.sum = 0.0
        self.n = 0

    def update(self, x: float) -> bool:
        """Add a value and test for change. Returns True if change detected."""
        self.n += 1
        self.x_mean += (x - self.x_mean) / self.n
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)

        if self.sum > self.lambda_threshold:
            self.reset()
            return True
        return False

    def reset(self):
        self.sum = 0.0
        self.x_mean = 0.0
        self.n = 0


class KSDetector:
    """
    Kolmogorov-Smirnov test for distributional shift between
    reference and current data (per-feature with Bonferroni correction).
    """

    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        self.reference_data = reference_data
        self.alpha = alpha
        self.n_features = reference_data.shape[1]
        self._last_p_values: Optional[np.ndarray] = None
        self._last_drifted_features: Optional[List[int]] = None

    def detect(self, current_data: np.ndarray) -> bool:
        """
        Test each feature for distributional shift.
        Uses Bonferroni correction to control family-wise error rate.
        Returns True if any feature shows significant drift.
        """
        corrected_alpha = self.alpha / self.n_features
        p_values = np.ones(self.n_features)
        drifted = []

        for i in range(self.n_features):
            stat, p_val = stats.ks_2samp(self.reference_data[:, i], current_data[:, i])
            p_values[i] = p_val
            if p_val < corrected_alpha:
                drifted.append(i)

        self._last_p_values = p_values
        self._last_drifted_features = drifted
        return len(drifted) > 0

    @property
    def drifted_features(self) -> List[int]:
        return self._last_drifted_features or []


class MMDDetector:
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel for two-sample testing.
    Uses the unbiased estimator of MMD².
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        alpha: float = 0.05,
        bandwidth: Optional[float] = None,
    ):
        self.reference_data = reference_data
        self.alpha = alpha

        # Median heuristic for bandwidth if not provided
        if bandwidth is None:
            dists = np.linalg.norm(
                reference_data[: min(500, len(reference_data)), None]
                - reference_data[None, : min(500, len(reference_data))],
                axis=-1,
            )
            self.bandwidth = float(np.median(dists[dists > 0])) + 1e-8
        else:
            self.bandwidth = bandwidth

        # Precompute reference kernel mean
        self._ref_gram_mean = self._kernel_mean(reference_data, reference_data)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF (Gaussian) kernel matrix."""
        sq_X = np.sum(X**2, axis=1, keepdims=True)
        sq_Y = np.sum(Y**2, axis=1, keepdims=True)
        dists_sq = sq_X + sq_Y.T - 2 * X @ Y.T
        return np.exp(-dists_sq / (2 * self.bandwidth**2))

    def _kernel_mean(self, X: np.ndarray, Y: np.ndarray) -> float:
        K = self._rbf_kernel(X, Y)
        n = K.shape[0]
        if X is Y or np.array_equal(X, Y):
            # Unbiased: exclude diagonal
            np.fill_diagonal(K, 0)
            return float(K.sum() / (n * (n - 1))) if n > 1 else 0.0
        return float(K.mean())

    def detect(self, current_data: np.ndarray) -> bool:
        """
        Compute MMD² between reference and current data.
        Uses permutation-based threshold estimation.
        """
        # Subsample for efficiency if datasets are large
        max_n = 500
        ref = self.reference_data
        curr = current_data
        if len(ref) > max_n:
            idx = np.random.choice(len(ref), max_n, replace=False)
            ref = ref[idx]
        if len(curr) > max_n:
            idx = np.random.choice(len(curr), max_n, replace=False)
            curr = curr[idx]

        k_xx = self._kernel_mean(ref, ref)
        k_yy = self._kernel_mean(curr, curr)
        k_xy = self._kernel_mean(ref, curr)

        mmd_sq = k_xx + k_yy - 2 * k_xy

        # Permutation test for threshold
        combined = np.vstack([ref, curr])
        n_ref = len(ref)
        n_perms = 100
        null_mmds = np.zeros(n_perms)

        for p in range(n_perms):
            perm = np.random.permutation(len(combined))
            perm_x = combined[perm[:n_ref]]
            perm_y = combined[perm[n_ref:]]

            pk_xx = self._kernel_mean(perm_x, perm_x)
            pk_yy = self._kernel_mean(perm_y, perm_y)
            pk_xy = self._kernel_mean(perm_x, perm_y)
            null_mmds[p] = pk_xx + pk_yy - 2 * pk_xy

        threshold = np.percentile(null_mmds, (1 - self.alpha) * 100)
        return float(mmd_sq) > float(threshold)


class ConceptDriftEngine:
    """
    Multi-signal consensus drift detection engine.
    Triggers retraining only if ≥ 2 detectors agree that drift has occurred.
    """

    def __init__(self, reference_features: np.ndarray, consensus_threshold: int = 2):
        self.adwin = ADWINDetector()
        self.ph = PageHinkleyDetector()
        self.ks = KSDetector(reference_features)
        self.mmd = MMDDetector(reference_features)
        self.consensus_threshold = consensus_threshold
        self._last_results: Dict[str, bool] = {}

    def evaluate(
        self, current_batch: np.ndarray, prediction_errors: List[float]
    ) -> bool:
        """
        Evaluate if concept drift has occurred.
        Requires ≥ consensus_threshold detectors to agree.

        Parameters
        ----------
        current_batch : np.ndarray
            Current feature batch (n_samples, n_features).
        prediction_errors : list of float
            Per-sample prediction errors or confidence scores.
        """
        # Process ALL errors through stream detectors (don't short-circuit)
        adwin_hits = sum(1 for e in prediction_errors if self.adwin.update(e))
        ph_hits = sum(1 for e in prediction_errors if self.ph.update(e))

        adwin_drift = adwin_hits > 0
        ph_drift = ph_hits > 0
        ks_drift = self.ks.detect(current_batch)
        mmd_drift = self.mmd.detect(current_batch)

        self._last_results = {
            "adwin": adwin_drift,
            "page_hinkley": ph_drift,
            "ks": ks_drift,
            "mmd": mmd_drift,
        }

        return sum(self._last_results.values()) >= self.consensus_threshold

    @property
    def last_results(self) -> Dict[str, bool]:
        return dict(self._last_results)


class ConsensusDriftDetector:
    """
    Lightweight streaming drift detector with an update() interface.
    Wraps ADWINDetector and PageHinkleyDetector; triggers when both agree.

    Parameters
    ----------
    threshold : int
        Minimum number of sub-detectors that must agree to signal drift.
    """

    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self._adwin = ADWINDetector()
        self._ph = PageHinkleyDetector()

    def update(self, value: float) -> bool:
        """Process one observation. Returns True when consensus drift detected."""
        hits = 0
        if self._adwin.update(value):
            hits += 1
        if self._ph.update(value):
            hits += 1
        return bool(hits >= self.threshold)
