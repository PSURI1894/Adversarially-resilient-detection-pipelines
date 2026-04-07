"""
================================================================================
CONFORMAL DEFENSE AGAINST CALIBRATION POISONING
================================================================================
Algorithms to safeguard empirical quantiles from adversary-injected calibration
data points.

Includes:
    - RobustCalibration:  partitioned majority-vote / trimmed-mean aggregation
    - CalibrationIntegrityMonitor:  KS-test + Isolation Forest drift detection
================================================================================
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp


class RobustCalibration:
    """
    Partition-based robust calibration defense.

    Splits calibration data into K disjoint subsets, computes q_hat
    independently on each, then aggregates.  Provable guarantee:
    valid coverage even if ≤ floor(K/2) − 1 partitions are fully poisoned.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    n_partitions : int
        Number of disjoint subsets K.
    aggregation : str
        'median' (robust to < 50% corruption) or 'trimmed_mean'.
    trim_fraction : float
        Fraction to trim from each side when using trimmed_mean.
    """

    def __init__(
        self, alpha=0.05, n_partitions=7, aggregation="median", trim_fraction=0.2
    ):
        self.alpha = alpha
        self.n_partitions = n_partitions
        self.aggregation = aggregation
        self.trim_fraction = trim_fraction
        self.q_hat = None
        self.partition_q_hats_ = []

    def _compute_quantile_on_partition(self, model, X_part, y_part):
        """Compute non-conformity quantile on a single partition."""
        probs = model.predict_proba(X_part)
        scores = 1.0 - probs[np.arange(len(y_part)), y_part]

        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(scores, q_level))

    def calibrate(self, model, X_cal, y_cal):
        """
        Partition calibration set into K subsets, compute per-partition q_hat,
        then aggregate robustly.
        """
        indices = np.arange(len(y_cal))
        np.random.shuffle(indices)
        partitions = np.array_split(indices, self.n_partitions)

        self.partition_q_hats_ = []
        for p_idx in partitions:
            if len(p_idx) < 2:
                continue
            q = self._compute_quantile_on_partition(model, X_cal[p_idx], y_cal[p_idx])
            self.partition_q_hats_.append(q)

        estimates = np.array(self.partition_q_hats_)

        if self.aggregation == "median":
            self.q_hat = float(np.median(estimates))
        elif self.aggregation == "trimmed_mean":
            k = int(len(estimates) * self.trim_fraction)
            if k > 0:
                trimmed = np.sort(estimates)[k:-k]
            else:
                trimmed = estimates
            self.q_hat = float(np.mean(trimmed))
        else:
            self.q_hat = float(np.median(estimates))

    def prediction_sets(self, X_test, model):
        """Generate prediction sets using the robustly estimated q_hat."""
        if self.q_hat is None:
            raise ValueError("Must call calibrate() first.")

        probs = model.predict_proba(X_test)
        pred_sets = []
        for p in probs:
            pset = [c for c in range(len(p)) if (1.0 - p[c]) <= self.q_hat]
            if not pset:
                pset = [int(np.argmax(p))]
            pred_sets.append(pset)
        return pred_sets

    @property
    def max_poisoned_partitions(self):
        """Maximum number of poisoned partitions that can be tolerated."""
        return self.n_partitions // 2 - 1


class CalibrationIntegrityMonitor:
    """
    Detects calibration data drift / poisoning via statistical tests.

    Uses:
        - Kolmogorov-Smirnov 2-sample test (distribution shift)
        - Isolation Forest (score-level anomaly detection)
        - Score moment monitoring (mean/variance shift)

    Parameters
    ----------
    baseline_scores : np.ndarray
        Clean (trusted) non-conformity scores from initial calibration.
    contamination : float
        Expected anomaly fraction for Isolation Forest.
    drift_threshold : float
        P-value threshold below which drift is flagged.
    """

    def __init__(self, baseline_scores, contamination=0.05, drift_threshold=0.05):
        self.baseline_scores = np.asarray(baseline_scores, dtype=float)
        self.drift_threshold = drift_threshold
        self.baseline_mean = float(np.mean(self.baseline_scores))
        self.baseline_std = float(np.std(self.baseline_scores)) + 1e-8

        self.detector = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        if len(self.baseline_scores) > 0:
            self.detector.fit(self.baseline_scores.reshape(-1, 1))

    def detect_drift(self, recent_scores):
        """
        Kolmogorov-Smirnov 2-sample test.

        Returns
        -------
        dict with keys: 'drift_detected' (bool), 'ks_stat', 'p_value'
        """
        recent = np.asarray(recent_scores, dtype=float)
        stat, p_value = ks_2samp(self.baseline_scores, recent)
        return {
            "drift_detected": bool(p_value < self.drift_threshold),
            "ks_stat": float(stat),
            "p_value": float(p_value),
        }

    def detect_anomalies(self, recent_scores):
        """
        Isolation Forest anomaly scan on new scores.

        Returns
        -------
        dict with keys: 'anomaly_fraction', 'anomaly_indices'
        """
        recent = np.asarray(recent_scores, dtype=float).reshape(-1, 1)
        preds = self.detector.predict(recent)
        anomaly_mask = preds == -1
        return {
            "anomaly_fraction": float(np.mean(anomaly_mask)),
            "anomaly_indices": np.where(anomaly_mask)[0].tolist(),
        }

    def detect_moment_shift(self, recent_scores, z_threshold=3.0):
        """
        Check if the mean of recent scores has shifted significantly
        relative to the baseline (z-test style).
        """
        recent = np.asarray(recent_scores, dtype=float)
        recent_mean = float(np.mean(recent))
        se = self.baseline_std / np.sqrt(len(recent))
        z = abs(recent_mean - self.baseline_mean) / (se + 1e-8)
        return {
            "moment_shift_detected": z > z_threshold,
            "z_score": float(z),
            "baseline_mean": self.baseline_mean,
            "recent_mean": recent_mean,
        }

    def full_integrity_check(self, recent_scores):
        """Run all integrity checks and return combined report."""
        return {
            "drift": self.detect_drift(recent_scores),
            "anomalies": self.detect_anomalies(recent_scores),
            "moment_shift": self.detect_moment_shift(recent_scores),
        }
