"""
================================================================================
ONLINE CONFORMAL PREDICTION — STREAMING NON-STATIONARY ENVIRONMENTS
================================================================================
Covers:
    - Adaptive Conformal Inference (ACI) with rolling quantile updates
    - Exponential forgetting for non-stationary distributions
    - Dynamic calibration window sizing
    - Exchangeability monitoring (detects when i.i.d. breaks)
================================================================================
"""

import numpy as np
from collections import deque


class OnlineConformalPredictor:
    """
    Streaming conformal predictor with Adaptive Conformal Inference (ACI).

    At each time step t:
        1. Receive new (x_t, y_t)
        2. Compute non-conformity score s_t = 1 − P(y_t | x_t)
        3. Update the running quantile estimate via ACI:
              α_t+1 = α_t + γ (α − err_t)
           where err_t = 1 if y_t ∉ C_t(x_t).
        4. Update the score buffer with exponential forgetting.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    gamma : float
        ACI learning rate for adaptive α updates.
    window_size : int
        Maximum number of recent scores to keep in the calibration buffer.
    forgetting_factor : float
        Exponential weight decay. 1.0 = no forgetting, 0.9 = recent 10× more important.
    """

    def __init__(
        self, alpha=0.05, gamma=0.01, window_size=5000, forgetting_factor=0.995
    ):
        self.alpha_target = alpha
        self.alpha_current = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.forgetting_factor = forgetting_factor

        self.score_buffer = deque(maxlen=window_size)
        self.weight_buffer = deque(maxlen=window_size)
        self.q_hat = None

        # Tracking
        self.coverage_history = []
        self.alpha_history = [alpha]
        self.exchangeability_stats = []

    # ------------------------------------------------------------------
    # Core update loop
    # ------------------------------------------------------------------

    def update(self, model, x_t, y_t):
        """
        Process a single observation (x_t, y_t).

        Returns the prediction set that would have been produced for x_t.
        """
        x_t = np.atleast_2d(x_t)
        probs = model.predict_proba(x_t)[0]
        score = 1.0 - probs[y_t]

        # Compute prediction set *before* updating (honest evaluation)
        if self.q_hat is not None:
            pred_set = [c for c in range(len(probs)) if (1.0 - probs[c]) <= self.q_hat]
            if not pred_set:
                pred_set = [int(np.argmax(probs))]
        else:
            pred_set = list(range(len(probs)))  # include all if not calibrated

        # Track coverage
        covered = int(y_t in pred_set)
        self.coverage_history.append(covered)

        # ACI update: adjust alpha
        err_t = 1 - covered
        self.alpha_current = self.alpha_current + self.gamma * (
            self.alpha_target - err_t
        )
        self.alpha_current = np.clip(self.alpha_current, 0.001, 0.5)
        self.alpha_history.append(self.alpha_current)

        # Add score to buffer with weight 1.0 (will decay relatively)
        self.score_buffer.append(score)
        self.weight_buffer.append(1.0)

        # Recompute q_hat with exponential forgetting
        self._update_quantile()

        return pred_set

    def update_batch(self, model, X_batch, y_batch):
        """Process a batch of observations sequentially."""
        all_sets = []
        for i in range(len(X_batch)):
            pset = self.update(model, X_batch[i], y_batch[i])
            all_sets.append(pset)
        return all_sets

    # ------------------------------------------------------------------
    # Quantile estimation with forgetting
    # ------------------------------------------------------------------

    def _update_quantile(self):
        """
        Compute weighted quantile from the score buffer.

        Weights decay exponentially: w_i = forgetting_factor^(n - i)
        where i is the insertion order (oldest = 0).
        """
        if len(self.score_buffer) < 2:
            self.q_hat = 1.0  # conservative until enough data
            return

        scores = np.array(self.score_buffer)
        n = len(scores)

        # Build decaying weights: most recent has weight 1.0
        weights = np.array([self.forgetting_factor ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Weighted quantile
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)

        target_level = 1.0 - self.alpha_current
        idx = np.searchsorted(cumsum_weights, target_level)
        idx = min(idx, len(sorted_scores) - 1)
        self.q_hat = float(sorted_scores[idx])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def prediction_sets(self, X_test, model):
        """Generate prediction sets for a batch using current q_hat."""
        if self.q_hat is None:
            raise ValueError("No data has been processed yet. Call update() first.")

        probs = model.predict_proba(X_test)
        pred_sets = []
        for p in probs:
            pset = [c for c in range(len(p)) if (1.0 - p[c]) <= self.q_hat]
            if not pset:
                pset = [int(np.argmax(p))]
            pred_sets.append(pset)
        return pred_sets

    # ------------------------------------------------------------------
    # Exchangeability monitoring
    # ------------------------------------------------------------------

    def check_exchangeability(self, lookback=200):
        """
        Monitor whether the i.i.d. (exchangeability) assumption holds.

        Uses a simple runs test on coverage indicators:
        if the sequence of coverage / non-coverage is non-random,
        exchangeability may be violated.

        Returns
        -------
        dict with 'exchangeable' (bool), 'runs_stat', 'expected_runs'
        """
        if len(self.coverage_history) < lookback:
            return {"exchangeable": True, "runs_stat": 0, "expected_runs": 0}

        seq = np.array(self.coverage_history[-lookback:])
        n1 = np.sum(seq == 1)
        n0 = np.sum(seq == 0)

        if n1 == 0 or n0 == 0:
            return {"exchangeable": False, "runs_stat": 0, "expected_runs": 0}

        # Count runs
        runs = 1 + np.sum(np.diff(seq) != 0)
        expected = 1 + 2 * n1 * n0 / (n1 + n0)
        variance = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (
            (n1 + n0) ** 2 * (n1 + n0 - 1)
        ) + 1e-8
        z = (runs - expected) / np.sqrt(variance)

        # |z| > 2 → reject exchangeability at ~95% confidence
        result = {
            "exchangeable": abs(z) < 2.0,
            "runs_stat": float(z),
            "expected_runs": float(expected),
            "actual_runs": int(runs),
        }
        self.exchangeability_stats.append(result)
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def rolling_coverage(self, window=100):
        """Compute rolling empirical coverage over last `window` predictions."""
        if len(self.coverage_history) < window:
            return (
                float(np.mean(self.coverage_history)) if self.coverage_history else 0.0
            )
        return float(np.mean(self.coverage_history[-window:]))

    def get_diagnostics(self):
        """Return diagnostic summary."""
        return {
            "n_processed": len(self.coverage_history),
            "current_alpha": float(self.alpha_current),
            "current_q_hat": float(self.q_hat) if self.q_hat else None,
            "rolling_coverage_100": self.rolling_coverage(100),
            "rolling_coverage_500": self.rolling_coverage(500),
            "buffer_size": len(self.score_buffer),
        }
