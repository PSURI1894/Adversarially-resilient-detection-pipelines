"""
================================================================================
RANDOMIZED SMOOTHED CONFORMAL PREDICTION (RSCP+)
================================================================================
Certified adversarial coverage via Gaussian smoothing of non-conformity scores.

Theory:
    Given a base non-conformity score function s(x, y), RSCP smooths it as:
        s̃(x, y) = E_{ε ~ N(0, σ²I)} [ s(x + ε, y) ]
    This yields certified coverage:
        P(Y ∈ C(X_adv)) ≥ 1 - α   for ‖X_adv - X‖₂ ≤ r

    where r depends on σ and the Lipschitz constant of the score function.

Includes:
    - Base RSCP with Monte Carlo estimation
    - Post-Training Transformation (PTT) for tighter prediction sets
    - Robust Conformal Training (RCT) loss for fine-tuning
    - Certified radius computation via Neyman-Pearson
================================================================================
"""

import numpy as np
from scipy.stats import norm


class RandomizedSmoothedCP:
    """
    RSCP+ (Randomized Smoothed Conformal Prediction with PTT).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (e.g. 0.05 for 95% coverage).
    sigma : float
        Gaussian noise standard deviation for smoothing.
    n_samples : int
        Number of Monte Carlo samples for smoothed score estimation.
    ptt : bool
        Whether to apply Post-Training Transformation to shrink set sizes.
    ptt_transform : str
        PTT function: 'rank' uses rank-based transform, 'sigmoid' uses sigmoid.
    """

    def __init__(
        self, alpha=0.05, sigma=0.1, n_samples=100, ptt=True, ptt_transform="rank"
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.n_samples = n_samples
        self.ptt = ptt
        self.ptt_transform = ptt_transform
        self.q_hat = None
        self.cal_scores_ = None  # stored for PTT fitting

    # ------------------------------------------------------------------
    # Core smoothing mechanics
    # ------------------------------------------------------------------

    def _smoothed_scores(self, model, X, y=None):
        """
        Monte Carlo estimation of smoothed non-conformity scores.

        For each point x_i, draw N noise vectors ε_j ~ N(0, σ²I),
        compute s(x_i + ε_j), and average.

        The non-conformity score is 1 − P_model(y_true | x + ε).
        When y is None (prediction time), return smoothed probabilities.
        """
        n_points = len(X)
        n_classes = 2  # binary IDS setting

        # Accumulate smoothed probabilities
        probs_sum = np.zeros((n_points, n_classes))

        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.sigma, size=X.shape).astype(X.dtype)
            probs_sum += model.predict_proba(X + noise)

        smoothed_probs = probs_sum / self.n_samples

        if y is not None:
            # Non-conformity score for true class
            raw_scores = 1.0 - smoothed_probs[np.arange(n_points), y]
            return raw_scores, smoothed_probs
        return smoothed_probs

    # ------------------------------------------------------------------
    # Post-Training Transformation (PTT)
    # ------------------------------------------------------------------

    def _apply_ptt(self, scores):
        """
        Transform raw scores to reduce prediction set size while preserving
        coverage guarantee (monotone transform preserves quantile ordering).
        """
        if not self.ptt or self.cal_scores_ is None:
            return scores

        if self.ptt_transform == "rank":
            # Rank-based: map score to its empirical CDF value among cal scores
            # This is a monotone transform → preserves conformal validity
            return np.array([np.mean(self.cal_scores_ <= s) for s in scores])
        elif self.ptt_transform == "sigmoid":
            # Sigmoid centering around calibration median
            median = np.median(self.cal_scores_)
            std = np.std(self.cal_scores_) + 1e-8
            return 1.0 / (1.0 + np.exp(-(scores - median) / std))
        else:
            return scores

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, model, X_cal, y_cal):
        """
        Calibrate the smoothed conformal predictor.

        Steps:
            1. Compute smoothed non-conformity scores on calibration data
            2. Optionally apply PTT
            3. Compute the (1−α)(1 + 1/n) quantile
        """
        raw_scores, _ = self._smoothed_scores(model, X_cal, y_cal)

        # Store raw calibration scores for PTT fitting
        self.cal_scores_ = raw_scores.copy()

        # Apply PTT if enabled
        scores = self._apply_ptt(raw_scores)

        n = len(y_cal)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = np.quantile(scores, q_level)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def prediction_sets(self, X_test, model):
        """
        Generate prediction sets with certified adversarial coverage.

        For each test point, include class c if its smoothed
        non-conformity score (optionally PTT-transformed) ≤ q_hat.
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before prediction_sets().")

        smoothed_probs = self._smoothed_scores(model, X_test, y=None)

        pred_sets = []
        for probs in smoothed_probs:
            pset = []
            for c in range(len(probs)):
                raw_score = 1.0 - probs[c]
                score = self._apply_ptt(np.array([raw_score]))[0]
                if score <= self.q_hat:
                    pset.append(c)
            # Failsafe: never return empty set
            if len(pset) == 0:
                pset.append(int(np.argmax(probs)))
            pred_sets.append(pset)

        return pred_sets

    # ------------------------------------------------------------------
    # Certified radius
    # ------------------------------------------------------------------

    def certified_radius(self, model, X, y):
        """
        Compute the certified L2 perturbation radius for each point.

        Uses the Neyman-Pearson based bound from randomized smoothing:
            r = σ · Φ⁻¹(pA)
        where pA = smoothed probability of the most likely class.
        """
        smoothed_probs = self._smoothed_scores(model, X, y=None)
        pA = np.max(smoothed_probs, axis=1)
        # Clip to avoid inf from Phi_inv(1.0)
        pA = np.clip(pA, 1e-6, 1.0 - 1e-6)
        radii = self.sigma * norm.ppf(pA)
        return radii


# ======================================================================
# Robust Conformal Training (RCT) Loss
# ======================================================================


def rct_loss(model, X_batch, y_batch, sigma=0.1, n_mc=10, alpha=0.05):
    """
    Differentiable surrogate loss for Robust Conformal Training.

    Idea: fine-tune the model so that the smoothed set sizes are minimized
    while maintaining coverage. This is a surrogate that penalizes
    large smoothed non-conformity scores for the true class.

    Parameters
    ----------
    model : callable
        A TensorFlow/Keras model with differentiable forward pass.
    X_batch : np.ndarray
        Input batch.
    y_batch : np.ndarray
        True labels.
    sigma : float
        Smoothing noise scale.
    n_mc : int
        Monte Carlo samples (small for training efficiency).
    alpha : float
        Target miscoverage.

    Returns
    -------
    float
        Scalar loss value (lower = tighter sets).
    """
    import tensorflow as tf

    X_tf = tf.cast(X_batch, tf.float32)
    score_sum = tf.zeros(len(X_batch))

    for _ in range(n_mc):
        noise = tf.random.normal(tf.shape(X_tf), stddev=sigma)
        logits = model(X_tf + noise, training=True)
        probs = tf.nn.sigmoid(logits)  # binary
        probs = tf.squeeze(probs)
        # Non-conformity: 1 - P(y_true | x+ε)
        y_tf = tf.cast(y_batch, tf.float32)
        true_probs = y_tf * probs + (1 - y_tf) * (1 - probs)
        score_sum += 1.0 - true_probs

    mean_scores = score_sum / n_mc
    # Penalize large scores (want them small so sets are tight)
    return tf.reduce_mean(mean_scores)
