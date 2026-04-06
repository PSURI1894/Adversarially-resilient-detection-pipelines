"""
================================================================================
ADAPTIVE CONFORMAL PREDICTION — APS, RAPS, AND MONDRIAN CP
================================================================================
Covers:
    - Adaptive Prediction Sets (APS): cumulative-softmax ordering
    - Regularized APS (RAPS): penalty for large set sizes
    - Class-conditional coverage (per-class α guarantees)
    - Mondrian conformal prediction for group-conditional validity
================================================================================
"""

import numpy as np
from collections import defaultdict


class AdaptiveConformalPredictor:
    """
    Adaptive Prediction Sets (APS) and Regularized APS (RAPS).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    method : str
        'APS' or 'RAPS'.
    k_reg : int
        RAPS regularization threshold — penalty kicks in for ranks > k_reg.
    penalty : float
        RAPS penalty coefficient λ.
    """

    def __init__(self, alpha=0.05, method="RAPS", k_reg=1, penalty=0.05):
        self.alpha = alpha
        self.method = method
        self.k_reg = k_reg
        self.penalty = penalty
        self.q_hat = None

    def _sort_probs(self, probs):
        """Sort class probabilities descending; return sorted probs and original indices."""
        sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
        return sorted_probs, sorted_indices

    def _conformity_score(self, probs_row, sorted_indices_row, cumsum_row, label):
        """
        Compute the conformity score for a single sample.

        APS score:  cumulative probability up to (and including) the true class.
        RAPS score: APS score + λ · max(0, rank − k_reg + 1).
        """
        rank = int(np.where(sorted_indices_row == label)[0][0])
        score = cumsum_row[rank]

        if self.method == "RAPS" and rank > self.k_reg - 1:
            score += self.penalty * (rank - self.k_reg + 1)

        # Add randomized tie-breaking U ~ Unif[0,1] * P(y_rank)
        # This smooths the empirical quantile and avoids conservatism
        u = np.random.uniform()
        score -= u * (cumsum_row[rank] - (cumsum_row[rank - 1] if rank > 0 else 0.0))

        return score

    def calibrate(self, model, X_cal, y_cal):
        """Calibrate on held-out calibration set."""
        probs = model.predict_proba(X_cal)
        sorted_probs, sorted_indices = self._sort_probs(probs)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        scores = np.array([
            self._conformity_score(probs[i], sorted_indices[i], cumsum_probs[i], y_cal[i])
            for i in range(len(y_cal))
        ])

        n = len(y_cal)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = np.quantile(scores, q_level)

    def prediction_sets(self, X_test, model):
        """Generate adaptive prediction sets."""
        if self.q_hat is None:
            raise ValueError("Must call calibrate() first.")

        probs = model.predict_proba(X_test)
        sorted_probs, sorted_indices = self._sort_probs(probs)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        pred_sets = []
        for i in range(len(X_test)):
            pset = []
            for rank in range(probs.shape[1]):
                cost = cumsum_probs[i, rank]
                if self.method == "RAPS" and rank > self.k_reg - 1:
                    cost += self.penalty * (rank - self.k_reg + 1)
                pset.append(int(sorted_indices[i, rank]))
                if cost >= self.q_hat:
                    break
            pred_sets.append(pset)

        return pred_sets

    def avg_set_size(self, X_test, model):
        """Utility: Average prediction set size."""
        sets = self.prediction_sets(X_test, model)
        return np.mean([len(s) for s in sets])


class ClassConditionalCP:
    """
    Class-conditional conformal prediction.

    Maintains separate calibration quantiles per class so that
    P(Y ∈ C(X) | Y = k) ≥ 1 − α   for all k.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.q_hats = {}  # {class_label: q_hat}

    def calibrate(self, model, X_cal, y_cal):
        probs = model.predict_proba(X_cal)
        classes = np.unique(y_cal)

        for cls in classes:
            mask = y_cal == cls
            true_probs = probs[mask, cls]
            scores = 1.0 - true_probs

            n = np.sum(mask)
            q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            self.q_hats[int(cls)] = np.quantile(scores, q_level)

    def prediction_sets(self, X_test, model):
        if not self.q_hats:
            raise ValueError("Must call calibrate() first.")

        probs = model.predict_proba(X_test)
        pred_sets = []
        for p in probs:
            pset = [c for c in self.q_hats if (1.0 - p[c]) <= self.q_hats[c]]
            if not pset:
                pset = [int(np.argmax(p))]
            pred_sets.append(pset)
        return pred_sets


class MondrianCP:
    """
    Mondrian Conformal Prediction for group-conditional validity.

    Provides coverage guarantees within predefined groups (e.g. protocol type,
    source subnet). Each group gets its own calibration quantile.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate per group.
    group_fn : callable
        Function mapping X_row → group_label (string or int).
    """

    def __init__(self, alpha=0.05, group_fn=None):
        self.alpha = alpha
        self.group_fn = group_fn or (lambda x: 0)  # default: single group
        self.group_q_hats = {}

    def calibrate(self, model, X_cal, y_cal):
        probs = model.predict_proba(X_cal)

        # Bucket by group
        group_scores = defaultdict(list)
        for i in range(len(y_cal)):
            g = self.group_fn(X_cal[i])
            score = 1.0 - probs[i, y_cal[i]]
            group_scores[g].append(score)

        for g, scores in group_scores.items():
            scores = np.array(scores)
            n = len(scores)
            q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            self.group_q_hats[g] = np.quantile(scores, q_level)

    def prediction_sets(self, X_test, model):
        if not self.group_q_hats:
            raise ValueError("Must call calibrate() first.")

        probs = model.predict_proba(X_test)
        pred_sets = []
        for i, p in enumerate(probs):
            g = self.group_fn(X_test[i])
            q = self.group_q_hats.get(g, max(self.group_q_hats.values()))
            pset = [c for c in range(len(p)) if (1.0 - p[c]) <= q]
            if not pset:
                pset = [int(np.argmax(p))]
            pred_sets.append(pset)
        return pred_sets
