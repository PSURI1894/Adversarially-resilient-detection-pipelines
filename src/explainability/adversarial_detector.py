"""
================================================================================
ADVERSARIAL DETECTION VIA ATTRIBUTION FINGERPRINTING
================================================================================
Novel contribution: adversarial samples often have abnormal attribution
patterns even when they successfully fool the classifier. We learn a
"normal" attribution distribution from clean data and flag deviations.

Includes:
    - AttributionFingerprintDetector (GMM on SHAP vectors + Mahalanobis)
    - FeatureSensitivityAnalyzer (gradient-based vulnerability analysis)
================================================================================
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional, List, Dict, Any


class AttributionFingerprintDetector:
    """
    Detects adversarial samples by analyzing their SHAP attribution
    fingerprint against a learned distribution of clean attributions.

    Pipeline:
        1. Fit: compute SHAP values on clean training data
        2. Model: fit a Gaussian Mixture Model on attribution vectors
        3. Score: for new samples, compute attribution Mahalanobis distance
        4. Flag: samples beyond threshold are suspicious

    Parameters
    ----------
    shap_explainer : SHAPExplainer
        Configured SHAP explainer for computing attribution vectors.
    n_components : int
        Number of GMM components for modeling the attribution distribution.
    threshold_percentile : float
        Percentile of clean scores to set as the detection threshold.
    """

    def __init__(
        self, shap_explainer, n_components: int = 3, threshold_percentile: float = 95.0
    ):
        self.shap_explainer = shap_explainer
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.gmm: Optional[GaussianMixture] = None
        self.threshold: Optional[float] = None
        self.clean_attributions_: Optional[np.ndarray] = None

    def fit(self, X_clean: np.ndarray):
        """
        Learn the clean attribution distribution.

        Parameters
        ----------
        X_clean : np.ndarray
            Clean (non-adversarial) samples to establish the baseline.
        """
        # Compute SHAP values for clean data
        attributions = self.shap_explainer.explain_batch(X_clean)
        self.clean_attributions_ = attributions

        # Fit GMM on attribution vectors
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=42,
            max_iter=200,
        )
        self.gmm.fit(attributions)

        # Set threshold from clean data scores
        clean_scores = -self.gmm.score_samples(attributions)
        self.threshold = float(np.percentile(clean_scores, self.threshold_percentile))

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for new samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Higher score = more anomalous attribution pattern.
        """
        if self.gmm is None:
            raise ValueError("Must call fit() first.")

        attributions = self.shap_explainer.explain_batch(X)
        return -self.gmm.score_samples(attributions)

    def detect(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect adversarial samples.

        Returns
        -------
        dict with keys:
            'is_adversarial': np.ndarray of bool
            'scores': np.ndarray
            'threshold': float
            'n_flagged': int
        """
        scores = self.score(X)
        is_adv = scores > self.threshold
        return {
            "is_adversarial": is_adv,
            "scores": scores,
            "threshold": self.threshold,
            "n_flagged": int(np.sum(is_adv)),
        }


class FeatureSensitivityAnalyzer:
    """
    Gradient-based feature sensitivity analysis.

    Identifies which features are most vulnerable to adversarial
    manipulation and generates hardening recommendations.

    Parameters
    ----------
    model : object
        Model to analyze (must support gradient computation or finite-diff).
    feature_names : list of str, optional
        Human-readable feature names.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names

    def compute_sensitivity(self, X: np.ndarray, delta: float = 1e-4) -> np.ndarray:
        """
        Compute per-feature sensitivity via finite differences.

        sensitivity_j = mean | ∂P(malicious) / ∂x_j |

        Parameters
        ----------
        X : np.ndarray
            Input samples.
        delta : float
            Finite difference step size.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Mean absolute sensitivity per feature.
        """
        n_features = X.shape[1]
        sensitivities = np.zeros(n_features)

        base_probs = self.model.predict_proba(X)[:, 1]

        for j in range(n_features):
            X_plus = X.copy()
            X_plus[:, j] += delta
            probs_plus = self.model.predict_proba(X_plus)[:, 1]

            grad_approx = (probs_plus - base_probs) / delta
            sensitivities[j] = float(np.mean(np.abs(grad_approx)))

        return sensitivities

    def vulnerability_report(self, X: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a hardening recommendation report.

        Returns
        -------
        dict with keys:
            'sensitivities': dict of feature_name → score
            'most_vulnerable': list of top-K feature names
            'recommendations': list of actionable hardening suggestions
        """
        sens = self.compute_sensitivity(X)
        names = self.feature_names or [f"f{i}" for i in range(len(sens))]

        ranked = sorted(zip(names, sens.tolist()), key=lambda t: t[1], reverse=True)
        most_vulnerable = [name for name, _ in ranked[:top_k]]

        recommendations = []
        for name, score in ranked[:top_k]:
            if "iat" in name.lower() or "duration" in name.lower():
                rec = f"Apply Z-score clipping to '{name}' (temporal feature, sensitivity={score:.4f})"
            elif "bytes" in name.lower() or "pkt" in name.lower():
                rec = f"Apply log-transform + robust scaling to '{name}' (volume feature, sensitivity={score:.4f})"
            elif "port" in name.lower() or "flag" in name.lower():
                rec = f"Consider discretizing '{name}' or marking immutable (protocol feature, sensitivity={score:.4f})"
            else:
                rec = f"Add noise-injection regularization for '{name}' (sensitivity={score:.4f})"
            recommendations.append(rec)

        return {
            "sensitivities": dict(ranked),
            "most_vulnerable": most_vulnerable,
            "recommendations": recommendations,
        }
