"""
================================================================================
LIME EXPLAINER — DOMAIN-AWARE TABULAR EXPLANATIONS
================================================================================
Provides:
    - Tabular LIME with domain-aware perturbation (respects feature constraints)
    - Custom distance kernel tuned for network flow feature space
    - Fidelity scoring to assess explanation quality
    - Top-K feature explanations formatted for analyst consumption
================================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.linear_model import Ridge


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for network IDS.

    Parameters
    ----------
    model : object
        The model to explain. Must have `predict_proba(X)`.
    feature_names : list of str, optional
        Human-readable feature names.
    n_samples : int
        Number of perturbed samples per explanation.
    kernel_width : float
        Width of the exponential kernel for locality weighting.
    non_negative_features : list of int, optional
        Indices of features that must remain ≥ 0 after perturbation.
    immutable_features : list of int, optional
        Indices of features that should never be perturbed (e.g., ports).
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None,
                 n_samples: int = 500, kernel_width: float = 0.75,
                 non_negative_features: Optional[List[int]] = None,
                 immutable_features: Optional[List[int]] = None):
        self.model = model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.non_negative_features = non_negative_features or []
        self.immutable_features = immutable_features or []

    # ------------------------------------------------------------------
    # Core explanation
    # ------------------------------------------------------------------

    def explain_instance(self, x: np.ndarray, top_k: int = 5,
                         target_class: int = 1) -> Dict[str, Any]:
        """
        Explain a single prediction using locally weighted linear model.

        Parameters
        ----------
        x : np.ndarray
            Single input sample (1D).
        top_k : int
            Number of top features to highlight.
        target_class : int
            Which class probability to explain.

        Returns
        -------
        dict with keys:
            'coefficients': np.ndarray (linear model weights)
            'intercept': float
            'fidelity': float (R² of local surrogate)
            'top_features': list of (name, weight) tuples
            'prediction': float
        """
        x = np.atleast_1d(x).flatten()
        n_features = len(x)

        # Generate domain-aware perturbed samples
        perturbed = self._generate_perturbations(x, n_features)

        # Get model predictions on perturbed data
        preds = self.model.predict_proba(perturbed)[:, target_class]

        # Compute locality weights via exponential kernel
        distances = np.sqrt(np.sum((perturbed - x) ** 2, axis=1))
        weights = np.exp(-distances ** 2 / (self.kernel_width ** 2))

        # Fit weighted ridge regression (surrogate)
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(perturbed, preds, sample_weight=weights)

        # Fidelity: weighted R²
        preds_surrogate = surrogate.predict(perturbed)
        ss_res = np.sum(weights * (preds - preds_surrogate) ** 2)
        ss_tot = np.sum(weights * (preds - np.average(preds, weights=weights)) ** 2)
        fidelity = 1.0 - ss_res / (ss_tot + 1e-8)

        # Top features
        names = self.feature_names or [f"f{i}" for i in range(n_features)]
        coef = surrogate.coef_
        top = sorted(
            zip(names, coef.tolist()),
            key=lambda t: abs(t[1]),
            reverse=True
        )[:top_k]

        return {
            "coefficients": coef,
            "intercept": float(surrogate.intercept_),
            "fidelity": float(fidelity),
            "top_features": top,
            "prediction": float(self.model.predict_proba(x.reshape(1, -1))[0, target_class]),
        }

    def explain_batch(self, X: np.ndarray, top_k: int = 5,
                      target_class: int = 1) -> List[Dict[str, Any]]:
        """Explain multiple instances."""
        return [self.explain_instance(X[i], top_k, target_class) for i in range(len(X))]

    # ------------------------------------------------------------------
    # Domain-aware perturbation
    # ------------------------------------------------------------------

    def _generate_perturbations(self, x: np.ndarray,
                                n_features: int) -> np.ndarray:
        """
        Generate perturbed samples respecting IDS domain constraints.

        - Non-negative features are clipped to ≥ 0
        - Immutable features are held constant
        - Perturbation scale adapts to feature magnitude
        """
        rng = np.random.default_rng()

        # Scale-aware noise: proportional to feature magnitude
        scale = np.abs(x) + 1e-6
        noise = rng.normal(0, scale * 0.3, size=(self.n_samples, n_features))
        perturbed = x + noise

        # Enforce non-negative constraints
        for idx in self.non_negative_features:
            perturbed[:, idx] = np.maximum(perturbed[:, idx], 0.0)

        # Enforce immutability
        for idx in self.immutable_features:
            perturbed[:, idx] = x[idx]

        return perturbed.astype(np.float32)

    # ------------------------------------------------------------------
    # Fidelity assessment
    # ------------------------------------------------------------------

    def assess_fidelity(self, X: np.ndarray, n_trials: int = 10) -> Dict[str, float]:
        """
        Aggregate fidelity score across multiple instances.

        Returns
        -------
        dict with 'mean_fidelity', 'min_fidelity', 'max_fidelity'
        """
        fidelities = []
        indices = np.random.choice(len(X), min(n_trials, len(X)), replace=False)
        for i in indices:
            result = self.explain_instance(X[i])
            fidelities.append(result["fidelity"])
        return {
            "mean_fidelity": float(np.mean(fidelities)),
            "min_fidelity": float(np.min(fidelities)),
            "max_fidelity": float(np.max(fidelities)),
        }
