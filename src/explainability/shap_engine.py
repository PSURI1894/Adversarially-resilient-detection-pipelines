"""
================================================================================
SHAP EXPLAINER — DUAL-MODE (TREE + DEEP + KERNEL)
================================================================================
Provides:
    - TreeSHAP for XGBoost (exact, O(TLD) complexity)
    - DeepSHAP for CNN/Transformer via TF gradient integration
    - KernelSHAP as model-agnostic fallback
    - Global explanations: feature importance rankings
    - Local explanations: per-alert waterfall data
    - Batch processing with LRU caching for SOC throughput
================================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Any


class SHAPExplainer:
    """
    Unified SHAP explanation engine supporting multiple backends.

    Parameters
    ----------
    model : object
        The model to explain. Must have `predict_proba(X)`.
    mode : str
        'tree' for XGBoost, 'deep' for TF/Keras, 'kernel' for any model.
    background_data : np.ndarray
        Background dataset for KernelSHAP / DeepSHAP (subset of training data).
    feature_names : list of str, optional
        Human-readable feature names for reports.
    """

    def __init__(
        self,
        model,
        mode: str = "kernel",
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.mode = mode
        self.background_data = background_data
        self.feature_names = feature_names
        self._explainer = None
        self._shap_cache: Dict[int, np.ndarray] = {}

    def _build_explainer(self):
        """Lazily construct the appropriate SHAP explainer."""
        if self._explainer is not None:
            return

        try:
            import shap

            if self.mode == "tree":
                self._explainer = shap.TreeExplainer(self.model)
            elif self.mode == "deep":
                self._explainer = shap.DeepExplainer(self.model, self.background_data)
            elif self.mode == "kernel":
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba, self.background_data
                )
            else:
                raise ValueError(f"Unknown SHAP mode: {self.mode}")
        except ImportError:
            # Fallback: manual permutation-based approximation
            self._explainer = _PermutationSHAP(self.model, self.background_data)

    # ------------------------------------------------------------------
    # Local explanations
    # ------------------------------------------------------------------

    def explain_instance(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Returns
        -------
        dict with keys:
            'shap_values': np.ndarray of shape (n_features,)
            'base_value': float (expected prediction)
            'prediction': float
            'top_features': list of (feature_name, shap_value) sorted by |value|
        """
        self._build_explainer()
        x = np.atleast_2d(x)

        # Check cache
        key = hash(x.tobytes())
        if key in self._shap_cache:
            shap_vals = self._shap_cache[key]
        else:
            shap_vals = self._compute_shap(x)[0]
            self._shap_cache[key] = shap_vals

        # Handle multi-output: take class-1 (malicious) values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        pred = self.model.predict_proba(x)[0, 1]
        base = self._get_base_value()

        names = self.feature_names or [f"f{i}" for i in range(len(shap_vals))]
        top = sorted(
            zip(names, shap_vals.tolist()), key=lambda t: abs(t[1]), reverse=True
        )

        return {
            "shap_values": shap_vals,
            "base_value": base,
            "prediction": float(pred),
            "top_features": top[:10],
        }

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        self._build_explainer()
        shap_vals = self._compute_shap(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1
        return shap_vals

    # ------------------------------------------------------------------
    # Global explanations
    # ------------------------------------------------------------------

    def global_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Mean absolute SHAP value per feature (global importance ranking).
        """
        vals = self.explain_batch(X)
        mean_abs = np.mean(np.abs(vals), axis=0)
        names = self.feature_names or [f"f{i}" for i in range(vals.shape[1])]
        importance = dict(
            sorted(zip(names, mean_abs.tolist()), key=lambda t: t[1], reverse=True)
        )
        return importance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_shap(self, X):
        if isinstance(self._explainer, _PermutationSHAP):
            return self._explainer.shap_values(X)
        return self._explainer.shap_values(X)

    def _get_base_value(self):
        if hasattr(self._explainer, "expected_value"):
            ev = self._explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                return float(ev[1]) if len(ev) > 1 else float(ev[0])
            return float(ev)
        return 0.5  # default


class _PermutationSHAP:
    """
    Lightweight permutation-based SHAP approximation.
    Used when the `shap` library is not installed.
    """

    def __init__(self, model, background, n_samples=50):
        self.model = model
        self.background = background
        self.n_samples = (
            min(n_samples, len(background)) if background is not None else 50
        )
        self.expected_value = None

    def shap_values(self, X):
        X = np.atleast_2d(X)
        n_points, n_features = X.shape

        if self.background is None:
            return np.zeros((n_points, n_features))

        bg = self.background[: self.n_samples]
        base_preds = self.model.predict_proba(bg)[:, 1]
        self.expected_value = [float(1 - base_preds.mean()), float(base_preds.mean())]

        all_shap = np.zeros((n_points, n_features))

        for i in range(n_points):
            for j in range(n_features):
                # Marginal contribution of feature j
                X_with = bg.copy()
                X_with[:, j] = X[i, j]
                preds_with = self.model.predict_proba(X_with)[:, 1]
                all_shap[i, j] = float(preds_with.mean() - base_preds.mean())

        return all_shap
