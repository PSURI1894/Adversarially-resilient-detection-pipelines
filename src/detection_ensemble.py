"""
================================================================================
IDS ENSEMBLE SYSTEM - RESEARCH-GRADE ARCHITECTURE
================================================================================
Project: Adversarially Resilient SOC Pipeline
Component: Person 2 (Detection & Learning Lead)

Hybrid ensemble of XGBoost + adversarially trained 1D-CNN
Designed for calibrated probability output (Person 3 compatible)
================================================================================
"""

import logging
import datetime
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# ==============================================================================
# ERRORS
# ==============================================================================

class EnsembleError(Exception):
    pass

class DataShapeMismatchError(EnsembleError):
    pass

# ==============================================================================
# METRICS / AUDIT
# ==============================================================================

class ModelAuditor:
    """SOC-grade audit metrics (no accuracy lies)."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ModelAuditor")
        self.history: List[Dict] = []

    @staticmethod
    def false_discovery_rate(y_true, y_pred) -> float:
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return fp / (fp + tp + 1e-9)

    def run_audit(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        y_pred = (y_prob > 0.5).astype(int)

        report = {
            "timestamp": str(datetime.datetime.now()),
            "f1": f1_score(y_true, y_pred),
            "fdr": self.false_discovery_rate(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
        }

        self.history.append(report)
        self.logger.info(f"AUDIT | F1={report['f1']:.4f} FDR={report['fdr']:.4f}")
        return report

# ==============================================================================
# ADVERSARIAL TRAINER (CNN)
# ==============================================================================

class ResilientTrainer:
    """
    Feature-aware adversarial training for tabular CNNs.
    Backward-compatible with test expectations.
    """

    def __init__(self, model_type_or_dim, config: Optional[Dict] = None):
        # === Backward compatibility ===
        if isinstance(model_type_or_dim, str):
            # Called as ResilientTrainer("CNN", {"input_dim": N})
            if config is None or "input_dim" not in config:
                raise ValueError("config with 'input_dim' required")
            self.model_type = model_type_or_dim
            self.input_dim = config["input_dim"]
        else:
            # Called as ResilientTrainer(input_dim)
            self.model_type = "CNN"
            self.input_dim = model_type_or_dim

        self.model = self._build()

    def _build(self):
        model = models.Sequential([
            layers.Input(shape=(self.input_dim, 1)),
            layers.Conv1D(64, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, eps: float = 0.05, epochs: int = 8):
        """
        Adversarial loop with scale-aware noise.
        """
        scale = np.std(X, axis=0, keepdims=True) + 1e-6

        for _ in range(epochs):
            noise = np.random.normal(0, eps * scale, size=X.shape)
            X_adv = X + noise

            X_mix = np.vstack([X, X_adv])
            y_mix = np.concatenate([y, y])

            self.model.fit(
                X_mix.reshape(-1, self.input_dim, 1),
                y_mix,
                epochs=1,
                batch_size=64,
                verbose=0
            )

# ==============================================================================
# ENSEMBLE ORCHESTRATOR
# ==============================================================================

class EnsembleOrchestrator(BaseEstimator, ClassifierMixin):
    """
    Person-2 final deliverable.
    Soft-voting ensemble compatible with conformal prediction.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.xgb = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False
        )
        self.nn = ResilientTrainer(input_dim)

    def _check_shape(self, X):
        if X.shape[1] != self.input_dim:
            raise DataShapeMismatchError(
                f"Expected {self.input_dim} features, got {X.shape[1]}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._check_shape(X)
        self.xgb.fit(X, y)
        self.nn.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_shape(X)

        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_nn = self.nn.model.predict(
            X.reshape(-1, self.input_dim, 1),
            verbose=0
        ).flatten()

        prob = 0.6 * p_xgb + 0.4 * p_nn
        return np.vstack([1 - prob, prob]).T
    
    # Backward compatibility for tests
    def fit_adversarially(self, X, y, eps=0.05, epochs=1):
        return self.fit(X, y, eps=eps, epochs=epochs)


# ==============================================================================
# EXECUTION GUARD (DO NOT TRAIN HERE)
# ==============================================================================

if __name__ == "__main__":
    print("[INFO] detection_ensemble.py loaded.")
    print("[INFO] This module is imported by main_pipeline.py — not run directly.")
