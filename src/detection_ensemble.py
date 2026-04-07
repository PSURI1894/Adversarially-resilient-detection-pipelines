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
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass

from src.models import TabTransformer, VAIDS, DeepEnsemble

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
        model = models.Sequential(
            [
                layers.Input(shape=(self.input_dim, 1)),
                layers.Conv1D(64, 3, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.GlobalAveragePooling1D(),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
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
                verbose=0,
            )


# ==============================================================================
# ENSEMBLE ORCHESTRATOR
# ==============================================================================


@dataclass
class EnsembleConfig:
    input_dim: int
    xgb: dict = None
    transformer: dict = None
    vae: dict = None
    deep_ensemble: dict = None
    weights: dict = None


class LegacyEnsembleOrchestrator(BaseEstimator, ClassifierMixin):
    """
    Backward compatibility wrapper.
    Soft-voting ensemble compatible with conformal prediction.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.xgb = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
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
            X.reshape(-1, self.input_dim, 1), verbose=0
        ).flatten()
        prob = 0.6 * p_xgb + 0.4 * p_nn
        return np.vstack([1 - prob, prob]).T

    def fit_adversarially(self, X, y, eps=0.05, epochs=1):
        return self.fit(X, y, eps=eps, epochs=epochs)


class EnsembleOrchestrator(BaseEstimator, ClassifierMixin):
    """
    Pluggable registry-based Deep Ensemble with Stacking Meta-Learner.
    """

    def __init__(self, config_or_dim=None, *, input_dim=None):
        if input_dim is not None and config_or_dim is None:
            config_or_dim = input_dim
        if isinstance(config_or_dim, int):
            self.config = EnsembleConfig(input_dim=config_or_dim)
        else:
            self.config = config_or_dim

        self.input_dim = self.config.input_dim

        # Pluggable model registry
        self.models = {
            "xgboost": xgb.XGBClassifier(
                **(self.config.xgb or {"eval_metric": "logloss"})
            ),
            "cnn": ResilientTrainer(self.input_dim),
            "transformer": TabTransformer(
                self.input_dim, **(self.config.transformer or {})
            ),
            "vae": VAIDS(self.input_dim, **(self.config.vae or {})),
            "deep_ensemble": DeepEnsemble(
                self.input_dim, **(self.config.deep_ensemble or {})
            ),
        }

        self.stacker = LogisticRegression()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        print("[INFO] Training base models...")
        # Train base models on the full dataset (except VAE, which trains unsupervised)
        self.models["xgboost"].fit(X, y)
        self.models["cnn"].fit(X, y)

        tf_X = tf.constant(X, dtype=tf.float32)
        tf_y = tf.constant(y, dtype=tf.float32)

        self.models["transformer"].compile(optimizer="adam", loss="binary_crossentropy")
        self.models["transformer"].fit(tf_X, tf_y, epochs=3, batch_size=256, verbose=0)

        # VAE is unsupervised on benign traffic (y==0)
        benign_X = X[y == 0]
        if len(benign_X) > 0:
            self.models["vae"].compile(optimizer="adam")
            self.models["vae"].fit(benign_X, epochs=3, batch_size=256, verbose=0)

        self.models["deep_ensemble"].fit(X, y)

        print("[INFO] Training stacking meta-learner...")
        # For a true stacker, we would use out-of-fold predictions.
        # For simplicity here, we train stacker on training data predictions directly,
        # but in a complete setup we'd use k-fold cross_val_predict.
        # We simulate the features for the logistic regression:
        preds = self._get_base_predictions(X)
        self.stacker.fit(preds, y)
        self.is_fitted = True
        return self

    def _get_base_predictions(self, X: np.ndarray):
        p_xgb = self.models["xgboost"].predict_proba(X)[:, 1]
        p_cnn = (
            self.models["cnn"]
            .model.predict(X.reshape(-1, self.input_dim, 1), verbose=0)
            .flatten()
        )
        p_trans = self.models["transformer"].predict_proba(
            tf.constant(X, dtype=tf.float32)
        )[:, 1]

        # VAE pseudo-probabilities
        try:
            p_vae = self.models["vae"].predict_proba(X)[:, 1]
        except BaseException:
            # Fallback if VAE not properly fitted
            p_vae = np.zeros(len(X))

        # Deep Ensemble
        p_de = self.models["deep_ensemble"].predict_proba(X)[:, 1]

        return np.column_stack([p_xgb, p_cnn, p_trans, p_vae, p_de])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise Exception("Model not fitted.")
        base_preds = self._get_base_predictions(X)
        return self.stacker.predict_proba(base_preds)


# ==============================================================================
# EXECUTION GUARD (DO NOT TRAIN HERE)
# ==============================================================================

if __name__ == "__main__":
    print("[INFO] detection_ensemble.py loaded.")
    print("[INFO] This module is imported by main_pipeline.py — not run directly.")
