"""
================================================================================
ADAPTIVE RETRAINING PIPELINE
================================================================================
<<<<<<< HEAD
Triggered on drift detection. Selects retraining window, active learning,
warm-starts model weights, and validates against performance metrics.
================================================================================
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class AdaptiveRetrainingPipeline:
    """
    Handles retraining ensemble models when concept drift is detected.
    """
    
    def __init__(self, ensemble_orchestrator: Any, 
                 validation_gate: float = 0.02, # 2% F1 improvement required to promote
                 active_learning_strategy: str = "uncertainty"):
        self.orchestrator = ensemble_orchestrator
        self.validation_gate = validation_gate
        self.strategy = active_learning_strategy
        self.history = []
        
    def retrain(self, current_model: Any, 
                X_train_new: np.ndarray, 
                y_train_new: np.ndarray,
                X_holdout: np.ndarray,
                y_holdout: np.ndarray) -> Any:
        """
        Retrain the model on new data.
        
        Steps:
        1. Select most informative samples (Active Learning)
        2. Retrain model (Warm-start)
        3. Validate against current model
        4. Promote or rollback
        """
        logger.info("Adaptive retraining triggered...")
        
        # 1. Active Learning (Sample Selection)
        if self.strategy == "uncertainty":
            # Select samples where model is most uncertain (prediction set size > 1)
            # For simplicity, we just use the entire new batch for now
            X_al, y_al = X_train_new, y_train_new
        else:
            X_al, y_al = X_train_new, y_train_new
        
        # 2. Perform Retraining
        # (Assuming orchestrator supports fit or warm-start)
        # For ensemble, we fit on the new data
        self.orchestrator.fit(X_al, y_al)
        
        # 3. Validation Gate
        # Evaluate performance on holdout data
        # (Assuming orchestrator supports evaluate or score)
        # Using a placeholder score for now
        performance_new = self._evaluate_model(self.orchestrator, X_holdout, y_holdout)
        performance_old = self._evaluate_model(current_model, X_holdout, y_holdout)
        
        gain = performance_new - performance_old
        logger.info(f"Retraining gain (F1): {gain:.4f}")
        
        if gain >= self.validation_gate:
            logger.info("Retraining successful! Promoting new model.")
            return self.orchestrator
        else:
            logger.warning("Validation gate failed. Rolling back to current model.")
            # Rollback: reinstate current_model weights to orchestrator
            # (Assuming orchestrator supports load/state sync)
            return current_model

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Helper to compute model F1-score/performance."""
        # This should call a metric evaluation layer (e.g., from src.models)
        # Simple accuracy for now
        probs = model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == y))
=======
Triggered on drift detection. Selects retraining window, performs
uncertainty-based active learning, warm-starts model weights, and
validates against performance metrics before promotion.

Optimisations over v1:
    - Real uncertainty-based active learning sample selection
    - F1-score evaluation (not raw accuracy)
    - Retraining history tracking with full metrics
    - Exponential decay and sliding window selection strategies
    - Automatic rollback on regression
================================================================================
"""

import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class AdaptiveRetrainingPipeline:
    """
    Handles retraining ensemble models when concept drift is detected.

    Parameters
    ----------
    ensemble_orchestrator : object
        The ensemble model to retrain (must support fit / predict_proba).
    validation_gate : float
        Minimum F1-score improvement required to promote the new model.
    active_learning_strategy : str
        Sample selection strategy: 'uncertainty', 'random', or 'all'.
    uncertainty_percentile : float
        Percentile threshold for uncertainty-based selection (0-100).
        Samples above this uncertainty percentile are selected.
    max_retrain_samples : int or None
        Maximum number of samples to use for retraining.
    """

    def __init__(self, ensemble_orchestrator,
                 validation_gate: float = 0.02,
                 active_learning_strategy: str = "uncertainty",
                 uncertainty_percentile: float = 70.0,
                 max_retrain_samples: Optional[int] = None):
        self.orchestrator = ensemble_orchestrator
        self.validation_gate = validation_gate
        self.strategy = active_learning_strategy
        self.uncertainty_percentile = uncertainty_percentile
        self.max_retrain_samples = max_retrain_samples
        self.history: List[Dict[str, Any]] = []

    def retrain(self, current_model,
                X_train_new: np.ndarray,
                y_train_new: np.ndarray,
                X_holdout: np.ndarray,
                y_holdout: np.ndarray):
        """
        Retrain the model on new data with validation gate.

        Steps:
        1. Select most informative samples (Active Learning)
        2. Retrain model (warm-start via fit on selected data)
        3. Validate new model vs current model on holdout set
        4. Promote if validation gate passes, else rollback

        Returns the promoted model or the original current_model.
        """
        t0 = time.time()
        logger.info("Adaptive retraining triggered...")

        # 1. Active Learning (Sample Selection)
        X_al, y_al = self._select_samples(
            current_model, X_train_new, y_train_new
        )
        logger.info(f"Active learning selected {len(X_al)}/{len(X_train_new)} samples "
                     f"(strategy={self.strategy})")

        # 2. Evaluate baseline performance
        f1_old = self._evaluate_f1(current_model, X_holdout, y_holdout)

        # 3. Retrain
        self.orchestrator.fit(X_al, y_al)

        # 4. Evaluate new model
        f1_new = self._evaluate_f1(self.orchestrator, X_holdout, y_holdout)
        gain = f1_new - f1_old

        # 5. Record history
        record = {
            "timestamp": time.time(),
            "duration_s": time.time() - t0,
            "n_samples_available": len(X_train_new),
            "n_samples_selected": len(X_al),
            "strategy": self.strategy,
            "f1_old": round(f1_old, 4),
            "f1_new": round(f1_new, 4),
            "gain": round(gain, 4),
            "promoted": gain >= self.validation_gate,
        }
        self.history.append(record)

        # 6. Gate decision
        if gain >= self.validation_gate:
            logger.info(f"Retraining promoted: F1 {f1_old:.4f} → {f1_new:.4f} "
                         f"(+{gain:.4f} ≥ gate {self.validation_gate})")
            return self.orchestrator
        else:
            logger.warning(f"Retraining rolled back: F1 gain {gain:.4f} "
                            f"< gate {self.validation_gate}")
            return current_model

    def _select_samples(self, model, X: np.ndarray, y: np.ndarray):
        """Select training samples based on the active learning strategy."""
        if self.strategy == "all" or len(X) == 0:
            return X, y

        if self.strategy == "uncertainty":
            return self._uncertainty_sampling(model, X, y)
        elif self.strategy == "random":
            return self._random_sampling(X, y)
        else:
            return X, y

    def _uncertainty_sampling(self, model, X: np.ndarray, y: np.ndarray):
        """Select samples where model is most uncertain (high entropy)."""
        try:
            probs = model.predict_proba(X)
            # Compute entropy of prediction distribution
            eps = 1e-10
            entropy = -np.sum(probs * np.log(probs + eps), axis=1)

            threshold = np.percentile(entropy, self.uncertainty_percentile)
            mask = entropy >= threshold

            # Ensure we have at least some samples
            if mask.sum() < 10:
                return X, y

            X_sel, y_sel = X[mask], y[mask]

            if self.max_retrain_samples and len(X_sel) > self.max_retrain_samples:
                idx = np.random.choice(len(X_sel), self.max_retrain_samples, replace=False)
                return X_sel[idx], y_sel[idx]

            return X_sel, y_sel
        except Exception as e:
            logger.warning(f"Uncertainty sampling failed ({e}), using all samples")
            return X, y

    def _random_sampling(self, X: np.ndarray, y: np.ndarray):
        """Random subsampling."""
        n = self.max_retrain_samples or len(X)
        n = min(n, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        return X[idx], y[idx]

    def _evaluate_f1(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Compute macro F1-score on the holdout set."""
        try:
            probs = model.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            return float(f1_score(y, preds, average="macro", zero_division=0))
        except Exception as e:
            logger.warning(f"F1 evaluation failed ({e}), returning 0.0")
            return 0.0

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full retraining history."""
        return list(self.history)

    @property
    def n_retrains(self) -> int:
        return len(self.history)

    @property
    def n_promotions(self) -> int:
        return sum(1 for r in self.history if r["promoted"])
>>>>>>> ef9ed20b07fceb0d5330bca702b5d7f7559e786e
