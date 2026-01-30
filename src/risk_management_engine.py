"""
================================================================================
IDS RISK MANAGEMENT & UNCERTAINTY QUANTIFICATION ENGINE
================================================================================
Manual Split Conformal Prediction + Risk Thermostat
Optimized, robust, environment-safe implementation
================================================================================
"""

import os
import numpy as np
from enum import Enum
from typing import List, Dict
import matplotlib.pyplot as plt

# ==============================================================================
# ENUMS & EXCEPTIONS
# ==============================================================================

class SOCState(Enum):
    STABLE = 1
    SUSPICIOUS = 2
    EVASION_LOCKED = 3
    FAILURE = 4

class CalibrationError(Exception):
    pass

# ==============================================================================
# CONFORMAL ENGINE (MANUAL SPLIT CP)
# ==============================================================================

class ConformalEngine:
    """
    Manual split conformal prediction using Least Ambiguous Classifier (LAC).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat: float | None = None
        self.calibrated = False

    def calibrate(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        if len(y_cal) < 1000:
            raise CalibrationError("Calibration set too small for guarantees")

        probs = model.predict_proba(X_cal)
        true_probs = probs[np.arange(len(y_cal)), y_cal]
        scores = 1.0 - true_probs

        n = scores.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method="higher")

        self.calibrated = True

    def prediction_sets(self, X: np.ndarray, model) -> List[List[int]]:
        if not self.calibrated or self.q_hat is None:
            raise CalibrationError("Conformal engine not calibrated")

        probs = model.predict_proba(X)
        pred_sets: List[List[int]] = []

        for p in probs:
            labels = [cls for cls in (0, 1) if (1.0 - p[cls]) <= self.q_hat]
            if not labels:
                labels = [0, 1]
            pred_sets.append(labels)

        return pred_sets

    # ------------------------------------------------------------------
    # Backward compatibility for tests
    # ------------------------------------------------------------------
    def get_prediction_sets(self, probs: np.ndarray):
        if self.q_hat is None:
            raise CalibrationError("Engine not calibrated")

        sets = []
        for p in probs:
            labels = [cls for cls in (0, 1) if (1.0 - p[cls]) <= self.q_hat]
            if not labels:
                labels = [0, 1]
            sets.append(labels)
        return sets

# ==============================================================================
# RISK THERMOSTAT (FSM CONTROLLER)
# ==============================================================================

class RiskThermostat:
    """
    Finite-state SOC controller driven by uncertainty + alert debt.
    """

    def __init__(
        self,
        analyst_capacity: int = 50,
        warning_threshold: float = 1.1,
        critical_threshold: float = 1.8
    ):
        self.state = SOCState.STABLE
        self.analyst_capacity = analyst_capacity
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.alert_debt = 0
        self.uncertainty_history: List[float] = []

    # ------------------------------------------------------------------
    # Backward compatibility for tests
    # ------------------------------------------------------------------
    def evaluate_risk(self, prediction_sets):
        return self.evaluate(prediction_sets)

    def evaluate(self, prediction_sets: List[List[int]]) -> SOCState:
        sizes = np.fromiter((len(s) for s in prediction_sets), dtype=float)
        avg_uncertainty = float(sizes.mean())

        self.uncertainty_history.append(avg_uncertainty)

        uncertain_count = int(np.sum(sizes > 1))
        self.alert_debt += uncertain_count

        # IMPORTANT: alert overload takes priority over failure
        if self.alert_debt > self.analyst_capacity:
            self.state = SOCState.EVASION_LOCKED
        elif avg_uncertainty > self.critical_threshold:
            self.state = SOCState.FAILURE
        elif avg_uncertainty > self.warning_threshold:
            self.state = SOCState.SUSPICIOUS
        else:
            self.state = SOCState.STABLE

        return self.state

    def playbook(self) -> Dict[str, str]:
        return {
            SOCState.STABLE: {"action": "Auto-block"},
            SOCState.SUSPICIOUS: {"action": "Analyst review"},
            SOCState.EVASION_LOCKED: {"action": "Throttle + Honeypot"},
            SOCState.FAILURE: {"action": "Fail-safe shutdown"},
        }[self.state]

# ==============================================================================
# DASHBOARD / VISUALIZATION
# ==============================================================================

class SOCDashboard:
    """
    Lightweight uncertainty visualization for SOC leads.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def plot_uncertainty(self, history: List[float]):
        plt.figure(figsize=(10, 4))
        plt.plot(history, linewidth=2, label="Avg Prediction Set Size")
        plt.axhline(1.1, linestyle="--", color="orange", label="Warning")
        plt.axhline(1.8, linestyle="--", color="red", label="Critical")
        plt.xlabel("Batch Index")
        plt.ylabel("Uncertainty")
        plt.title("SOC Risk Thermostat – Uncertainty")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "risk_thermostat.png"))
        plt.close()
# ==============================================================================