"""
================================================================================
IDS RISK MANAGEMENT & UNCERTAINTY QUANTIFICATION ENGINE — v2
================================================================================
Features:
    ✓ ConformalBackend enum selecting SPLIT / RSCP / RSCP_PLUS / ONLINE / ADAPTIVE
    ✓ ConformalEngine as thin adapter delegating to src/conformal/ backends
    ✓ RiskThermostat v2 with hysteresis, cooldown timers, multi-signal FSM,
      and continuous 0-100 severity scoring
    ✓ Full backward compatibility via legacy evaluate_risk / playbook API
================================================================================
"""

import os
import time
import numpy as np
from enum import Enum
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from src.conformal.rscp import RandomizedSmoothedCP
from src.conformal.multi_class_cp import AdaptiveConformalPredictor
from src.conformal.online_cp import OnlineConformalPredictor


# ==============================================================================
# ENUMS & EXCEPTIONS
# ==============================================================================


class SOCState(Enum):
    STABLE = 1
    SUSPICIOUS = 2
    EVASION_LOCKED = 3
    FAILURE = 4


class ConformalBackend(Enum):
    SPLIT = "split"
    RSCP = "rscp"
    RSCP_PLUS = "rscp_plus"
    ONLINE = "online"
    ADAPTIVE = "adaptive"


class CalibrationError(Exception):
    pass


# ==============================================================================
# CONFORMAL ENGINE — THIN ADAPTER
# ==============================================================================


class ConformalEngine:
    """
    Unified adapter that delegates to the appropriate conformal backend.

    Maintains backward compatibility: the default SPLIT backend replicates
    the original manual split conformal prediction behavior exactly.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    backend : ConformalBackend
        Which conformal method to use.
    backend_kwargs : dict
        Extra kwargs forwarded to the selected backend constructor.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        backend: ConformalBackend = ConformalBackend.SPLIT,
        **backend_kwargs,
    ):
        self.alpha = alpha
        self.backend_type = backend
        self.calibrated = False

        if backend == ConformalBackend.SPLIT:
            self._backend = _SplitCP(alpha)
        elif backend == ConformalBackend.RSCP:
            self._backend = RandomizedSmoothedCP(
                alpha=alpha, ptt=False, **backend_kwargs
            )
        elif backend == ConformalBackend.RSCP_PLUS:
            self._backend = RandomizedSmoothedCP(
                alpha=alpha, ptt=True, **backend_kwargs
            )
        elif backend == ConformalBackend.ONLINE:
            self._backend = OnlineConformalPredictor(alpha=alpha, **backend_kwargs)
        elif backend == ConformalBackend.ADAPTIVE:
            self._backend = AdaptiveConformalPredictor(alpha=alpha, **backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate the selected backend."""
        if self.backend_type == ConformalBackend.ONLINE:
            # Online CP calibrates via streaming; seed it with initial batch
            self._backend.update_batch(model, X_cal, y_cal)
        else:
            self._backend.calibrate(model, X_cal, y_cal)
        self.calibrated = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def prediction_sets(self, X: np.ndarray, model) -> List[List[int]]:
        if not self.calibrated:
            raise CalibrationError("Conformal engine not calibrated")
        return self._backend.prediction_sets(X, model)

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    @property
    def q_hat(self):
        return getattr(self._backend, "q_hat", None)

    @q_hat.setter
    def q_hat(self, value):
        self._backend.q_hat = value

    def get_prediction_sets(self, probs: np.ndarray):
        """Legacy API: takes probabilities directly (SPLIT backend only)."""
        if not isinstance(self._backend, _SplitCP):
            raise CalibrationError("get_prediction_sets only works with SPLIT backend")
        return self._backend.get_prediction_sets(probs)


class _SplitCP:
    """
    Internal vanilla split conformal prediction (LAC).
    Extracted to keep ConformalEngine clean.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.q_hat: Optional[float] = None

    def calibrate(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        if len(y_cal) < 50:
            raise CalibrationError("Calibration set too small for guarantees")

        probs = model.predict_proba(X_cal)
        true_probs = probs[np.arange(len(y_cal)), y_cal]
        scores = 1.0 - true_probs

        n = scores.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level, method="higher")

    def prediction_sets(self, X: np.ndarray, model) -> List[List[int]]:
        if self.q_hat is None:
            raise CalibrationError("Not calibrated")

        probs = model.predict_proba(X)
        return self._sets_from_probs(probs)

    def get_prediction_sets(self, probs: np.ndarray):
        if self.q_hat is None:
            raise CalibrationError("Not calibrated")
        return self._sets_from_probs(probs)

    def _sets_from_probs(self, probs):
        pred_sets = []
        for p in probs:
            labels = [cls for cls in (0, 1) if (1.0 - p[cls]) <= self.q_hat]
            if not labels:
                labels = [0, 1]
            pred_sets.append(labels)
        return pred_sets


# ==============================================================================
# RISK THERMOSTAT v2 — MULTI-SIGNAL FSM WITH HYSTERESIS
# ==============================================================================


class RiskThermostat:
    """
    Finite-state SOC controller driven by multiple signals.

    v2 improvements over v1:
        - **Multi-signal input**: uncertainty + alert debt + calibration drift
          + model confidence disagreement
        - **Hysteresis**: state transitions require sustained threshold breach
          (prevents flapping)
        - **Cooldown timers**: minimum time in each state before transition
        - **Severity scoring**: continuous 0-100 risk score alongside discrete state

    Parameters
    ----------
    analyst_capacity : int
        SOC analyst queue capacity before overload.
    warning_threshold : float
        Average set size triggering SUSPICIOUS.
    critical_threshold : float
        Average set size triggering FAILURE.
    hysteresis_steps : int
        Number of consecutive evaluations a threshold must be breached
        before the state actually transitions.
    cooldown_seconds : float
        Minimum seconds in a state before allowing transition.
    """

    def __init__(
        self,
        analyst_capacity: int = 50,
        warning_threshold: float = 1.1,
        critical_threshold: float = 1.8,
        hysteresis_steps: int = 3,
        cooldown_seconds: float = 0.0,
    ):
        self.state = SOCState.STABLE
        self.analyst_capacity = analyst_capacity
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.hysteresis_steps = hysteresis_steps
        self.cooldown_seconds = cooldown_seconds

        self.alert_debt: float = 0.0
        self.uncertainty_history: List[float] = []
        self.severity_history: List[float] = []

        # Hysteresis tracking: count consecutive steps proposing each state
        self._proposed_state_counts: Dict[SOCState, int] = {s: 0 for s in SOCState}
        self._last_transition_time: float = time.time()

        # Multi-signal registers
        self.calibration_drift_score: float = 0.0
        self.disagreement_score: float = 0.0

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def evaluate_risk(self, prediction_sets):
        return self.evaluate(prediction_sets)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prediction_sets: List[List[int]],
        calibration_drift: float = 0.0,
        disagreement: float = 0.0,
    ) -> SOCState:
        """
        Evaluate the current SOC state from multiple signals.

        Parameters
        ----------
        prediction_sets : list of lists
            Conformal prediction sets for the current batch.
        calibration_drift : float
            KS-stat or equivalent drift measure from CalibrationIntegrityMonitor.
        disagreement : float
            Model confidence disagreement (e.g., max variance across ensemble).
        """
        sizes = np.fromiter((len(s) for s in prediction_sets), dtype=float)
        avg_uncertainty = float(sizes.mean())

        self.uncertainty_history.append(avg_uncertainty)
        self.calibration_drift_score = calibration_drift
        self.disagreement_score = disagreement

        uncertain_count = int(np.sum(sizes > 1))
        self.alert_debt += uncertain_count

        # ----- Compute severity score (0-100) -----
        severity = self._compute_severity(
            avg_uncertainty, calibration_drift, disagreement
        )
        self.severity_history.append(severity)

        # ----- Determine proposed state -----
        proposed = self._propose_state(avg_uncertainty)

        # ----- Apply hysteresis -----
        if proposed != self.state:
            self._proposed_state_counts[proposed] += 1
            # Reset counters for other non-current states
            for s in SOCState:
                if s != proposed and s != self.state:
                    self._proposed_state_counts[s] = 0

            # Check if hysteresis threshold is met AND cooldown has elapsed
            elapsed = time.time() - self._last_transition_time
            if (
                self._proposed_state_counts[proposed] >= self.hysteresis_steps
                and elapsed >= self.cooldown_seconds
            ):
                self.state = proposed
                self._last_transition_time = time.time()
                # Reset all counters
                for s in SOCState:
                    self._proposed_state_counts[s] = 0
        else:
            # Already in the proposed state — reset all other counters
            for s in SOCState:
                if s != self.state:
                    self._proposed_state_counts[s] = 0

        return self.state

    def _propose_state(self, avg_uncertainty: float) -> SOCState:
        """Determine which state the signals are proposing."""
        # Alert overload takes priority
        if self.alert_debt > self.analyst_capacity:
            return SOCState.EVASION_LOCKED
        elif avg_uncertainty > self.critical_threshold:
            return SOCState.FAILURE
        elif avg_uncertainty > self.warning_threshold:
            return SOCState.SUSPICIOUS
        else:
            return SOCState.STABLE

    def _compute_severity(
        self, avg_uncertainty: float, calibration_drift: float, disagreement: float
    ) -> float:
        """
        Continuous 0-100 risk severity score fusing multiple signals.

        Weights: uncertainty (50%), alert debt ratio (20%),
                 calibration drift (15%), disagreement (15%).
        """
        # Normalize uncertainty: 1.0 → 0, critical_threshold → 80
        u_norm = np.clip(
            (avg_uncertainty - 1.0) / (self.critical_threshold - 1.0) * 80, 0, 100
        )

        # Alert debt ratio (0-100)
        debt_ratio = np.clip(
            self.alert_debt / (self.analyst_capacity + 1e-8) * 100, 0, 100
        )

        # Drift (assumed 0-1 range, scale to 0-100)
        drift_norm = np.clip(calibration_drift * 100, 0, 100)

        # Disagreement (assumed 0-1 range, scale to 0-100)
        disagree_norm = np.clip(disagreement * 100, 0, 100)

        severity = (
            0.50 * u_norm + 0.20 * debt_ratio + 0.15 * drift_norm + 0.15 * disagree_norm
        )

        return float(np.clip(severity, 0, 100))

    # ------------------------------------------------------------------
    # Playbook & diagnostics
    # ------------------------------------------------------------------

    def playbook(self) -> Dict[str, str]:
        return {
            SOCState.STABLE: {"action": "Auto-block"},
            SOCState.SUSPICIOUS: {"action": "Analyst review"},
            SOCState.EVASION_LOCKED: {"action": "Throttle + Honeypot"},
            SOCState.FAILURE: {"action": "Fail-safe shutdown"},
        }[self.state]

    @property
    def severity(self) -> float:
        """Current severity score."""
        return self.severity_history[-1] if self.severity_history else 0.0

    def get_diagnostics(self) -> Dict:
        return {
            "state": self.state.name,
            "severity": self.severity,
            "alert_debt": self.alert_debt,
            "calibration_drift": self.calibration_drift_score,
            "disagreement": self.disagreement_score,
            "n_evaluations": len(self.uncertainty_history),
        }


# ==============================================================================
# DASHBOARD / VISUALIZATION
# ==============================================================================


class SOCDashboard:
    """Lightweight uncertainty visualization for SOC leads."""

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

    def plot_severity(self, severity_history: List[float]):
        plt.figure(figsize=(10, 4))
        plt.fill_between(
            range(len(severity_history)), severity_history, alpha=0.4, color="crimson"
        )
        plt.plot(severity_history, linewidth=2, color="crimson", label="Severity Score")
        plt.axhline(30, linestyle="--", color="orange", alpha=0.7, label="Warning")
        plt.axhline(70, linestyle="--", color="red", alpha=0.7, label="Critical")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Severity (0-100)")
        plt.title("SOC Risk Severity Timeline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "severity_timeline.png"))
        plt.close()
