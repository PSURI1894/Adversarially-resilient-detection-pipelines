"""
================================================================================
FULLY ENHANCED SIMULATION ENGINE – ADAPTIVE ADVERSARIAL SOC
================================================================================
Includes:
✓ Attack strength → SOC phase plots
✓ Alert-debt decay
✓ Adaptive retraining
✓ Feature-targeted evasion
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from src.utils import get_logger, ensure_directories
from src.detection_ensemble import EnsembleOrchestrator
from src.risk_management_engine import ConformalEngine, RiskThermostat, SOCState

# ==============================================================================
# CONFIG
# ==============================================================================

DATA_PATH = "data/processed/processed_flows.csv"
RESULTS_PATH = "reports/audit_logs/simulation_results.csv"
PLOT_PATH = "reports/figures/attack_phase_transition.png"

RANDOM_STATE = 42
SUBSAMPLE = 200_000
BATCH_SIZE = 10_000

EVASION_LEVELS = [0.0, 0.05, 0.10, 0.20]
POISON_LEVELS = [0.0, 0.01, 0.03]

ALPHA = 0.05
ANALYST_CAPACITY = 50
ALERT_DECAY = 0.4          # analyst clears 40% debt per batch
RETRAIN_WINDOW = 50_000    # samples used for adaptive retraining

np.random.seed(RANDOM_STATE)

# ==============================================================================
# ATTACK OPERATORS
# ==============================================================================

def feature_targeted_evasion(X: np.ndarray, eps: float, feature_idx: List[int]) -> np.ndarray:
    """
    Perturbs only attacker-controllable flow features.
    """
    X_adv = X.copy()
    scale = np.std(X_adv[:, feature_idx], axis=0, keepdims=True) + 1e-6
    noise = np.random.normal(0, eps * scale, size=(X_adv.shape[0], len(feature_idx)))
    X_adv[:, feature_idx] += noise
    return X_adv


def poison_labels(y: np.ndarray, frac: float) -> np.ndarray:
    y_new = y.copy()
    n = int(len(y) * frac)
    idx = np.random.choice(len(y), n, replace=False)
    y_new[idx] = 1 - y_new[idx]
    return y_new

# ==============================================================================
# SIMULATION
# ==============================================================================

def run_simulation():
    ensure_directories()
    logger = get_logger("SimulationEngine")

    logger.info("=== FULLY ENHANCED ADVERSARIAL SIMULATION STARTED ===")

    # --------------------------------------------------------------------------
    # LOAD & SUBSAMPLE DATA
    # --------------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    idx = np.random.choice(len(X), SUBSAMPLE, replace=False)
    X, y = X[idx], y[idx]

    input_dim = X.shape[1]

    # Assume temporal + packet-related features are mutable
    mutable_features = [
        i for i, c in enumerate(df.columns)
        if any(k in c.lower() for k in ["iat", "pkt", "duration", "bytes"])
    ]

    results: List[Dict] = []

    # --------------------------------------------------------------------------
    # BASELINE MODEL
    # --------------------------------------------------------------------------
    model = EnsembleOrchestrator(input_dim)
    model.fit(X, y)

    conformal = ConformalEngine(alpha=ALPHA)
    conformal.calibrate(model, X, y)

    # --------------------------------------------------------------------------
    # EVASION SWEEP WITH ADAPTATION
    # --------------------------------------------------------------------------
    for eps in EVASION_LEVELS:
        logger.info(f"Evasion strength ε={eps}")
        thermostat = RiskThermostat(
            analyst_capacity=ANALYST_CAPACITY
        )

        X_adv = feature_targeted_evasion(X, eps, mutable_features)

        for i in range(0, len(X_adv), BATCH_SIZE):
            batch = X_adv[i:i+BATCH_SIZE]
            sets = conformal.prediction_sets(batch, model)
            state = thermostat.evaluate(sets)

            # Alert debt decay
            thermostat.alert_debt *= (1 - ALERT_DECAY)

            # Adaptive retraining
            if state in {SOCState.EVASION_LOCKED, SOCState.FAILURE}:
                logger.warning("Adaptive retraining triggered.")
                retrain_X = X_adv[i:i+RETRAIN_WINDOW]
                retrain_y = y[i:i+RETRAIN_WINDOW]

                model = EnsembleOrchestrator(input_dim)
                model.fit(retrain_X, retrain_y)

                conformal = ConformalEngine(alpha=ALPHA)
                conformal.calibrate(model, retrain_X, retrain_y)

                thermostat.alert_debt = 0  # reset after retrain

        results.append({
            "attack": "evasion",
            "strength": eps,
            "final_state": state.name,
            "avg_uncertainty": np.mean(thermostat.uncertainty_history),
            "alert_debt": thermostat.alert_debt
        })

    # --------------------------------------------------------------------------
    # POISONING SWEEP
    # --------------------------------------------------------------------------
    for frac in POISON_LEVELS:
        logger.info(f"Poisoning fraction={frac}")
        y_poison = poison_labels(y, frac)

        model = EnsembleOrchestrator(input_dim)
        model.fit(X, y_poison)

        conformal = ConformalEngine(alpha=ALPHA)
        conformal.calibrate(model, X, y_poison)

        thermostat = RiskThermostat(analyst_capacity=ANALYST_CAPACITY)

        for i in range(0, len(X), BATCH_SIZE):
            batch = X[i:i+BATCH_SIZE]
            sets = conformal.prediction_sets(batch, model)
            state = thermostat.evaluate(sets)
            thermostat.alert_debt *= (1 - ALERT_DECAY)

        results.append({
            "attack": "poisoning",
            "strength": frac,
            "final_state": state.name,
            "avg_uncertainty": np.mean(thermostat.uncertainty_history),
            "alert_debt": thermostat.alert_debt
        })

    # --------------------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------------------
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_PATH, index=False)

    # --------------------------------------------------------------------------
    # PLOT PHASE TRANSITION
    # --------------------------------------------------------------------------
    state_map = {
        "STABLE": 0,
        "SUSPICIOUS": 1,
        "EVASION_LOCKED": 2,
        "FAILURE": 3
    }

    evasion_df = res_df[res_df.attack == "evasion"]
    y_states = evasion_df.final_state.map(state_map)

    plt.figure(figsize=(10, 4))
    plt.plot(evasion_df.strength, y_states, marker="o")
    plt.yticks(list(state_map.values()), list(state_map.keys()))
    plt.xlabel("Evasion Strength (ε)")
    plt.ylabel("SOC State")
    plt.title("Attack Strength vs SOC Phase Transition")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    logger.info("=== ENHANCED SIMULATION COMPLETE ===")
    logger.info(f"Results saved to {RESULTS_PATH}")
    logger.info(f"Phase plot saved to {PLOT_PATH}")

# ==============================================================================
# ENTRY
# ==============================================================================

if __name__ == "__main__":
    run_simulation()
