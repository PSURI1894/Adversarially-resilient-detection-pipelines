"""
================================================================================
FULLY ENHANCED SIMULATION ENGINE – ADAPTIVE ADVERSARIAL SOC
================================================================================
Includes:
✓ Attack strength → SOC phase plots
✓ Alert-debt decay
✓ Adaptive retraining
✓ Feature-targeted evasion
✓ [NEW] Streaming Mode (Kafka Simulation)
✓ [NEW] Concept Drift Injection (Sudden/Gradual)
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.utils import get_logger, ensure_directories
from src.data_infrastructure import AdversarialArsenal, AttackStrategy
from src.detection_ensemble import EnsembleOrchestrator
from src.risk_management_engine import ConformalEngine, RiskThermostat, SOCState
from src.streaming import FlowProducer, FlowConsumer, RealtimeInferenceService
from src.drift import ConceptDriftEngine, AdaptiveRetrainingPipeline

# ==============================================================================
# CONFIG
# ==============================================================================

DATA_PATH = "data/processed/processed_flows.csv"
RESULTS_PATH = "reports/audit_logs/simulation_results.csv"
PLOT_PATH = "reports/figures/attack_phase_transition.png"

RANDOM_STATE = 42
SUBSAMPLE = 50_000
BATCH_SIZE = 5_000

EVASION_LEVELS = [0.0, 0.05, 0.10, 0.20]
DRIFT_TYPES = ["none", "sudden", "gradual"]

ALPHA = 0.05
ANALYST_CAPACITY = 50
ALERT_DECAY = 0.4

np.random.seed(RANDOM_STATE)

# ==============================================================================
# DRIFT INJECTION UTILITIES
# ==============================================================================

def inject_drift(X: np.ndarray, drift_type: str, progress: float) -> np.ndarray:
    """
    Inject distributional shift into features.
    
    progress: float from 0.0 to 1.0 representing simulation progress.
    """
    X_drift = X.copy()
    if drift_type == "none":
        return X_drift
        
    # Features often affected by drift: IAT, duration, bytes
    # Index assumption: let's pick some indices or use names if we had them
    # For simulation, we'll shift the first 3 columns
    if drift_type == "sudden":
        if progress > 0.5:
            X_drift[:, :3] *= 5.0  # Intense sudden shift
    elif drift_type == "gradual":
        # Gradual multiplier from 1.0 to 4.0
        multiplier = 1.0 + (3.0 * progress)
        X_drift[:, :3] *= multiplier
        
    return X_drift

# ==============================================================================
# SIMULATION
# ==============================================================================

def run_simulation(mode: str = "batch"):
    ensure_directories()
    logger = get_logger("SimulationEngine")
    logger.info(f"=== SIMULATION STARTED (Mode: {mode}) ===")

    # --------------------------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data not found at {DATA_PATH}. Run data_infrastructure.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    X_all = df.drop(columns=["label"]).values
    y_all = df["label"].values

    idx = np.random.choice(len(X_all), min(SUBSAMPLE, len(X_all)), replace=False)
    X, y = X_all[idx], y_all[idx]

    input_dim = X.shape[1]
    feature_names = df.drop(columns=["label"]).columns.tolist()
    arsenal = AdversarialArsenal(feature_names)

    # --------------------------------------------------------------------------
    # 2. INITIAL CALIBRATION
    # --------------------------------------------------------------------------
    model = EnsembleOrchestrator(input_dim)
    # Use first 20% for initial train/cal
    split_idx = int(0.2 * len(X))
    X_init, y_init = X[:split_idx], y[:split_idx]
    model.fit(X_init, y_init)

    conformal = ConformalEngine(alpha=ALPHA)
    conformal.calibrate(model, X_init, y_init)

    results: List[Dict] = []

    # --------------------------------------------------------------------------
    # 3. STREAMING LOOP WITH DRIFT & EVASION
    # --------------------------------------------------------------------------
    X_stream = X[split_idx:]
    y_stream = y[split_idx:]

    for drift_type in DRIFT_TYPES:
        for eps in EVASION_LEVELS:
            logger.info(f"Running scenario: Drift={drift_type}, Evasion ε={eps}")
            
            thermostat = RiskThermostat(analyst_capacity=ANALYST_CAPACITY)
            drift_engine = ConceptDriftEngine(X_init)
            retrainer = AdaptiveRetrainingPipeline(EnsembleOrchestrator(input_dim))
            
            # Setup streaming infrastructure if in streaming mode
            producer = None
            consumer = None
            if mode == "streaming":
                producer = FlowProducer(topic="raw-traffic")
                consumer = FlowConsumer(input_topic="raw-traffic", output_topic="enriched-features")
                infer_service = RealtimeInferenceService(model, conformal, input_dim=input_dim)

            # Performance tracking
            prediction_errors = []
            
            # Attack Strategy
            strategy = AttackStrategy(attack_type="pgd", params={"epsilon": eps})
            
            for i in range(0, len(X_stream), BATCH_SIZE):
                progress = i / len(X_stream)
                batch_X = X_stream[i : i + BATCH_SIZE]
                batch_y = y_stream[i : i + BATCH_SIZE]

                # Inject Drift
                batch_X = inject_drift(batch_X, drift_type, progress)

                # Inject Evasion (on malicious samples only usually, but arsenal handles)
                batch_X_adv = arsenal.evasion_by_jitter(batch_X, eps)

                # Inference & Evaluation
                if mode == "streaming":
                    # Simulate Kafka flow
                    for j in range(len(batch_X_adv)):
                        producer.send({"features": batch_X_adv[j].tolist(), "label": int(batch_y[j])})
                    
                    enriched_batch = consumer.consume_batch(len(batch_X_adv))
                    inference_results = infer_service.predict_batch(enriched_batch)
                    
                    pred_sets = [res["prediction_set"] for res in inference_results]
                    errors = [1 if res["prediction"] != res["label_true"] else 0 for res in inference_results]
                    prediction_errors.extend(errors)
                    batch_X_eval = np.array([res["features"][:input_dim] for res in enriched_batch])
                else:
                    pred_sets = conformal.prediction_sets(batch_X_adv, model)
                    probs = model.predict_proba(batch_X_adv)
                    preds = np.argmax(probs, axis=1)
                    errors = (preds != batch_y).astype(int).tolist()
                    prediction_errors.extend(errors[-BATCH_SIZE:])
                    batch_X_eval = batch_X_adv

                # Update State
                state = thermostat.evaluate(pred_sets)
                thermostat.alert_debt *= (1 - ALERT_DECAY)

                # Check Drift & Trigger Retraining
                if drift_engine.evaluate(batch_X_eval, errors):
                    logger.warning(f"Drift detected at batch {i//BATCH_SIZE}! Triggering retrain.")
                    # In real system, we'd pull from FeatureStore/buffer
                    # Here we use the latest batch for retraining
                    model = retrainer.retrain(
                        current_model=model,
                        X_train_new=batch_X_eval,
                        y_train_new=batch_y,
                        X_holdout=X_init, # Simplified
                        y_holdout=y_init
                    )
                    conformal.calibrate(model, batch_X_eval, batch_y)
                    thermostat.alert_debt = 0  # Reset debt after system update

            results.append({
                "drift_type": drift_type,
                "evasion_eps": eps,
                "final_state": state.name,
                "avg_uncertainty": np.mean(thermostat.uncertainty_history),
                "final_severity": thermostat.severity,
                "error_rate": np.mean(prediction_errors)
            })

    # --------------------------------------------------------------------------
    # 4. SAVE & PLOT
    # --------------------------------------------------------------------------
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Results saved to {RESULTS_PATH}")

    # Plotting logic (simplified for multi-scenario)
    plt.figure(figsize=(12, 6))
    for d_type in DRIFT_TYPES:
        subset = res_df[res_df.drift_type == d_type]
        plt.plot(subset.evasion_eps, subset.final_severity, marker="o", label=f"Drift: {d_type}")
    
    plt.xlabel("Evasion Strength (ε)")
    plt.ylabel("Final Risk Severity (0-100)")
    plt.title("Impact of Drift & Evasion on System Severity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_PATH)
    logger.info(f"Phase plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "streaming"])
    args = parser.parse_args()
    
    run_simulation(mode=args.mode)
