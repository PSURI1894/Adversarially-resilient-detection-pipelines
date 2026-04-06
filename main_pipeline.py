"""
================================================================================
MAIN SOC PIPELINE – SYSTEM ORCHESTRATOR
================================================================================
Project: Adversarially Resilient SOC Pipeline

This file is the SINGLE executable entry point.
Supports:
1. Batch Mode: Traditional training and offline evaluation.
2. [NEW] Streaming Mode: Kafka-based real-time inference with drift detection.

Execution Flow:
1. Load processed IDS data
2. Train/Load ensemble detector
3. Calibrate conformal uncertainty
4. In-flight: Inference -> Drift Detection -> Adaptive Retraining
5. SOC Response & Reporting
================================================================================
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger, ensure_directories
from src.detection_ensemble import EnsembleOrchestrator, ModelAuditor, EnsembleConfig
from src.models import TemperatureScaling, IsotonicCalibration
from src.explainability import (
    SHAPExplainer,
    LIMEExplainer,
    AttributionFingerprintDetector,
    FeatureSensitivityAnalyzer,
    IncidentReporter,
)
from src.risk_management_engine import (
    ConformalEngine,
    RiskThermostat,
    SOCDashboard,
    SOCState
)
from src.streaming import FlowConsumer, RealtimeInferenceService, FeatureStore
from src.drift import ConceptDriftEngine, AdaptiveRetrainingPipeline

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = "data/processed/processed_flows.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SPLIT = 0.25
ALPHA = 0.05
ANALYST_CAPACITY = 50

# ==============================================================================
# CALIBRATED MODEL WRAPPER
# ==============================================================================

class CalibratedEnsemble:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
    def predict_proba(self, X):
        probs = self.base_model.predict_proba(X)
        # Handle 1D or 2D logic
        if probs.ndim == 2:
            p1 = probs[:, 1]
        else:
            p1 = probs
        p1_cal = self.calibrator.predict_proba(p1)
        return np.vstack([1 - p1_cal, p1_cal]).T

# ==============================================================================
# PIPELINE MODES
# ==============================================================================

def run_batch(X_train, X_cal, X_test, y_train, y_cal, y_test, input_dim, feature_names):
    logger = get_logger("BatchPipeline")
    logger.info("--- STARTING BATCH PIPELINE ---")

    # 1. Train Ensemble
    config = EnsembleConfig(input_dim=input_dim)
    ensemble = EnsembleOrchestrator(config)
    ensemble.fit(X_train, y_train)

    # 2. Probability Calibration
    calibrator = IsotonicCalibration()
    calibrator.fit(ensemble.predict_proba(X_cal)[:, 1], y_cal)
    calibrated_ensemble = CalibratedEnsemble(ensemble, calibrator)

    # 3. Conformal Calibration
    conformal = ConformalEngine(alpha=ALPHA)
    conformal.calibrate(calibrated_ensemble, X_cal, y_cal)

    # 4. Inference
    test_probs = calibrated_ensemble.predict_proba(X_test)[:, 1]
    prediction_sets = conformal.prediction_sets(X_test, calibrated_ensemble)

    # 5. SOC Assessment
    thermostat = RiskThermostat(analyst_capacity=ANALYST_CAPACITY)
    state = thermostat.evaluate(prediction_sets)
    logger.info(f"Final SOC State: {state.name}")

    # 6. XAI & Reporting
    reporter = IncidentReporter(feature_names=feature_names)
    # (Simplified reporting for 1 alert)
    if np.any(test_probs > 0.5):
        idx = np.where(test_probs > 0.5)[0][0]
        report = reporter.generate_report(
            X_test[idx], test_probs[idx], prediction_sets[idx],
            risk_score=thermostat.severity, soc_state=state.name
        )
        reporter.export_json(report)

    logger.info("--- BATCH PIPELINE COMPLETE ---")
    return calibrated_ensemble, conformal

def run_streaming(model, conformal_engine, input_dim, feature_names):
    logger = get_logger("StreamingPipeline")
    logger.info("--- STARTING STREAMING PIPELINE ---")

    # 1. Setup Infrastructure
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    redis_url = os.getenv("REDIS_URL")
    
    consumer = FlowConsumer(bootstrap_servers=bootstrap)
    inf_service = RealtimeInferenceService(model, conformal_engine, input_dim=input_dim)
    feat_store = FeatureStore(redis_url=redis_url)
    drift_engine = ConceptDriftEngine(np.zeros((10, input_dim))) # Placeholder ref
    retrainer = AdaptiveRetrainingPipeline(EnsembleOrchestrator(input_dim))

    logger.info("Streaming loop active. Listening for flows...")
    
    try:
        while True:
            # Pull batch for efficiency
            batch = consumer.consume_batch(n=100, timeout=1.0)
            if not batch:
                continue

            # Run Inference
            results = inf_service.predict_batch(batch)
            
            # Store in FeatureStore for backfilling/consistency
            for i, res in enumerate(results):
                key = f"flow-{res['timestamp']}-{i}"
                feat_store.put(key, np.array(batch[i]["features"]), metadata=res)

            # Drift Detection
            current_X = np.array([b["features"][:input_dim] for b in batch])
            errors = [1 if r["prediction"] != r["label_true"] else 0 for r in results]
            
            if drift_engine.evaluate(current_X, errors):
                logger.warning("CONCEPT DRIFT DETECTED! Initiating adaptive retraining...")
                # Retrain with the latest batch data
                new_model = retrainer.retrain(
                    model, current_X, np.array([b["label"] for b in batch]),
                    current_X, np.array([b["label"] for b in batch]) # Simplified holdout
                )
                # In real scenario, we'd recalibrate on a fresh pool
                inf_service.model = new_model
                logger.info("Model updated in inference service.")

            # Health Check Log
            if results:
                health = inf_service.health_check()
                logger.info(f"Status: {health['status']} | P99 Latency: {health['latency_p99_ms']}ms")

    except KeyboardInterrupt:
        logger.info("Streaming stopped by user.")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SOC Pipeline Orchestrator")
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "streaming"])
    args = parser.parse_args()

    ensure_directories()
    logger = get_logger("MainPipeline")

    # Load Data (required for both to get schema)
    df = pd.read_csv(DATA_PATH)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    input_dim = X.shape[1]
    feature_names = df.drop(columns=["label"]).columns.tolist()

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=CALIBRATION_SPLIT, stratify=y_temp, random_state=RANDOM_STATE)

    if args.mode == "batch":
        run_batch(X_train, X_cal, X_test, y_train, y_cal, y_test, input_dim, feature_names)
    else:
        # Pre-train for streaming
        logger.info("Pre-training base model for streaming inference...")
        model, conformal = run_batch(X_train, X_cal, X_test, y_train, y_cal, y_test, input_dim, feature_names)
        run_streaming(model, conformal, input_dim, feature_names)

if __name__ == "__main__":
    main()