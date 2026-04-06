"""
================================================================================
MAIN SOC PIPELINE – SYSTEM ORCHESTRATOR
================================================================================
Project: Adversarially Resilient SOC Pipeline

This file is the SINGLE executable entry point.
Supports:
1. Batch Mode: Traditional training and offline evaluation.
2. Streaming Mode: Kafka-based real-time inference with drift detection.

Features (Week 6):
    - Hydra/YAML configuration with fallback to defaults
    - MLflow experiment tracking at every pipeline stage
    - Prometheus metrics emission
    - Structured logging with correlation IDs
    - Timing instrumentation for all stages

Execution Flow:
1. Load configuration (Hydra / YAML fallback)
2. Load processed IDS data
3. Train/Load ensemble detector
4. Calibrate conformal uncertainty
5. In-flight: Inference → Drift Detection → Adaptive Retraining
6. Track experiment metrics (MLflow / local fallback)
7. SOC Response & Reporting
================================================================================
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    get_logger, ensure_directories, timed_stage,
    get_correlation_id, set_correlation_id,
)
from src.detection_ensemble import EnsembleOrchestrator, ModelAuditor, EnsembleConfig
from src.risk_management_engine import (
    ConformalEngine,
    RiskThermostat,
    SOCDashboard,
    SOCState,
)
from src.streaming import FlowConsumer, RealtimeInferenceService, FeatureStore
from src.drift import ConceptDriftEngine, AdaptiveRetrainingPipeline
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.monitoring import ProductionMonitor

# ==============================================================================
# CONFIGURATION LOADER
# ==============================================================================

def load_config(config_path: str = "configs/default.yaml") -> dict:
    """
    Load pipeline configuration from YAML with OmegaConf (Hydra-compatible).
    Falls back to built-in defaults if OmegaConf/YAML unavailable.
    """
    try:
        from omegaconf import OmegaConf
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    # Fallback defaults
    return {
        "data": {
            "processed_path": "data/processed/processed_flows.csv",
            "random_state": 42,
            "test_size": 0.20,
            "calibration_split": 0.25,
        },
        "conformal": {"alpha": 0.05},
        "risk": {"analyst_capacity": 50},
        "streaming": {
            "kafka": {"bootstrap_servers": None},
            "feature_store": {"redis_url": None},
        },
        "mlops": {
            "experiment": {"name": "ids-pipeline", "tracking_uri": None},
            "monitoring": {"prometheus_port": 9090, "enable_server": False},
        },
    }


# ==============================================================================
# CALIBRATED MODEL WRAPPER
# ==============================================================================

class CalibratedEnsemble:
    def __init__(self, base_model, calibrator=None):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        probs = self.base_model.predict_proba(X)
        if self.calibrator is None:
            return probs
        p1 = probs[:, 1] if probs.ndim == 2 else probs
        p1_cal = self.calibrator.predict_proba(p1)
        return np.vstack([1 - p1_cal, p1_cal]).T


# ==============================================================================
# BATCH PIPELINE
# ==============================================================================

def run_batch(X_train, X_cal, X_test, y_train, y_cal, y_test,
              input_dim, feature_names, cfg, tracker, monitor):
    logger = get_logger("BatchPipeline")
    logger.info("--- STARTING BATCH PIPELINE ---")

    conf_cfg = cfg["conformal"]
    risk_cfg = cfg["risk"]

    # 1. Train Ensemble
    with timed_stage("ensemble_training") as timer:
        config = EnsembleConfig(input_dim=input_dim)
        ensemble = EnsembleOrchestrator(config)
        ensemble.fit(X_train, y_train)

    tracker.log_pipeline_stage("training", {
        "duration_ms": timer.elapsed_ms,
        "n_train_samples": len(X_train),
        "input_dim": input_dim,
    })

    # 2. Probability Calibration
    calibrated_ensemble = CalibratedEnsemble(ensemble)
    try:
        from src.models import IsotonicCalibration
        with timed_stage("probability_calibration"):
            calibrator = IsotonicCalibration()
            calibrator.fit(ensemble.predict_proba(X_cal)[:, 1], y_cal)
            calibrated_ensemble = CalibratedEnsemble(ensemble, calibrator)
    except ImportError:
        logger.warning("Calibration module unavailable, using raw probabilities")

    # 3. Conformal Calibration
    with timed_stage("conformal_calibration"):
        conformal = ConformalEngine(alpha=conf_cfg["alpha"])
        conformal.calibrate(calibrated_ensemble, X_cal, y_cal)

    tracker.log_pipeline_stage("conformal", {
        "alpha": conf_cfg["alpha"],
        "q_hat": conformal.q_hat or 0.0,
    })

    # 4. Baseline Audit
    with timed_stage("baseline_audit"):
        auditor = ModelAuditor(logger)
        test_probs = calibrated_ensemble.predict_proba(X_test)[:, 1]
        audit_report = auditor.run_audit(y_test, test_probs)

    tracker.log_pipeline_stage("evaluation", audit_report)

    # 5. Conformal Inference
    with timed_stage("conformal_inference"):
        prediction_sets = conformal.prediction_sets(X_test, calibrated_ensemble)

    avg_set_size = float(np.mean([len(s) for s in prediction_sets]))
    coverage = float(np.mean([
        1 if y_test[i] in prediction_sets[i] else 0
        for i in range(len(y_test))
    ]))
    tracker.log_metrics({
        "conformal/avg_set_size": avg_set_size,
        "conformal/coverage": coverage,
    })
    monitor.record_set_size(avg_set_size)

    # 6. Risk Thermostat
    thermostat = RiskThermostat(
        analyst_capacity=risk_cfg.get("analyst_capacity", 50),
        warning_threshold=risk_cfg.get("warning_threshold", 1.1),
        critical_threshold=risk_cfg.get("critical_threshold", 1.8),
        hysteresis_steps=risk_cfg.get("hysteresis_steps", 3),
    )
    state = thermostat.evaluate(prediction_sets)
    playbook = thermostat.playbook()

    logger.info(f"SOC STATE: {state.name} | Severity: {thermostat.severity:.1f}")
    logger.info(f"SOC PLAYBOOK: {playbook}")

    tracker.log_metrics({
        "risk/severity": thermostat.severity,
        "risk/alert_debt": thermostat.alert_debt,
    })
    tracker.set_tag("soc_state", state.name)

    # 7. XAI & Reporting (if available)
    try:
        from src.explainability import IncidentReporter
        reporter = IncidentReporter(feature_names=feature_names)
        if np.any(test_probs > 0.5):
            idx = np.where(test_probs > 0.5)[0][0]
            report = reporter.generate_report(
                X_test[idx], test_probs[idx], prediction_sets[idx],
                risk_score=thermostat.severity, soc_state=state.name,
            )
            reporter.export_json(report)
    except ImportError:
        logger.info("Explainability module not available, skipping XAI reports")

    # 8. Dashboard
    with timed_stage("dashboard_generation"):
        dashboard = SOCDashboard()
        dashboard.plot_uncertainty(thermostat.uncertainty_history)
        dashboard.plot_severity(thermostat.severity_history)

    for fig_name in ["risk_thermostat.png", "severity_timeline.png"]:
        fig_path = os.path.join("reports", "figures", fig_name)
        if os.path.exists(fig_path):
            tracker.log_artifact(fig_path)

    logger.info("--- BATCH PIPELINE COMPLETE ---")
    return calibrated_ensemble, conformal


# ==============================================================================
# STREAMING PIPELINE
# ==============================================================================

def run_streaming(model, conformal_engine, input_dim, feature_names, cfg, monitor):
    logger = get_logger("StreamingPipeline")
    logger.info("--- STARTING STREAMING PIPELINE ---")

    stream_cfg = cfg.get("streaming", {})
    kafka_cfg = stream_cfg.get("kafka", {})
    fs_cfg = stream_cfg.get("feature_store", {})

    bootstrap = kafka_cfg.get("bootstrap_servers") or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    redis_url = fs_cfg.get("redis_url") or os.getenv("REDIS_URL")

    consumer = FlowConsumer(bootstrap_servers=bootstrap)
    inf_service = RealtimeInferenceService(model, conformal_engine, input_dim=input_dim)
    feat_store = FeatureStore(redis_url=redis_url)
    drift_engine = ConceptDriftEngine(np.zeros((100, input_dim)))
    retrainer = AdaptiveRetrainingPipeline(EnsembleOrchestrator(input_dim))

    logger.info("Streaming loop active. Listening for flows...")

    try:
        while True:
            batch = consumer.consume_batch(n=100, timeout=1.0)
            if not batch:
                continue

            results = inf_service.predict_batch(batch)
            monitor.record_inference_batch(results)

            for i, res in enumerate(results):
                key = f"flow-{res['timestamp']}-{i}"
                feat_store.put(key, np.array(batch[i]["features"]), metadata=res)

            current_X = np.array([b["features"][:input_dim] for b in batch])
            errors = [1 if r["prediction"] != r["label_true"] else 0 for r in results]

            if drift_engine.evaluate(current_X, errors):
                logger.warning("CONCEPT DRIFT DETECTED! Initiating adaptive retraining...")
                monitor.record_retraining()
                new_model = retrainer.retrain(
                    model, current_X, np.array([b["label"] for b in batch]),
                    current_X, np.array([b["label"] for b in batch]),
                )
                inf_service.model = new_model
                logger.info("Model updated in inference service.")

            if results:
                health = inf_service.health_check()
                logger.info(f"Status: {health['status']} | P99: {health['latency_p99_ms']}ms")

            # Check monitoring alert rules
            alerts = monitor.check_alert_rules()
            for alert in alerts:
                logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")

    except KeyboardInterrupt:
        logger.info("Streaming stopped by user.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SOC Pipeline Orchestrator")
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "streaming"])
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    ensure_directories()
    set_correlation_id("")  # Generate fresh ID
    cid = get_correlation_id()
    logger = get_logger("MainPipeline")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    mlops_cfg = cfg.get("mlops", {})

    logger.info(f"=== SOC PIPELINE STARTED (correlation_id={cid}, mode={args.mode}) ===")

    # Initialise MLOps
    tracker = ExperimentTracker(
        experiment_name=mlops_cfg.get("experiment", {}).get("name", "ids-pipeline"),
        tracking_uri=mlops_cfg.get("experiment", {}).get("tracking_uri"),
    )
    monitor = ProductionMonitor(
        port=mlops_cfg.get("monitoring", {}).get("prometheus_port", 9090),
        enable_server=mlops_cfg.get("monitoring", {}).get("enable_server", False),
    )

    run_id = tracker.start_run(
        run_name=f"pipeline-{cid}",
        tags={"correlation_id": cid, "mode": args.mode, "pipeline_version": "v2"},
    )
    tracker.log_params({
        "data/test_size": data_cfg["test_size"],
        "data/calibration_split": data_cfg["calibration_split"],
        "conformal/alpha": cfg["conformal"]["alpha"],
        "mode": args.mode,
    })

    try:
        # Load Data
        data_path = data_cfg["processed_path"]
        if not os.path.exists(data_path):
            logger.error(f"Processed dataset not found: {data_path}")
            raise FileNotFoundError(data_path)

        with timed_stage("data_loading"):
            df = pd.read_csv(data_path)

        y = df["label"].values
        X = df.drop(columns=["label"]).values
        input_dim = X.shape[1]
        feature_names = df.drop(columns=["label"]).columns.tolist()

        # Splits
        with timed_stage("data_splitting"):
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=data_cfg["test_size"],
                stratify=y, random_state=data_cfg["random_state"],
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_temp, y_temp, test_size=data_cfg["calibration_split"],
                stratify=y_temp, random_state=data_cfg["random_state"],
            )

        logger.info(f"Splits | Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

        if args.mode == "batch":
            run_batch(X_train, X_cal, X_test, y_train, y_cal, y_test,
                      input_dim, feature_names, cfg, tracker, monitor)
        else:
            logger.info("Pre-training base model for streaming inference...")
            model, conformal = run_batch(
                X_train, X_cal, X_test, y_train, y_cal, y_test,
                input_dim, feature_names, cfg, tracker, monitor,
            )
            run_streaming(model, conformal, input_dim, feature_names, cfg, monitor)

        logger.info(f"=== SOC PIPELINE COMPLETED (correlation_id={cid}) ===")

    finally:
        tracker.end_run()


if __name__ == "__main__":
    main()
