"""
================================================================================
MAIN SOC PIPELINE – SYSTEM ORCHESTRATOR
================================================================================
Project: Adversarially Resilient SOC Pipeline

This file is the SINGLE executable entry point.
All other modules are imported as libraries.

Execution Flow:
1. Load processed IDS data (Person 1)
2. Train ensemble detector (Person 2)
3. Calibrate conformal uncertainty (Person 3)
4. Evaluate Risk Thermostat + SOC response
5. Generate uncertainty dashboard
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger, ensure_directories
from src.detection_ensemble import EnsembleOrchestrator, ModelAuditor
from src.risk_management_engine import (
    ConformalEngine,
    RiskThermostat,
    SOCDashboard
)

# ==============================================================================
# CONFIGURATION (SYSTEM CONTRACT)
# ==============================================================================

DATA_PATH = "data/processed/processed_flows.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SPLIT = 0.25   # fraction of remaining after test split

ALPHA = 0.05               # conformal error rate (95% coverage)
ANALYST_CAPACITY = 50

# ==============================================================================
# PIPELINE
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # 0. SYSTEM BOOTSTRAP
    # --------------------------------------------------------------------------
    ensure_directories()
    logger = get_logger("MainPipeline")

    logger.info("=== SOC PIPELINE STARTED ===")

    # --------------------------------------------------------------------------
    # 1. LOAD PROCESSED DATA (PERSON 1)
    # --------------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        logger.error(f"Processed dataset not found: {DATA_PATH}")
        raise FileNotFoundError(DATA_PATH)

    logger.info("Loading processed IDS dataset...")
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Processed data must contain a 'label' column")

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    input_dim = X.shape[1]
    logger.info(f"Dataset loaded | Samples: {len(X)} | Features: {input_dim}")

    # --------------------------------------------------------------------------
    # 2. TRAIN / CALIBRATION / TEST SPLITS
    # --------------------------------------------------------------------------
    logger.info("Creating stratified data splits...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp,
        test_size=CALIBRATION_SPLIT,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    logger.info(
        f"Split sizes | "
        f"Train: {len(X_train)} | "
        f"Cal: {len(X_cal)} | "
        f"Test: {len(X_test)}"
    )

    # --------------------------------------------------------------------------
    # 3. TRAIN ENSEMBLE DETECTOR (PERSON 2)
    # --------------------------------------------------------------------------
    logger.info("Training ensemble threat detector...")
    ensemble = EnsembleOrchestrator(input_dim=input_dim)
    ensemble.fit(X_train, y_train)

    # --------------------------------------------------------------------------
    # 4. BASELINE PERFORMANCE AUDIT (NO UNCERTAINTY)
    # --------------------------------------------------------------------------
    logger.info("Running baseline audit on test set...")
    auditor = ModelAuditor(logger)

    test_probs = ensemble.predict_proba(X_test)[:, 1]
    auditor.run_audit(y_test, test_probs)

    # --------------------------------------------------------------------------
    # 5. CONFORMAL CALIBRATION (PERSON 3 – MATH LAYER)
    # --------------------------------------------------------------------------
    logger.info("Calibrating conformal prediction engine...")
    conformal = ConformalEngine(alpha=ALPHA)
    conformal.calibrate(
        model=ensemble,
        X_cal=X_cal,
        y_cal=y_cal
    )

    # --------------------------------------------------------------------------
    # 6. UNCERTAINTY-AWARE INFERENCE
    # --------------------------------------------------------------------------
    logger.info("Generating conformal prediction sets...")
    prediction_sets = conformal.prediction_sets(X_test, ensemble)

    # --------------------------------------------------------------------------
    # 7. RISK THERMOSTAT (PERSON 3 – CONTROL LAYER)
    # --------------------------------------------------------------------------
    thermostat = RiskThermostat(
        analyst_capacity=ANALYST_CAPACITY
    )

    soc_state = thermostat.evaluate(prediction_sets)
    playbook = thermostat.playbook()

    logger.info(f"SOC STATE: {soc_state.name}")
    logger.info(f"SOC PLAYBOOK: {playbook}")

    # --------------------------------------------------------------------------
    # 8. SOC DASHBOARD / VISUALIZATION
    # --------------------------------------------------------------------------
    logger.info("Generating SOC uncertainty dashboard...")
    dashboard = SOCDashboard()
    dashboard.plot_uncertainty(thermostat.uncertainty_history)

    logger.info("=== SOC PIPELINE COMPLETED SUCCESSFULLY ===")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()