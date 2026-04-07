# Architecture

## Overview

ARDP is a four-layer adversarially resilient intrusion detection pipeline.
Each layer is independently deployable and communicates via well-defined interfaces.

---

## Layer 1 — Data Layer

### Kafka Consumer (`src/streaming/kafka_consumer.py`)
- Subscribes to a configurable Kafka topic (`net.flows`)
- Deserialises Avro / JSON flow records
- Applies `RobustScaler` normalisation (fit on training set, persisted via MLflow)

### Feature Store (`src/streaming/feature_store.py`)
- Sliding-window temporal statistics: mean, variance, min/max, 25/75 percentiles
- Windows: 1 s, 10 s, 60 s per source IP
- Backed by an in-memory deque with configurable max-size

### Adversarial Augmentation (`src/attacks/`)
- During training: online PGD perturbations applied to minibatches
- During inference: disabled (detection, not generation)

---

## Layer 2 — Detection Layer

### Deep Ensemble (`src/models/deep_ensemble.py`)
Four members, each with independent random initialisation and bootstrap sampling:

| Member | Architecture | Strength |
|---|---|---|
| XGBoost | Gradient-boosted trees, 200 estimators, depth 6 | Categorical / sparse features |
| 1D-CNN | 3 × Conv1D + GlobalMaxPool + Dense | Local temporal patterns |
| TabTransformer | Column embedding + 4-head attention + MLP | Inter-feature interactions |
| VAE-IDS | Encoder → μ/σ → Decoder; anomaly = recon error + KL | Unsupervised zero-day |

**Ensemble output**: calibrated soft-vote mean probability + epistemic uncertainty (variance).

### Model Registry (`src/mlops/model_registry.py`)
- MLflow backend (`mlruns/`)
- Versioned model checkpoints; `Production` stage tag
- Auto-promotes if new model beats current champion on held-out validation F1

---

## Layer 3 — Uncertainty & Defense Layer

### RSCP+ (`src/conformal/rscp.py`)
1. **Calibration**: compute Monte Carlo smoothed non-conformity scores on calibration set
2. **PTT**: rank-transform scores for tighter set sizes
3. **Threshold**: `q_hat` = (1−α)(1+1/n_cal) quantile
4. **Inference**: prediction set = {y : s̃(x,y) ≤ q_hat}
5. **Certified radius**: r* = σ · Φ⁻¹((1−α)/2)

### Adversarial Detector (`src/explainability/adversarial_detector.py`)
- SHAP TreeExplainer (XGBoost member) + KernelExplainer (CNN member)
- Fingerprint distance from clean centroid → anomaly score
- Threshold calibrated at 95th percentile of clean validation set

### Risk Thermostat (`src/risk_management_engine.py`)
Multi-signal Finite State Machine:

```
NORMAL ──(coverage drops)──► ELEVATED ──(prolonged)──► CRISIS
  ▲                                                        │
  └──────────────(recovery + retraining complete)──────────┘
```

State governs: alert threshold, model capacity, playbook selection.

---

## Layer 4 — Operational Layer

### FastAPI Server (`src/api/server.py`)
- `/predict`: single-flow inference
- `/predict/batch`: vectorised batch
- `/ws/alerts`: WebSocket real-time alert push

### React Dashboard (`dashboard/src/`)
- Real-time threat timeline, model confidence strip, drift alert panel
- Connects to WebSocket endpoint; refreshes at 1 s intervals

### Concept Drift Monitor (`src/drift/drift_detector.py`)
- ADWIN, Page-Hinkley, KS test, MMD — all run in parallel
- Consensus: ≥ 2 of 4 detectors → trigger retraining

### Adaptive Retrainer (`src/drift/adaptive_retrainer.py`)
- Fetches freshest N flows from feature store
- Retrains all ensemble members with warm-start (continued training)
- Re-calibrates RSCP+ on new calibration shard
- Registers new model version in MLflow; A/B shadow traffic before promotion

---

## Data Flow Diagram

```
Kafka ──► Consumer ──► Feature Store ──► [Training Path]
                                    │       Adversarial Augment
                                    │       Deep Ensemble fit
                                    │       RSCP+ calibrate
                                    │       MLflow register
                                    │
                                    └──► [Inference Path]
                                             RSCP+ predict_set
                                             Adversarial Detector score
                                             Risk FSM decide
                                             WebSocket push
                                             Drift Monitor update
```
