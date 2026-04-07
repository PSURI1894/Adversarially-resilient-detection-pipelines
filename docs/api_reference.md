# API Reference

The ARDP backend exposes a FastAPI server on port **8000**.
Interactive documentation: `http://localhost:8000/docs` (Swagger UI).

---

## REST Endpoints

### `GET /health`

Returns pipeline health status.

**Response** `200 OK`
```json
{
  "status": "healthy",
  "model_version": "1.3.0",
  "drift_state": "NORMAL",
  "conformal_coverage_last_batch": 0.963,
  "uptime_seconds": 12483
}
```

---

### `POST /predict`

Single-flow adversarial-aware inference.

**Request body**
```json
{
  "features": [0.12, -0.45, 1.03, ...],   // 80-dimensional feature vector
  "return_set": true,                        // include conformal prediction set
  "return_explanation": false                // include SHAP explanation
}
```

**Response** `200 OK`
```json
{
  "label": 1,
  "probability": 0.923,
  "epistemic_uncertainty": 0.041,
  "prediction_set": [1],
  "adversarial_score": 0.12,
  "risk_level": "LOW",
  "latency_ms": 3.7
}
```

| Field | Type | Description |
|---|---|---|
| `label` | int | Predicted class (0 = benign, 1 = attack) |
| `probability` | float | Ensemble mean probability for class 1 |
| `epistemic_uncertainty` | float | Variance across ensemble members |
| `prediction_set` | list[int] | RSCP+ conformal prediction set |
| `adversarial_score` | float | SHAP-fingerprint anomaly score (z-score) |
| `risk_level` | str | Current FSM state |

---

### `POST /predict/batch`

Vectorised batch inference (up to 10,000 flows).

**Request body**
```json
{
  "features": [[...], [...], ...],
  "return_set": false
}
```

**Response** `200 OK`
```json
{
  "labels": [0, 1, 1, 0, ...],
  "probabilities": [0.04, 0.91, 0.87, 0.11, ...],
  "epistemic_uncertainties": [0.01, 0.04, 0.03, 0.01, ...],
  "batch_coverage": 0.961,
  "latency_ms_total": 48.2
}
```

---

### `GET /metrics`

Pipeline performance metrics (last 1000 samples).

**Response** `200 OK`
```json
{
  "accuracy_1000": 0.971,
  "f1_1000": 0.968,
  "conformal_coverage_1000": 0.958,
  "avg_set_size_1000": 1.23,
  "adversarial_flags_1000": 7,
  "drift_detector_states": {
    "adwin": "NORMAL",
    "page_hinkley": "NORMAL",
    "ks_test": "ELEVATED",
    "mmd": "NORMAL"
  },
  "retraining_in_progress": false
}
```

---

### `GET /drift/status`

Current drift detection state.

**Response** `200 OK`
```json
{
  "consensus_state": "NORMAL",
  "detectors": {
    "adwin": {"state": "NORMAL", "drift_score": 0.001},
    "page_hinkley": {"state": "NORMAL", "cumulative_sum": 0.23},
    "ks_test": {"state": "ELEVATED", "p_value": 0.031},
    "mmd": {"state": "NORMAL", "mmd_score": 0.004}
  },
  "last_retrain_timestamp": "2025-04-01T14:32:10Z",
  "retraining_triggered": false
}
```

---

### `POST /retrain`

Manually trigger adaptive retraining (requires `Authorization: Bearer <token>`).

**Response** `202 Accepted`
```json
{
  "message": "Retraining scheduled",
  "job_id": "retrain-20250407-093012"
}
```

---

### `GET /model/info`

Current model version metadata from MLflow.

**Response** `200 OK`
```json
{
  "model_name": "ardp_ensemble",
  "version": "1.3.0",
  "stage": "Production",
  "run_id": "abc123def456",
  "metrics": {
    "val_accuracy": 0.974,
    "val_f1": 0.971,
    "val_conformal_coverage": 0.961
  },
  "registered_at": "2025-04-01T10:15:00Z"
}
```

---

## WebSocket — `/ws/alerts`

Real-time alert stream. Connect with any WebSocket client.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/alerts');
ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log(alert);
};
```

**Message format**
```json
{
  "timestamp": "2025-04-07T09:30:15.123Z",
  "flow_id": "flow-00029481",
  "label": 1,
  "probability": 0.97,
  "prediction_set": [1],
  "adversarial_flag": false,
  "risk_level": "ELEVATED",
  "src_ip": "10.0.0.55",
  "dst_ip": "192.168.1.1",
  "protocol": "TCP"
}
```

---

## Authentication

API authentication uses Bearer tokens. Set the `ARDP_API_KEY` environment variable
before starting the server:

```bash
export ARDP_API_KEY="your-secret-key"
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Protected endpoints: `POST /retrain`, `DELETE /model/{version}`.
All read/predict endpoints are unauthenticated (suitable for internal deployment).

---

## Error Codes

| HTTP Code | Meaning |
|---|---|
| 200 | Success |
| 202 | Accepted (async operation scheduled) |
| 400 | Invalid input (wrong feature dimension, malformed JSON) |
| 401 | Unauthorized (missing/invalid API key) |
| 422 | Unprocessable entity (validation error) |
| 503 | Model not loaded / pipeline not ready |
