"""
================================================================================
SOC DASHBOARD API — FASTAPI APPLICATION
================================================================================
REST + WebSocket endpoints for the real-time SOC dashboard.

Endpoints:
    GET  /api/status          — Current SOC state, model version, drift score
    GET  /api/alerts          — Paginated alert history with filters
    GET  /api/metrics         — Model performance time series
    GET  /api/metrics/history — Historical metrics over time
    GET  /api/explain/{id}    — SHAP + LIME explanation for specific alert
    POST /api/simulate        — Trigger adversarial simulation
    WS   /ws/live             — Real-time alert + state push
================================================================================
"""

import asyncio
import math
import os
import random
import time
import uuid
import logging
from typing import Dict, Optional, List
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


# ==============================================================================
# REQUEST / RESPONSE MODELS
# ==============================================================================


class SimulateRequest(BaseModel):
    attack_type: str = "pgd"
    epsilon: float = 0.1
    n_samples: int = 1000


class AlertResponse(BaseModel):
    id: str
    timestamp: float
    prediction: int
    probabilities: List[float]
    prediction_set: List[int]
    uncertainty: str
    latency_ms: float
    severity: Optional[float] = None


class StatusResponse(BaseModel):
    soc_state: str
    severity: float
    alert_debt: float
    calibration_drift: float
    disagreement: float
    n_evaluations: int
    model_version: str
    uptime_seconds: float
    active_connections: int


# ==============================================================================
# IN-MEMORY STORES (populated by pipeline at runtime)
# ==============================================================================


class PipelineState:
    """Shared state between the pipeline and the API layer."""

    def __init__(self):
        self.start_time = time.time()
        self.soc_state = "STABLE"
        self.severity = 0.0
        self.alert_debt = 0.0
        self.calibration_drift = 0.0
        self.disagreement = 0.0
        self.n_evaluations = 0
        self.model_version = "v1"

        self.alerts: deque = deque(maxlen=10_000)
        self.metrics_history: deque = deque(maxlen=5_000)
        self.uncertainty_history: List[float] = []
        self.severity_history: List[float] = []
        self.latency_history: List[float] = []

        # Performance metrics time series
        self.f1_history: List[Dict] = []
        self.fdr_history: List[Dict] = []
        self.auc_history: List[Dict] = []
        self.set_size_history: List[Dict] = []
        self.drift_history: List[Dict] = []

    def push_alert(self, alert: Dict):
        alert["id"] = alert.get("id", uuid.uuid4().hex[:8])
        alert["timestamp"] = alert.get("timestamp", time.time())
        self.alerts.appendleft(alert)

    def push_metrics(self, metrics: Dict):
        metrics["timestamp"] = time.time()
        self.metrics_history.appendleft(metrics)

    def update_state(self, diagnostics: Dict):
        self.soc_state = diagnostics.get("state", self.soc_state)
        self.severity = diagnostics.get("severity", self.severity)
        self.alert_debt = diagnostics.get("alert_debt", self.alert_debt)
        self.calibration_drift = diagnostics.get(
            "calibration_drift", self.calibration_drift
        )
        self.disagreement = diagnostics.get("disagreement", self.disagreement)
        self.n_evaluations = diagnostics.get("n_evaluations", self.n_evaluations)


# ==============================================================================
# APP FACTORY
# ==============================================================================


def create_app(pipeline_state: Optional[PipelineState] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    pipeline_state : PipelineState, optional
        Shared state from the running pipeline. Creates a new one if None.
    """
    state = pipeline_state or PipelineState()
    ws_manager = WebSocketManager()

    app = FastAPI(
        title="SOC Pipeline Dashboard API",
        description="Real-time adversarially resilient IDS monitoring",
        version="2.0.0",
    )

    # CORS for React dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # REST Endpoints
    # ------------------------------------------------------------------

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        """Current SOC state, model version, drift score."""
        return StatusResponse(
            soc_state=state.soc_state,
            severity=state.severity,
            alert_debt=state.alert_debt,
            calibration_drift=state.calibration_drift,
            disagreement=state.disagreement,
            n_evaluations=state.n_evaluations,
            model_version=state.model_version,
            uptime_seconds=time.time() - state.start_time,
            active_connections=ws_manager.active_connections,
        )

    @app.get("/api/alerts")
    async def get_alerts(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        severity: Optional[str] = Query(None, description="Filter: HIGH or LOW"),
    ):
        """Paginated alert history with optional severity filter."""
        alerts = list(state.alerts)

        if severity:
            alerts = [a for a in alerts if a.get("uncertainty") == severity]

        total = len(alerts)
        page = alerts[offset : offset + limit]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "alerts": page,
        }

    @app.get("/api/metrics")
    async def get_metrics():
        """Current model performance metrics snapshot."""
        return {
            "uncertainty_history": state.uncertainty_history[-200:],
            "severity_history": state.severity_history[-200:],
            "latency_history": state.latency_history[-200:],
            "f1_history": state.f1_history[-50:],
            "fdr_history": state.fdr_history[-50:],
            "auc_history": state.auc_history[-50:],
            "set_size_history": state.set_size_history[-200:],
            "drift_history": state.drift_history[-100:],
            "total_alerts": len(state.alerts),
            "soc_state": state.soc_state,
            "severity": state.severity,
        }

    @app.get("/api/metrics/history")
    async def get_metrics_history(
        metric: str = Query("severity", description="Metric name"),
        limit: int = Query(200, ge=1, le=2000),
    ):
        """Historical time series for a specific metric."""
        history_map = {
            "severity": state.severity_history,
            "uncertainty": state.uncertainty_history,
            "latency": state.latency_history,
        }
        data = history_map.get(metric, [])
        return {
            "metric": metric,
            "data": data[-limit:],
            "count": len(data),
        }

    @app.get("/api/explain/{alert_id}")
    async def get_explanation(alert_id: str):
        """SHAP + LIME explanation for a specific alert."""
        # Find alert
        alert = None
        for a in state.alerts:
            if a.get("id") == alert_id:
                alert = a
                break

        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Return explanation data (populated by pipeline if available)
        return {
            "alert_id": alert_id,
            "alert": alert,
            "shap_values": alert.get("shap_values", []),
            "shap_features": alert.get("shap_features", []),
            "lime_explanation": alert.get("lime_explanation", []),
            "top_features": alert.get("top_features", []),
        }

    @app.post("/api/simulate")
    async def trigger_simulation(request: SimulateRequest):
        """Trigger an adversarial simulation with given parameters."""
        sim_id = uuid.uuid4().hex[:8]

        # Notify connected clients
        await ws_manager.broadcast(
            {
                "type": "simulation_started",
                "data": {
                    "sim_id": sim_id,
                    "attack_type": request.attack_type,
                    "epsilon": request.epsilon,
                    "n_samples": request.n_samples,
                },
            },
            topic="state",
        )

        return {
            "sim_id": sim_id,
            "status": "started",
            "attack_type": request.attack_type,
            "epsilon": request.epsilon,
            "n_samples": request.n_samples,
        }

    @app.get("/api/connections")
    async def get_connections():
        """WebSocket connection pool statistics."""
        return ws_manager.get_connection_stats()

    # ------------------------------------------------------------------
    # WebSocket Endpoint
    # ------------------------------------------------------------------

    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        client_id = uuid.uuid4().hex[:8]
        await ws_manager.connect(websocket, client_id)

        try:
            while True:
                data = await websocket.receive_text()
                await ws_manager.handle_client_message(client_id, data)
        except WebSocketDisconnect:
            await ws_manager.disconnect(client_id)

    # ------------------------------------------------------------------
    # Expose shared state and manager for pipeline integration
    # ------------------------------------------------------------------

    app.state.pipeline = state
    app.state.ws_manager = ws_manager

    # ------------------------------------------------------------------
    # Demo mode: continuous synthetic data generator
    # ------------------------------------------------------------------

    DEMO_MODE = os.environ.get("ARDP_MODE", "demo") == "demo"

    _FEATURES = [
        "flow_duration", "pkt_len_mean", "bwd_pkt_len_max",
        "flow_iat_mean", "fwd_iat_total", "bwd_iat_total",
        "psh_flag_cnt", "ack_flag_cnt", "init_fwd_win_byts",
    ]

    epsilon = 0.1  # attack strength; updated by /api/simulate

    async def _demo_loop():
        """Push synthetic alerts and metrics every ~2 s in demo mode."""
        nonlocal epsilon
        tick = 0

        while True:
            await asyncio.sleep(2)
            tick += 1
            t = time.time()

            # Slowly escalate then cool down (sine wave severity)
            phase = (tick % 60) / 60.0                        # 0 → 1 over 2 min
            base_severity = 0.3 + 0.7 * abs(math.sin(math.pi * phase))
            noise = random.uniform(-0.05, 0.05)
            severity = min(1.0, max(0.0, base_severity * (1 + epsilon * 3) + noise))

            # SOC state
            if severity > 0.75:
                soc_state = "CRISIS"
            elif severity > 0.45:
                soc_state = "ELEVATED"
            else:
                soc_state = "STABLE"

            alert_debt = severity * 80 + random.uniform(-5, 5)

            # Update shared state
            state.soc_state = soc_state
            state.severity = round(severity, 3)
            state.alert_debt = round(max(0, alert_debt), 1)
            state.calibration_drift = round(random.uniform(0, 0.12) * epsilon * 5, 3)
            state.disagreement = round(random.uniform(0.0, 0.3) * severity, 3)
            state.n_evaluations += random.randint(80, 150)

            # Synthetic alert
            pred = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
            p = [random.uniform(0, 0.3), random.uniform(0, 0.5), random.uniform(0, 0.4)]
            p_sum = sum(p) or 1
            p = [round(x / p_sum, 3) for x in p]
            uncertainty = "HIGH" if severity > 0.6 else "LOW"
            shap_vals = [round(random.uniform(-0.5, 0.5), 3) for _ in _FEATURES]

            alert = {
                "id": uuid.uuid4().hex[:8],
                "timestamp": t,
                "prediction": pred,
                "probabilities": p,
                "prediction_set": [pred] if uncertainty == "LOW" else [pred, (pred + 1) % 3],
                "uncertainty": uncertainty,
                "latency_ms": round(random.uniform(8, 45), 1),
                "severity": round(severity, 3),
                "shap_values": shap_vals,
                "shap_features": _FEATURES,
                "top_features": sorted(
                    zip(_FEATURES, shap_vals), key=lambda x: abs(x[1]), reverse=True
                )[:5],
            }
            state.push_alert(alert)

            # Update history lists
            state.uncertainty_history.append(round(random.uniform(0.1, 0.9) * severity, 3))
            state.severity_history.append(round(severity, 3))
            state.latency_history.append(alert["latency_ms"])

            f1 = round(max(0, 1 - severity * 0.6 + random.uniform(-0.05, 0.05)), 3)
            state.f1_history.append({"timestamp": t, "value": f1})
            state.fdr_history.append({"timestamp": t, "value": round(severity * 0.4 + random.uniform(0, 0.1), 3)})
            state.auc_history.append({"timestamp": t, "value": round(max(0.5, 1 - severity * 0.3), 3)})
            state.set_size_history.append({"timestamp": t, "value": round(1 + severity, 2)})
            state.drift_history.append({"timestamp": t, "value": state.calibration_drift})

            # Broadcast to WebSocket clients
            await ws_manager.broadcast({"type": "alert", "data": alert}, topic="alerts")
            await ws_manager.broadcast(
                {
                    "type": "state",
                    "data": {
                        "soc_state": soc_state,
                        "severity": state.severity,
                        "alert_debt": state.alert_debt,
                        "calibration_drift": state.calibration_drift,
                        "n_evaluations": state.n_evaluations,
                    },
                },
                topic="state",
            )

    if DEMO_MODE:
        @app.on_event("startup")
        async def start_demo():
            asyncio.create_task(_demo_loop())

        # Patch /api/simulate to update attack strength so the demo reacts
        original_simulate = trigger_simulation

        @app.post("/api/simulate", include_in_schema=False)
        async def trigger_simulation_demo(request: SimulateRequest):
            nonlocal epsilon
            epsilon = request.epsilon
            return await original_simulate(request)

    return app


# ==============================================================================
# STANDALONE SERVER
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
