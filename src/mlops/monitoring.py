"""
================================================================================
PRODUCTION MONITOR — PROMETHEUS METRICS & GRAFANA DASHBOARDS
================================================================================
Exposes pipeline metrics via Prometheus client library and generates
Grafana dashboard JSON templates.

Falls back to in-memory metric collection when prometheus_client is
unavailable.

Metrics exported:
    - ids_prediction_latency_ms (histogram)
    - ids_alert_rate (counter)
    - ids_conformal_set_size_avg (gauge)
    - ids_drift_score (gauge)
    - ids_model_version (info)
    - ids_retraining_count (counter)
================================================================================
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

_PROM_AVAILABLE = False
try:
    from prometheus_client import (
        Histogram,
        Counter,
        Gauge,
        Info,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except ImportError:
    pass


class ProductionMonitor:
    """
    Pipeline monitoring with Prometheus metrics + Grafana dashboard generation.

    Parameters
    ----------
    port : int
        Port for Prometheus HTTP metrics endpoint.
    enable_server : bool
        Whether to start the HTTP metrics server on init.
    dashboard_output_dir : str
        Directory for generated Grafana dashboard JSON.
    """

    def __init__(
        self,
        port: int = 9090,
        enable_server: bool = False,
        dashboard_output_dir: str = "reports/dashboards",
    ):
        self.port = port
        self.dashboard_dir = Path(dashboard_output_dir)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

        self._use_prometheus = _PROM_AVAILABLE
        self._local_metrics: Dict[str, List] = defaultdict(list)

        if self._use_prometheus:
            self._latency = Histogram(
                "ids_prediction_latency_ms",
                "Prediction latency in milliseconds",
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            )
            self._alert_count = Counter(
                "ids_alert_rate",
                "Total alerts generated",
            )
            self._set_size = Gauge(
                "ids_conformal_set_size_avg",
                "Average conformal prediction set size",
            )
            self._drift_score = Gauge(
                "ids_drift_score",
                "Current concept drift score",
            )
            self._model_info = Info(
                "ids_model",
                "Current model version information",
            )
            self._retrain_count = Counter(
                "ids_retraining_count",
                "Total model retraining events",
            )

            if enable_server:
                try:
                    start_http_server(port)
                    logger.info(f"Prometheus metrics server on :{port}")
                except Exception as e:
                    logger.warning(f"Failed to start metrics server: {e}")
        else:
            logger.info("Prometheus unavailable, using in-memory metrics")

    # ------------------------------------------------------------------
    # Record metrics
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float):
        """Record a prediction latency observation."""
        if self._use_prometheus:
            self._latency.observe(latency_ms)
        self._local_metrics["latency_ms"].append(
            {"value": latency_ms, "timestamp": time.time()}
        )

    def record_alert(self):
        """Increment alert counter."""
        if self._use_prometheus:
            self._alert_count.inc()
        self._local_metrics["alerts"].append({"timestamp": time.time()})

    def record_set_size(self, avg_size: float):
        """Update average conformal set size gauge."""
        if self._use_prometheus:
            self._set_size.set(avg_size)
        self._local_metrics["set_size"].append(
            {"value": avg_size, "timestamp": time.time()}
        )

    def record_drift_score(self, score: float):
        """Update concept drift score gauge."""
        if self._use_prometheus:
            self._drift_score.set(score)
        self._local_metrics["drift_score"].append(
            {"value": score, "timestamp": time.time()}
        )

    def record_model_version(self, version: str, stage: str = "production"):
        """Update model version info."""
        if self._use_prometheus:
            self._model_info.info({"version": version, "stage": stage})
        self._local_metrics["model_version"].append(
            {"version": version, "stage": stage, "timestamp": time.time()}
        )

    def record_retraining(self):
        """Increment retraining counter."""
        if self._use_prometheus:
            self._retrain_count.inc()
        self._local_metrics["retraining"].append({"timestamp": time.time()})

    # ------------------------------------------------------------------
    # Batch recording from inference service
    # ------------------------------------------------------------------

    def record_inference_batch(self, results: List[Dict[str, Any]]):
        """Record metrics from a batch of inference results."""
        for r in results:
            self.record_latency(r.get("latency_ms", 0))
            if r.get("prediction") == 1:
                self.record_alert()

        set_sizes = [len(r.get("prediction_set", [1])) for r in results]
        if set_sizes:
            self.record_set_size(sum(set_sizes) / len(set_sizes))

    # ------------------------------------------------------------------
    # Alert rules
    # ------------------------------------------------------------------

    def check_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Evaluate alert rules and return any that are firing.

        Rules:
            - latency_high: P99 latency > 100ms
            - uncertainty_high: avg set size > 1.5
            - drift_detected: drift score > 0.5
        """
        alerts = []

        latencies = [m["value"] for m in self._local_metrics.get("latency_ms", [])]
        if latencies:
            import numpy as np

            p99 = float(np.percentile(latencies[-1000:], 99))
            if p99 > 100:
                alerts.append(
                    {
                        "rule": "latency_high",
                        "severity": "warning",
                        "message": f"P99 latency {p99:.1f}ms > 100ms threshold",
                        "value": p99,
                    }
                )

        set_sizes = self._local_metrics.get("set_size", [])
        if set_sizes:
            latest = set_sizes[-1]["value"]
            if latest > 1.5:
                alerts.append(
                    {
                        "rule": "uncertainty_high",
                        "severity": "critical",
                        "message": f"Avg set size {latest:.2f} > 1.5 threshold",
                        "value": latest,
                    }
                )

        drift_scores = self._local_metrics.get("drift_score", [])
        if drift_scores:
            latest = drift_scores[-1]["value"]
            if latest > 0.5:
                alerts.append(
                    {
                        "rule": "drift_detected",
                        "severity": "warning",
                        "message": f"Drift score {latest:.3f} > 0.5 threshold",
                        "value": latest,
                    }
                )

        return alerts

    # ------------------------------------------------------------------
    # Grafana dashboard generation
    # ------------------------------------------------------------------

    def generate_grafana_dashboard(self) -> Dict:
        """Generate a Grafana dashboard JSON template for IDS monitoring."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "SOC Pipeline — Production Monitoring",
                "tags": ["ids", "security", "mlops"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Prediction Latency (P50/P99)",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.50, rate(ids_prediction_latency_ms_bucket[5m]))"
                            },
                            {
                                "expr": "histogram_quantile(0.99, rate(ids_prediction_latency_ms_bucket[5m]))"
                            },
                        ],
                    },
                    {
                        "title": "Alert Rate",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {"expr": "rate(ids_alert_rate_total[5m])"},
                        ],
                    },
                    {
                        "title": "Conformal Set Size (Avg)",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
                        "targets": [
                            {"expr": "ids_conformal_set_size_avg"},
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "steps": [
                                        {"value": 0, "color": "green"},
                                        {"value": 1.1, "color": "orange"},
                                        {"value": 1.5, "color": "red"},
                                    ]
                                }
                            }
                        },
                    },
                    {
                        "title": "Drift Score",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
                        "targets": [
                            {"expr": "ids_drift_score"},
                        ],
                    },
                    {
                        "title": "Model Retraining Events",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
                        "targets": [
                            {"expr": "ids_retraining_count_total"},
                        ],
                    },
                    {
                        "title": "Model Version",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
                        "targets": [
                            {"expr": "ids_model_info"},
                        ],
                    },
                ],
                "refresh": "10s",
                "time": {"from": "now-1h", "to": "now"},
            },
        }

        output_path = self.dashboard_dir / "grafana_dashboard.json"
        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2)

        logger.info(f"Grafana dashboard saved to {output_path}")
        return dashboard

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of all collected metrics."""
        snapshot = {}
        for key, values in self._local_metrics.items():
            if values:
                snapshot[key] = {
                    "count": len(values),
                    "latest": values[-1],
                }
        return snapshot
