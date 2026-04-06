"""
================================================================================
REAL-TIME INFERENCE SERVICE
================================================================================
Consumes enriched features, runs ensemble prediction + conformal sets,
publishes results. Tracks latency and exposes health check.
================================================================================
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)


class RealtimeInferenceService:
    """
    Runs ensemble prediction + conformal set generation on streaming data.

    Parameters
    ----------
    model : object
        Model with predict_proba(X).
    conformal_engine : object, optional
        Conformal engine with prediction_sets(X, model).
    input_dim : int
        Expected feature dimensionality (raw only, before window agg).
    latency_target_ms : float
        P99 latency target in milliseconds.
    """

    def __init__(self, model, conformal_engine=None, input_dim: int = 10,
                 latency_target_ms: float = 50.0):
        self.model = model
        self.conformal_engine = conformal_engine
        self.input_dim = input_dim
        self.latency_target_ms = latency_target_ms

        self._latency_buffer = deque(maxlen=10_000)
        self.stats = {
            "inferences": 0,
            "alerts": 0,
            "uncertain": 0,
            "latency_violations": 0,
        }
        self._healthy = True

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single enriched feature vector.

        Parameters
        ----------
        features : np.ndarray
            Enriched feature vector (may be longer than input_dim due to agg).

        Returns
        -------
        dict with prediction, probabilities, conformal set, latency.
        """
        t0 = time.perf_counter()

        # Truncate to raw feature dim for model (window aggs are metadata)
        x = features[:self.input_dim].reshape(1, -1).astype(np.float32)
        probs = self.model.predict_proba(x)[0]
        pred_label = int(np.argmax(probs))

        # Conformal prediction set
        pred_set = [pred_label]
        if self.conformal_engine:
            try:
                pred_set = self.conformal_engine.prediction_sets(x, self.model)[0]
            except Exception:
                pred_set = [pred_label]

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_buffer.append(latency_ms)

        # Stats
        self.stats["inferences"] += 1
        if pred_label == 1:
            self.stats["alerts"] += 1
        if len(pred_set) > 1:
            self.stats["uncertain"] += 1
        if latency_ms > self.latency_target_ms:
            self.stats["latency_violations"] += 1

        return {
            "prediction": pred_label,
            "probabilities": probs.tolist(),
            "prediction_set": pred_set,
            "uncertainty": "HIGH" if len(pred_set) > 1 else "LOW",
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference on a batch of enriched records."""
        results = []
        for record in batch:
            features = np.array(record["features"], dtype=np.float32)
            result = self.predict(features)
            result["label_true"] = record.get("label", -1)
            result["timestamp"] = record.get("timestamp", time.time())
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Health & monitoring
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Kubernetes liveness probe compatible health check."""
        latencies = list(self._latency_buffer)
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0

        self._healthy = p99 < self.latency_target_ms * 2  # 2× target = degraded

        return {
            "status": "healthy" if self._healthy else "degraded",
            "inferences": self.stats["inferences"],
            "alerts": self.stats["alerts"],
            "uncertain": self.stats["uncertain"],
            "latency_p50_ms": round(p50, 3),
            "latency_p99_ms": round(p99, 3),
            "latency_target_ms": self.latency_target_ms,
            "violations": self.stats["latency_violations"],
        }

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def get_latency_percentile(self, p: float = 99.0) -> float:
        """Get a specific latency percentile."""
        if not self._latency_buffer:
            return 0.0
        return float(np.percentile(list(self._latency_buffer), p))
