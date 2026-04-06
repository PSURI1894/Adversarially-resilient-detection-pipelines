"""
================================================================================
KAFKA FLOW CONSUMER — FEATURE ENGINEERING & SLIDING WINDOW AGGREGATION
================================================================================
Subscribes to raw-traffic topic, applies in-flight feature engineering with
sliding window temporal aggregation, and publishes enriched features.

Includes:
    - Sliding window aggregation (1-min, 5-min, 15-min)
    - Consumer group management for horizontal scaling
    - In-memory bus fallback for local dev/testing
================================================================================
"""

import time
import json
import queue
import logging
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)


class SlidingWindowAggregator:
    """
    Maintains sliding windows of configurable durations and computes
    temporal aggregation statistics over incoming flow features.

    Parameters
    ----------
    window_sizes : list of float
        Window durations in seconds (e.g., [60, 300, 900]).
    n_features : int
        Number of raw features per flow.
    """

    def __init__(self, window_sizes: List[float] = None, n_features: int = 10):
        self.window_sizes = window_sizes or [60.0, 300.0, 900.0]
        self.n_features = n_features
        self._buffers: Dict[float, deque] = {
            w: deque() for w in self.window_sizes
        }

    def update(self, features: np.ndarray, timestamp: float):
        """Add a new observation and prune expired entries."""
        for w in self.window_sizes:
            self._buffers[w].append((timestamp, features))
            # Prune expired
            while self._buffers[w] and (timestamp - self._buffers[w][0][0]) > w:
                self._buffers[w].popleft()

    def aggregate(self) -> np.ndarray:
        """
        Compute aggregation statistics across all windows.

        Returns array of: [mean, std, count] × each window × each feature
        This gets appended to the raw feature vector.
        """
        agg = []
        for w in self.window_sizes:
            if len(self._buffers[w]) == 0:
                # No data in window — return zeros
                agg.extend([0.0] * (self.n_features * 2 + 1))
            else:
                data = np.array([f for _, f in self._buffers[w]])
                agg.extend(np.mean(data, axis=0).tolist())
                agg.extend(np.std(data, axis=0).tolist())
                agg.append(float(len(data)))
        return np.array(agg, dtype=np.float32)


class FlowConsumer:
    """
    Subscribes to a Kafka topic (or in-memory bus), applies feature engineering,
    and publishes enriched features.

    Parameters
    ----------
    input_topic : str
        Topic to consume from.
    output_topic : str
        Topic to publish enriched features to.
    bootstrap_servers : str or None
        Kafka address. None → in-memory bus.
    group_id : str
        Consumer group ID for horizontal scaling.
    feature_transform : callable, optional
        Custom transform fn: raw_features → enriched_features.
    window_sizes : list of float
        Sliding window sizes for temporal aggregation.
    """

    def __init__(self, input_topic: str = "raw-traffic",
                 output_topic: str = "enriched-features",
                 bootstrap_servers: Optional[str] = None,
                 group_id: str = "ids-consumer-group",
                 feature_transform: Optional[Callable] = None,
                 window_sizes: Optional[List[float]] = None):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group_id = group_id
        self.feature_transform = feature_transform
        self._use_kafka = False
        self._running = False

        self.aggregator = SlidingWindowAggregator(window_sizes)
        self.stats = {"consumed": 0, "published": 0, "errors": 0}

        if bootstrap_servers:
            try:
                from kafka import KafkaConsumer, KafkaProducer
                self._consumer = KafkaConsumer(
                    input_topic,
                    bootstrap_servers=bootstrap_servers,
                    group_id=group_id,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    auto_offset_reset="latest",
                )
                self._producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
                self._use_kafka = True
            except Exception as e:
                logger.warning(f"Kafka unavailable ({e}), using in-memory bus")

        if not self._use_kafka:
            from src.streaming.kafka_producer import _InMemoryBus
            self._input_bus = _InMemoryBus.get_topic(self.input_topic)
            self._output_bus = _InMemoryBus.get_topic(self.output_topic)

    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single flow record: transform features + add window aggs.
        """
        try:
            raw = np.array(record["features"], dtype=np.float32)
            ts = record.get("timestamp", time.time())

            # Apply custom transform if provided
            if self.feature_transform:
                raw = self.feature_transform(raw)

            # Update sliding windows
            self.aggregator.update(raw, ts)
            window_feats = self.aggregator.aggregate()

            enriched = {
                "features": np.concatenate([raw, window_feats]).tolist(),
                "label": record.get("label", -1),
                "timestamp": ts,
                "raw_dim": len(raw),
            }
            return enriched
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Processing error: {e}")
            return None

    def consume_one(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Consume and process a single record (pull mode)."""
        if self._use_kafka:
            # kafka poll returns batches
            msgs = self._consumer.poll(timeout_ms=int(timeout * 1000), max_records=1)
            for _, records in msgs.items():
                for msg in records:
                    self.stats["consumed"] += 1
                    enriched = self.process_record(msg.value)
                    if enriched:
                        self._producer.send(self.output_topic, value=enriched)
                        self.stats["published"] += 1
                    return enriched
        else:
            try:
                record = self._input_bus.get(timeout=timeout)
                self.stats["consumed"] += 1
                enriched = self.process_record(record)
                if enriched:
                    self._output_bus.put_nowait(enriched)
                    self.stats["published"] += 1
                return enriched
            except queue.Empty:
                return None

    def consume_batch(self, n: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Consume up to n records."""
        results = []
        deadline = time.time() + timeout
        while len(results) < n and time.time() < deadline:
            record = self.consume_one(timeout=0.1)
            if record is not None:
                results.append(record)
        return results

    def run(self, max_records: Optional[int] = None):
        """Run continuous consumption loop."""
        self._running = True
        count = 0
        while self._running:
            self.consume_one()
            count += 1
            if max_records and count >= max_records:
                break

    def stop(self):
        self._running = False
