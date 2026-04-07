"""
================================================================================
KAFKA FLOW PRODUCER — CSV / PCAP REPLAY INTO RAW-TRAFFIC TOPIC
================================================================================
Reads network flow data and publishes to a Kafka topic with configurable
throttling for replay simulation. Falls back to an in-memory queue when
Kafka is unavailable (local dev / test mode).

Includes:
    - Avro-style schema validation (dict-based, no external dependency)
    - Configurable throughput throttling
    - Back-pressure handling via producer buffering
    - Chunked CSV ingestion for memory efficiency
================================================================================
"""

import time
import json
import queue
import logging
import threading
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ── SCHEMA ─────────────────────────────────────────────────────

FLOW_SCHEMA = {
    "type": "record",
    "name": "NetworkFlow",
    "fields": ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
               "protocol", "features", "label"],
}

_REQUIRED_FIELDS = frozenset({"features"})


def validate_flow(record: Dict[str, Any]) -> bool:
    """Basic schema validation for flow records."""
    if not isinstance(record, dict):
        return False
    return _REQUIRED_FIELDS.issubset(record.keys())


# ── IN-MEMORY BUS (Kafka fallback) ────────────────────────────

class _InMemoryBus:
    """Thread-safe in-memory message bus for testing without Kafka."""
    _topics: Dict[str, queue.Queue] = {}
    _lock = threading.Lock()

    @classmethod
    def get_topic(cls, name: str) -> queue.Queue:
        with cls._lock:
            if name not in cls._topics:
                cls._topics[name] = queue.Queue(maxsize=100_000)
            return cls._topics[name]

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._topics.clear()


# ── PRODUCER ───────────────────────────────────────────────────

class FlowProducer:
    """
    Publishes network flow records to a Kafka topic (or in-memory bus).

    Parameters
    ----------
    topic : str
        Target topic name.
    bootstrap_servers : str or None
        Kafka cluster address. None → use in-memory bus.
    throttle_rps : float
        Max records per second (0 = unlimited).
    buffer_size : int
        Internal buffer size for back-pressure.
    """

    def __init__(self, topic: str = "raw-traffic",
                 bootstrap_servers: Optional[str] = None,
                 throttle_rps: float = 0,
                 buffer_size: int = 10_000):
        self.topic = topic
        self.throttle_rps = throttle_rps
        self.buffer_size = buffer_size
        self._kafka_producer = None
        self._use_kafka = False
        self.stats = {"sent": 0, "errors": 0, "bytes": 0}

        if bootstrap_servers:
            try:
                from kafka import KafkaProducer
                self._kafka_producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    buffer_memory=buffer_size * 1024,
                    batch_size=16384,
                    linger_ms=10,
                )
                self._use_kafka = True
                logger.info(f"Kafka producer connected to {bootstrap_servers}")
            except Exception as e:
                logger.warning(f"Kafka unavailable ({e}), falling back to in-memory bus")

        if not self._use_kafka:
            self._bus = _InMemoryBus.get_topic(self.topic)

    def send(self, record: Dict[str, Any]) -> bool:
        """Send a single flow record. Returns True on success."""
        if not validate_flow(record):
            self.stats["errors"] += 1
            return False

        if self._use_kafka:
            self._kafka_producer.send(self.topic, value=record)
        else:
            try:
                self._bus.put_nowait(record)
            except queue.Full:
                self.stats["errors"] += 1
                return False

        payload_size = len(json.dumps(record))
        self.stats["sent"] += 1
        self.stats["bytes"] += payload_size
        return True

    def send_batch(self, records: List[Dict[str, Any]]):
        """Send a batch of records with optional throttling."""
        delay = 1.0 / self.throttle_rps if self.throttle_rps > 0 else 0
        for record in records:
            self.send(record)
            if delay > 0:
                time.sleep(delay)

    def publish_csv(self, csv_path: str, feature_columns: Optional[List[str]] = None,
                    label_column: str = "label", max_rows: Optional[int] = None,
                    chunk_size: int = 10_000):
        """
        Read a CSV file and publish each row as a flow record.

        Uses chunked reading + vectorised extraction for memory efficiency
        and speed (avoids iterrows).

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        feature_columns : list of str, optional
            Feature columns. If None, all columns except label.
        label_column : str
            Label column name.
        max_rows : int, optional
            Max rows to publish (for testing).
        chunk_size : int
            Rows per chunk for streaming reads.
        """
        rows_sent = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            if feature_columns is None:
                feature_columns = [c for c in chunk.columns if c != label_column]

            feats = chunk[feature_columns].values
            labels = chunk[label_column].values if label_column in chunk.columns else np.full(len(chunk), -1)
            now = time.time()

            for i in range(len(chunk)):
                if max_rows and rows_sent >= max_rows:
                    return
                record = {
                    "features": feats[i].tolist(),
                    "label": int(labels[i]),
                    "timestamp": now,
                }
                self.send(record)
                rows_sent += 1

    def flush(self):
        if self._use_kafka and self._kafka_producer:
            self._kafka_producer.flush()

    def close(self):
        self.flush()
        if self._use_kafka and self._kafka_producer:
            self._kafka_producer.close()
