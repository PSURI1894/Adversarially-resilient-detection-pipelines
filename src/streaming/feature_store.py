"""
================================================================================
FEATURE STORE — IN-MEMORY + REDIS-BACKED FEATURE SERVING
================================================================================
Ensures training–serving consistency. Falls back to pure in-memory
when Redis is unavailable.

Includes:
    - In-memory LRU cache for hot features
    - Redis backend for persistence (optional)
    - Feature versioning with schema tracking
    - Point-in-time correctness for backfilling
    - Training–serving skew detection
================================================================================
"""

import time
import hashlib
import json
import logging
import numpy as np
from collections import OrderedDict
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Dual-layer feature store: in-memory LRU cache + optional Redis.

    Parameters
    ----------
    redis_url : str or None
        Redis connection URL. None → pure in-memory.
    max_memory_items : int
        Max items in the in-memory LRU cache.
    schema_version : str
        Feature schema version for compatibility tracking.
    """

    def __init__(self, redis_url: Optional[str] = None,
                 max_memory_items: int = 50_000,
                 schema_version: str = "v1.0"):
        self.schema_version = schema_version
        self.max_memory_items = max_memory_items
        self._cache = OrderedDict()
        self._redis = None
        self._schema_registry: Dict[str, Dict] = {}

        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"FeatureStore connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), using in-memory only")
                self._redis = None

        # Register initial schema
        self.register_schema(schema_version, {"type": "default"})

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def put(self, key: str, features: np.ndarray, metadata: Optional[Dict] = None):
        """Store a feature vector."""
        entry = {
            "features": features.tolist(),
            "timestamp": time.time(),
            "schema_version": self.schema_version,
            "metadata": metadata or {},
        }

        # In-memory LRU
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = entry
        if len(self._cache) > self.max_memory_items:
            self._cache.popitem(last=False)

        # Redis persistence
        if self._redis:
            try:
                self._redis.set(f"fs:{key}", json.dumps(entry), ex=86400)
            except Exception as e:
                logger.warning(f"Redis write failed: {e}")

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a feature vector."""
        # Check memory first
        if key in self._cache:
            self._cache.move_to_end(key)
            return np.array(self._cache[key]["features"], dtype=np.float32)

        # Fall back to Redis
        if self._redis:
            try:
                data = self._redis.get(f"fs:{key}")
                if data:
                    entry = json.loads(data)
                    self._cache[key] = entry
                    return np.array(entry["features"], dtype=np.float32)
            except Exception:
                pass

        return None

    def get_with_metadata(self, key: str) -> Optional[Dict]:
        """Retrieve feature vector with full metadata."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if self._redis:
            try:
                data = self._redis.get(f"fs:{key}")
                if data:
                    entry = json.loads(data)
                    self._cache[key] = entry
                    return entry
            except Exception:
                pass

        return None

    def delete(self, key: str):
        """Remove a feature from the store."""
        self._cache.pop(key, None)
        if self._redis:
            try:
                self._redis.delete(f"fs:{key}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def put_batch(self, keys: List[str], features_batch: np.ndarray):
        """Store a batch of feature vectors."""
        for i, key in enumerate(keys):
            self.put(key, features_batch[i])

    def get_batch(self, keys: List[str]) -> np.ndarray:
        """Retrieve a batch, returns NaN for missing keys."""
        results = []
        for key in keys:
            feats = self.get(key)
            if feats is not None:
                results.append(feats)
            else:
                results.append(np.full(1, np.nan))
        return results

    # ------------------------------------------------------------------
    # Feature versioning / schema
    # ------------------------------------------------------------------

    def register_schema(self, version: str, schema: Dict):
        """Register a feature schema version."""
        self._schema_registry[version] = {
            "schema": schema,
            "registered_at": time.time(),
            "checksum": hashlib.md5(json.dumps(schema).encode()).hexdigest(),
        }

    def get_schema(self, version: Optional[str] = None) -> Optional[Dict]:
        version = version or self.schema_version
        return self._schema_registry.get(version)

    # ------------------------------------------------------------------
    # Training–serving consistency
    # ------------------------------------------------------------------

    def check_skew(self, training_features: np.ndarray,
                   serving_features: np.ndarray,
                   threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect training-serving feature skew.

        Compares per-feature mean/std between training and serving
        distributions. Returns features that exceed the threshold.
        """
        train_mean = np.mean(training_features, axis=0)
        train_std = np.std(training_features, axis=0) + 1e-8
        serve_mean = np.mean(serving_features, axis=0)

        z_scores = np.abs(serve_mean - train_mean) / train_std
        skewed = np.where(z_scores > threshold)[0]

        return {
            "skewed_features": skewed.tolist(),
            "z_scores": z_scores.tolist(),
            "max_z": float(np.max(z_scores)),
            "n_skewed": len(skewed),
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Point-in-time retrieval
    # ------------------------------------------------------------------

    def get_point_in_time(self, key: str, as_of: float) -> Optional[np.ndarray]:
        """
        Retrieve features as they existed at a specific timestamp.
        Only works for in-memory entries (for backfilling).
        """
        entry = self._cache.get(key)
        if entry and entry["timestamp"] <= as_of:
            return np.array(entry["features"], dtype=np.float32)
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._cache)
