"""
================================================================================
TEST SUITE — STREAMING ARCHITECTURE & CONCEPT DRIFT
================================================================================
≥18 test cases covering:
- Kafka producer/consumer roundtrips (in-memory bus)
- Sliding window aggregation correctness
- Feature store consistency (training vs serving)
- Drift detector sensitivity (ADWIN, Page-Hinkley, KS, MMD)
- Adaptive retraining triggers and validation gate
- Real-time inference latency P99 tracking
- Backpressure and error handling
================================================================================
"""

import pytest
import numpy as np
import time

from src.streaming import FlowProducer, FlowConsumer, RealtimeInferenceService, FeatureStore
from src.streaming.kafka_producer import _InMemoryBus
from src.drift import (
    ConceptDriftEngine, ADWINDetector, PageHinkleyDetector,
    KSDetector, MMDDetector, AdaptiveRetrainingPipeline,
)


# ── MOCK MODEL ─────────────────────────────────────────────────

class MockModel:
    """Simple model that predicts malicious if sum of features > 0."""
    def predict_proba(self, X):
        X = np.atleast_2d(X)
        scores = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.vstack([1 - scores, scores]).T

    def fit(self, X, y):
        pass


@pytest.fixture(autouse=True)
def reset_bus():
    """Reset the in-memory bus between tests to prevent cross-contamination."""
    _InMemoryBus.reset()
    yield
    _InMemoryBus.reset()


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


# ═══════════════════════════════════════════════════════════════
# 1. STREAMING PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestStreamingPipeline:
    def test_producer_consumer_roundtrip(self):
        """Test message integrity through the in-memory bus."""
        producer = FlowProducer(topic="test-roundtrip")
        consumer = FlowConsumer(input_topic="test-roundtrip", output_topic="test-enriched")

        record = {"features": [1.0] * 10, "label": 1, "timestamp": time.time()}
        producer.send(record)

        received = consumer.consume_one(timeout=2.0)
        assert received is not None
        assert "features" in received
        assert len(received["features"]) > 10  # Enriched with window aggs
        assert received["label"] == 1

    def test_producer_send_returns_success(self):
        """Test that send returns True on success."""
        producer = FlowProducer(topic="test-send")
        result = producer.send({"features": [1.0, 2.0]})
        assert result is True
        assert producer.stats["sent"] == 1

    def test_producer_rejects_invalid_record(self):
        """Test that records without 'features' are rejected."""
        producer = FlowProducer(topic="test-invalid")
        result = producer.send({"label": 1})
        assert result is False
        assert producer.stats["errors"] == 1
        assert producer.stats["sent"] == 0

    def test_sliding_window_aggregation(self):
        """Test that window aggregations are computed correctly."""
        consumer = FlowConsumer(
            input_topic="test-agg-in",
            output_topic="test-agg-out",
            window_sizes=[1.0, 5.0],
        )
        feat = [10.0] * 5

        record = {"features": feat, "timestamp": time.time()}
        consumer.process_record(record)
        enriched = consumer.process_record(record)

        # Raw(5) + [Mean(5) + Std(5) + Count(1)] × 2 windows = 5 + 22 = 27
        assert len(enriched["features"]) == 27
        # Mean should be 10.0 for identical features
        assert enriched["features"][5] == pytest.approx(10.0, abs=0.1)

    def test_consumer_batch_consumption(self):
        """Test consuming multiple records in batch mode."""
        producer = FlowProducer(topic="test-batch")
        consumer = FlowConsumer(input_topic="test-batch", output_topic="test-batch-out")

        for i in range(5):
            producer.send({"features": [float(i)] * 10, "label": 0})

        batch = consumer.consume_batch(n=5, timeout=2.0)
        assert len(batch) == 5

    def test_inference_service_latency(self, mock_model):
        """Test P99 latency tracking."""
        service = RealtimeInferenceService(mock_model, input_dim=10)
        batch = [{"features": [0.0] * 20, "label": 0} for _ in range(10)]
        service.predict_batch(batch)

        health = service.health_check()
        assert health["inferences"] == 10
        assert health["latency_p99_ms"] > 0
        assert health["status"] == "healthy"

    def test_inference_service_single_predict(self, mock_model):
        """Test single-record prediction with dict input."""
        service = RealtimeInferenceService(mock_model, input_dim=10)
        result = service.predict({"features": [0.5] * 20})
        assert "prediction" in result
        assert "probabilities" in result
        assert result["prediction"] in (0, 1)

    def test_inference_batch_vectorised(self, mock_model):
        """Test that batch prediction returns correct structure."""
        service = RealtimeInferenceService(mock_model, input_dim=10)
        batch = [{"features": [1.0] * 10, "label": 1, "timestamp": 123.0} for _ in range(3)]
        results = service.predict_batch(batch)

        assert len(results) == 3
        for r in results:
            assert "prediction" in r
            assert "label_true" in r
            assert "timestamp" in r


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE STORE TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureStore:
    def test_lru_cache_behavior(self):
        """Test that LRU eviction works correctly."""
        fs = FeatureStore(max_memory_items=2)
        fs.put("k1", np.array([1.0]))
        fs.put("k2", np.array([2.0]))
        fs.put("k3", np.array([3.0]))

        assert fs.get("k1") is None  # Evicted
        assert fs.get("k2") is not None

    def test_skew_detection(self):
        """Test training-serving skew detection."""
        fs = FeatureStore()
        rng = np.random.RandomState(42)
        train_X = rng.normal(0, 1, (100, 5))
        serve_X = rng.normal(5, 1, (100, 5))  # Shifted mean

        skew = fs.check_skew(train_X, serve_X, threshold=0.1)
        assert skew["n_skewed"] > 0
        assert skew["max_z"] > 1.0

    def test_feature_store_persistence_serialization(self):
        """Test put/get with metadata roundtrip."""
        fs = FeatureStore()
        k = "node-1"
        f = np.array([1.2, 3.4], dtype=np.float32)
        fs.put(k, f, metadata={"id": 123})

        loaded = fs.get_with_metadata(k)
        assert np.allclose(loaded["features"], [1.2, 3.4])
        assert loaded["metadata"]["id"] == 123
        assert loaded["schema_version"] == "v1.0"

    def test_get_batch_returns_list(self):
        """Test batch retrieval returns correct types."""
        fs = FeatureStore()
        fs.put("a", np.array([1.0, 2.0]))
        fs.put("b", np.array([3.0, 4.0]))

        results = fs.get_batch(["a", "b", "missing"], expected_dim=2)
        assert len(results) == 3
        assert np.allclose(results[0], [1.0, 2.0])
        assert np.allclose(results[1], [3.0, 4.0])
        assert np.all(np.isnan(results[2]))

    def test_schema_uses_sha256(self):
        """Test that schema registry uses SHA-256 checksums."""
        fs = FeatureStore()
        schema = fs.get_schema("v1.0")
        assert schema is not None
        assert len(schema["checksum"]) == 64  # SHA-256 hex digest length


# ═══════════════════════════════════════════════════════════════
# 3. DRIFT DETECTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestDriftDetection:
    def test_adwin_sensitivity(self):
        """Test ADWIN detects mean shift in stream."""
        detector = ADWINDetector(delta=0.01, window_size=50)
        for _ in range(50):
            detector.update(0.1)
        drift = False
        for _ in range(50):
            if detector.update(0.9):
                drift = True
                break
        assert drift is True

    def test_adwin_no_false_positive_on_stable(self):
        """Test ADWIN doesn't fire on stable stream."""
        detector = ADWINDetector(delta=0.01, window_size=100)
        rng = np.random.RandomState(42)
        drifts = sum(1 for _ in range(200) if detector.update(0.5 + rng.normal(0, 0.01)))
        assert drifts == 0

    def test_adwin_reset(self):
        """Test ADWIN reset clears state."""
        detector = ADWINDetector()
        for _ in range(50):
            detector.update(0.5)
        detector.reset()
        assert len(detector.stream) == 0

    def test_ks_detector_with_bonferroni(self):
        """Test KS detects distributional shift with Bonferroni correction."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, (200, 3))
        curr = rng.normal(10, 1, (200, 3))
        detector = KSDetector(ref)
        assert detector.detect(curr) is True
        assert len(detector.drifted_features) > 0

    def test_mmd_detector_rbf_kernel(self):
        """Test MMD with RBF kernel detects distribution shift."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, (200, 3))
        curr = rng.normal(5, 1, (200, 3))
        detector = MMDDetector(ref)
        assert detector.detect(curr) is True

    def test_drift_engine_consensus(self):
        """Test multi-signal consensus requires ≥2 detectors to agree."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, (200, 3))
        curr = rng.normal(10, 1, (200, 3))
        engine = ConceptDriftEngine(ref)
        # KS + MMD should both fire with large shift
        assert engine.evaluate(curr, [0.0] * 10) is True

    def test_drift_engine_no_drift_on_clean_data(self):
        """Test no false positive on identical distributions."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, (200, 3))
        engine = ConceptDriftEngine(ref)
        assert engine.evaluate(ref, [0.0] * 10) is False

    def test_drift_engine_tracks_results(self):
        """Test that last_results property returns per-detector results."""
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, (200, 3))
        engine = ConceptDriftEngine(ref)
        engine.evaluate(ref, [0.0] * 10)
        results = engine.last_results
        assert "adwin" in results
        assert "page_hinkley" in results
        assert "ks" in results
        assert "mmd" in results

    def test_ph_detector_resets(self):
        """Test Page-Hinkley reset clears state."""
        detector = PageHinkleyDetector(lambda_threshold=1)
        detector.update(100.0)
        assert detector.n > 0
        detector.reset()
        assert detector.n == 0


# ═══════════════════════════════════════════════════════════════
# 4. ADAPTIVE RETRAINING TESTS
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveRetraining:
    def test_validation_gate_promotion(self, mock_model, sample_data):
        """Test that a good model is promoted when gate is 0."""
        X, y = sample_data
        orchestrator = MockModel()
        retrainer = AdaptiveRetrainingPipeline(orchestrator, validation_gate=0.0)

        new_model = retrainer.retrain(mock_model, X, y, X, y)
        assert new_model is orchestrator  # Promoted

    def test_validation_gate_rollback(self, mock_model, sample_data):
        """Test rollback when new model doesn't improve enough."""
        X, y = sample_data

        class BadModel:
            def predict_proba(self, X):
                return np.full((len(X), 2), 0.5)
            def fit(self, X, y):
                pass

        bad = BadModel()
        retrainer = AdaptiveRetrainingPipeline(bad, validation_gate=0.5)
        result = retrainer.retrain(mock_model, X, y, X, y)
        # Should rollback to mock_model since bad model can't beat it by 0.5
        assert result is mock_model

    def test_retraining_history_tracking(self, mock_model, sample_data):
        """Test that retraining history is properly recorded."""
        X, y = sample_data
        orchestrator = MockModel()
        retrainer = AdaptiveRetrainingPipeline(orchestrator, validation_gate=0.0)

        retrainer.retrain(mock_model, X, y, X, y)
        assert retrainer.n_retrains == 1
        assert len(retrainer.history) == 1

        record = retrainer.history[0]
        assert "f1_old" in record
        assert "f1_new" in record
        assert "gain" in record
        assert "promoted" in record
        assert record["promoted"] is True

    def test_uncertainty_sampling_selects_subset(self, mock_model, sample_data):
        """Test that uncertainty sampling selects uncertain samples."""
        X, y = sample_data
        retrainer = AdaptiveRetrainingPipeline(
            MockModel(),
            active_learning_strategy="uncertainty",
            uncertainty_percentile=50.0,
        )
        X_sel, y_sel = retrainer._select_samples(mock_model, X, y)
        # Should select ~50% of samples
        assert len(X_sel) < len(X)
        assert len(X_sel) > 0


# ═══════════════════════════════════════════════════════════════
# 5. HEALTH & ERROR HANDLING TESTS
# ═════════════════════════════════════════════��═════════════════

def test_inference_health_degraded_on_high_latency(mock_model):
    """Test health check reports degraded when latency exceeds target."""
    service = RealtimeInferenceService(mock_model, latency_target_ms=0.0001)
    service.predict(np.array([0.0] * 10))
    health = service.health_check()
    assert health["status"] == "degraded"


def test_producer_stats_track_bytes():
    """Test that producer tracks byte throughput."""
    producer = FlowProducer(topic="test-bytes")
    producer.send({"features": [1.0, 2.0, 3.0]})
    assert producer.stats["bytes"] > 0


def test_consumer_handles_empty_queue():
    """Test that consumer returns None on empty queue."""
    consumer = FlowConsumer(input_topic="test-empty", output_topic="test-empty-out")
    result = consumer.consume_one(timeout=0.1)
    assert result is None
