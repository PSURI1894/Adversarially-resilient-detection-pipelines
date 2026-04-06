"""
================================================================================
TEST SUITE — EXPLAINABILITY & ADVERSARIAL DETECTION (XAI LAYER)
================================================================================
≥15 test cases covering SHAP, LIME, Attribution Fingerprinting,
Feature Sensitivity Analysis, and Incident Report Generation.
================================================================================
"""
import pytest
import numpy as np
import os
import json
import tempfile


# ── MOCK MODEL ─────────────────────────────────────────────────

class MockModel:
    """Returns probabilities based on feature sum."""
    def predict_proba(self, X):
        X = np.atleast_2d(X)
        scores = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.vstack([1 - scores, scores]).T


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 8).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


@pytest.fixture
def feature_names():
    return ["flow_iat_mean", "pkt_count", "bytes_sent", "duration",
            "src_port", "dst_port", "flags", "protocol"]


# ═══════════════════════════════════════════════════════════════
# 1. SHAP EXPLAINER TESTS
# ═══════════════════════════════════════════════════════════════

class TestSHAPExplainer:
    def test_local_explanation_keys(self, mock_model, sample_data, feature_names):
        from src.explainability.shap_engine import SHAPExplainer
        X, _ = sample_data
        explainer = SHAPExplainer(mock_model, mode="kernel",
                                  background_data=X[:20], feature_names=feature_names)
        result = explainer.explain_instance(X[0])
        assert "shap_values" in result
        assert "base_value" in result
        assert "prediction" in result
        assert "top_features" in result

    def test_shap_values_shape(self, mock_model, sample_data):
        from src.explainability.shap_engine import SHAPExplainer
        X, _ = sample_data
        explainer = SHAPExplainer(mock_model, mode="kernel", background_data=X[:20])
        result = explainer.explain_instance(X[0])
        assert len(result["shap_values"]) == X.shape[1]

    def test_batch_explanation_shape(self, mock_model, sample_data):
        from src.explainability.shap_engine import SHAPExplainer
        X, _ = sample_data
        explainer = SHAPExplainer(mock_model, mode="kernel", background_data=X[:20])
        vals = explainer.explain_batch(X[:10])
        assert vals.shape == (10, X.shape[1])

    def test_global_importance_ranking(self, mock_model, sample_data, feature_names):
        from src.explainability.shap_engine import SHAPExplainer
        X, _ = sample_data
        explainer = SHAPExplainer(mock_model, mode="kernel",
                                  background_data=X[:20], feature_names=feature_names)
        imp = explainer.global_importance(X[:15])
        assert len(imp) == len(feature_names)
        # Values should be sorted descending
        values = list(imp.values())
        assert values[0] >= values[-1]

    def test_shap_completeness_approximation(self, mock_model, sample_data):
        """SHAP values should approximately sum to prediction - base_value."""
        from src.explainability.shap_engine import SHAPExplainer
        X, _ = sample_data
        explainer = SHAPExplainer(mock_model, mode="kernel", background_data=X[:30])
        result = explainer.explain_instance(X[0])
        shap_sum = np.sum(result["shap_values"])
        margin = result["prediction"] - result["base_value"]
        # Allow generous tolerance for permutation approximation
        assert abs(shap_sum - margin) < 0.5


# ═══════════════════════════════════════════════════════════════
# 2. LIME EXPLAINER TESTS
# ═══════════════════════════════════════════════════════════════

class TestLIMEExplainer:
    def test_local_explanation_keys(self, mock_model, sample_data, feature_names):
        from src.explainability.lime_engine import LIMEExplainer
        X, _ = sample_data
        explainer = LIMEExplainer(mock_model, feature_names=feature_names)
        result = explainer.explain_instance(X[0])
        assert "coefficients" in result
        assert "fidelity" in result
        assert "top_features" in result
        assert "prediction" in result

    def test_fidelity_positive(self, mock_model, sample_data):
        from src.explainability.lime_engine import LIMEExplainer
        X, _ = sample_data
        explainer = LIMEExplainer(mock_model, n_samples=200)
        result = explainer.explain_instance(X[0])
        # Fidelity should be > 0 (surrogate captures some signal)
        assert result["fidelity"] > 0.0

    def test_immutable_features_respected(self, mock_model, sample_data):
        from src.explainability.lime_engine import LIMEExplainer
        X, _ = sample_data
        # Mark features 4,5 as immutable
        explainer = LIMEExplainer(mock_model, immutable_features=[4, 5])
        result = explainer.explain_instance(X[0])
        # Should still produce explanation without crashing
        assert len(result["coefficients"]) == X.shape[1]

    def test_batch_explanations(self, mock_model, sample_data):
        from src.explainability.lime_engine import LIMEExplainer
        X, _ = sample_data
        explainer = LIMEExplainer(mock_model, n_samples=100)
        results = explainer.explain_batch(X[:5])
        assert len(results) == 5
        assert all("fidelity" in r for r in results)

    def test_fidelity_assessment(self, mock_model, sample_data):
        from src.explainability.lime_engine import LIMEExplainer
        X, _ = sample_data
        explainer = LIMEExplainer(mock_model, n_samples=200)
        assess = explainer.assess_fidelity(X, n_trials=5)
        assert "mean_fidelity" in assess
        assert assess["mean_fidelity"] > -1  # can be negative R² in extreme cases


# ═══════════════════════════════════════════════════════════════
# 3. ATTRIBUTION FINGERPRINT DETECTOR TESTS
# ═══════════════════════════════════════════════════════════════

class TestAttributionFingerprintDetector:
    def test_fit_and_score(self, mock_model, sample_data):
        from src.explainability.shap_engine import SHAPExplainer
        from src.explainability.adversarial_detector import AttributionFingerprintDetector
        X, _ = sample_data
        shap_exp = SHAPExplainer(mock_model, mode="kernel", background_data=X[:20])
        detector = AttributionFingerprintDetector(shap_exp, n_components=2)
        detector.fit(X[:50])
        assert detector.gmm is not None
        assert detector.threshold is not None

    def test_detect_returns_structure(self, mock_model, sample_data):
        from src.explainability.shap_engine import SHAPExplainer
        from src.explainability.adversarial_detector import AttributionFingerprintDetector
        X, _ = sample_data
        shap_exp = SHAPExplainer(mock_model, mode="kernel", background_data=X[:20])
        detector = AttributionFingerprintDetector(shap_exp, n_components=2)
        detector.fit(X[:50])
        result = detector.detect(X[50:60])
        assert "is_adversarial" in result
        assert "scores" in result
        assert "n_flagged" in result
        assert len(result["scores"]) == 10

    def test_adversarial_samples_flagged_more(self, mock_model, sample_data):
        """Perturbed samples should have higher anomaly scores on average."""
        from src.explainability.shap_engine import SHAPExplainer
        from src.explainability.adversarial_detector import AttributionFingerprintDetector
        X, _ = sample_data
        shap_exp = SHAPExplainer(mock_model, mode="kernel", background_data=X[:20])
        detector = AttributionFingerprintDetector(shap_exp, n_components=2,
                                                   threshold_percentile=90)
        detector.fit(X[:80])
        # Clean scores
        clean_scores = detector.score(X[80:100])
        # "Adversarial" = heavily perturbed
        X_adv = X[80:100] + np.random.normal(0, 5, size=X[80:100].shape)
        adv_scores = detector.score(X_adv)
        # Adversarial should have higher mean score
        assert np.mean(adv_scores) > np.mean(clean_scores)


# ═══════════════════════════════════════════════════════════════
# 4. FEATURE SENSITIVITY ANALYZER TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeatureSensitivityAnalyzer:
    def test_sensitivity_shape(self, mock_model, sample_data):
        from src.explainability.adversarial_detector import FeatureSensitivityAnalyzer
        X, _ = sample_data
        analyzer = FeatureSensitivityAnalyzer(mock_model)
        sens = analyzer.compute_sensitivity(X[:20])
        assert sens.shape == (X.shape[1],)
        assert np.all(sens >= 0)

    def test_vulnerability_report_keys(self, mock_model, sample_data, feature_names):
        from src.explainability.adversarial_detector import FeatureSensitivityAnalyzer
        X, _ = sample_data
        analyzer = FeatureSensitivityAnalyzer(mock_model, feature_names=feature_names)
        report = analyzer.vulnerability_report(X[:20], top_k=3)
        assert "sensitivities" in report
        assert "most_vulnerable" in report
        assert "recommendations" in report
        assert len(report["most_vulnerable"]) == 3
        assert len(report["recommendations"]) == 3


# ═══════════════════════════════════════════════════════════════
# 5. INCIDENT REPORT GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════

class TestIncidentReporter:
    def test_report_structure(self, feature_names):
        from src.explainability.report_generator import IncidentReporter
        reporter = IncidentReporter(feature_names=feature_names)
        sample = np.random.randn(8)
        report = reporter.generate_report(
            sample=sample, prediction=0.85,
            prediction_set=[1], risk_score=45.0, soc_state="SUSPICIOUS"
        )
        assert "alert_id" in report
        assert "severity" in report
        assert "priority" in report
        assert report["prediction"]["label"] == "MALICIOUS"

    def test_severity_priority_mapping(self, feature_names):
        from src.explainability.report_generator import IncidentReporter
        reporter = IncidentReporter(feature_names=feature_names)
        sample = np.random.randn(8)
        # High-severity case
        report_hi = reporter.generate_report(
            sample=sample, prediction=0.95,
            prediction_set=[0, 1], risk_score=80.0, soc_state="FAILURE"
        )
        assert report_hi["priority"] == "P1-CRITICAL"
        # Low-severity case
        report_lo = reporter.generate_report(
            sample=sample, prediction=0.1,
            prediction_set=[0], risk_score=5.0, soc_state="STABLE"
        )
        assert report_lo["priority"] == "P4-LOW"

    def test_json_export(self, feature_names):
        from src.explainability.report_generator import IncidentReporter
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = IncidentReporter(output_dir=tmpdir, feature_names=feature_names)
            report = reporter.generate_report(
                sample=np.zeros(8), prediction=0.7,
                prediction_set=[1], alert_id="TEST-001"
            )
            path = reporter.export_json(report)
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["alert_id"] == "TEST-001"

    def test_html_export(self, feature_names):
        from src.explainability.report_generator import IncidentReporter
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = IncidentReporter(output_dir=tmpdir, feature_names=feature_names)
            report = reporter.generate_report(
                sample=np.zeros(8), prediction=0.9,
                prediction_set=[0, 1], soc_state="EVASION_LOCKED"
            )
            path = reporter.export_html(report)
            assert os.path.exists(path)
            with open(path) as f:
                html = f.read()
            assert "Incident Report" in html
            assert "EVASION_LOCKED" in html

    def test_csv_batch_export(self, feature_names):
        from src.explainability.report_generator import IncidentReporter
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = IncidentReporter(output_dir=tmpdir, feature_names=feature_names)
            reports = reporter.generate_batch_reports(
                X=np.random.randn(5, 8),
                predictions=np.array([0.1, 0.5, 0.7, 0.9, 0.3]),
                prediction_sets=[[0], [0, 1], [1], [1], [0]],
            )
            path = reporter.export_csv_summary(reports)
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5 rows
