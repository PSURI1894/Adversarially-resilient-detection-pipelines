"""
================================================================================
TEST SUITE — CERTIFIED CONFORMAL DEFENSE (RSCP+)
================================================================================
≥20 test cases covering RSCP+, APS/RAPS, poison defense, online CP,
ConformalEngine adapter, and RiskThermostat v2.
================================================================================
"""

import pytest
import numpy as np


# ── MOCK MODEL ─────────────────────────────────────────────────


class MockModel:
    """Returns calibrated-ish probabilities based on feature sum."""

    def predict_proba(self, X):
        scores = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.vstack([1 - scores, scores]).T


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(500, 10).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


@pytest.fixture
def cal_test_split(sample_data):
    X, y = sample_data
    return X[:300], y[:300], X[300:], y[300:]


# ═══════════════════════════════════════════════════════════════
# 1. RSCP+ TESTS
# ═══════════════════════════════════════════════════════════════


class TestRSCP:
    def test_calibration_sets_q_hat(self, mock_model, cal_test_split):
        from src.conformal.rscp import RandomizedSmoothedCP

        X_cal, y_cal, _, _ = cal_test_split
        rscp = RandomizedSmoothedCP(alpha=0.1, sigma=0.05, n_samples=20, ptt=False)
        rscp.calibrate(mock_model, X_cal, y_cal)
        assert rscp.q_hat is not None
        assert 0 < rscp.q_hat < 1

    def test_prediction_sets_nonempty(self, mock_model, cal_test_split):
        from src.conformal.rscp import RandomizedSmoothedCP

        X_cal, y_cal, X_test, _ = cal_test_split
        rscp = RandomizedSmoothedCP(alpha=0.1, sigma=0.05, n_samples=20, ptt=False)
        rscp.calibrate(mock_model, X_cal, y_cal)
        sets = rscp.prediction_sets(X_test, mock_model)
        assert len(sets) == len(X_test)
        assert all(len(s) >= 1 for s in sets)

    def test_coverage_holds_approximately(self, mock_model, cal_test_split):
        from src.conformal.rscp import RandomizedSmoothedCP

        X_cal, y_cal, X_test, y_test = cal_test_split
        alpha = 0.1
        rscp = RandomizedSmoothedCP(alpha=alpha, sigma=0.05, n_samples=30, ptt=False)
        rscp.calibrate(mock_model, X_cal, y_cal)
        sets = rscp.prediction_sets(X_test, mock_model)
        coverage = np.mean([y_test[i] in sets[i] for i in range(len(y_test))])
        # Coverage should be close to 1-alpha (allow some slack for MC noise)
        assert coverage >= 1 - alpha - 0.1

    def test_ptt_reduces_set_size(self, mock_model, cal_test_split):
        from src.conformal.rscp import RandomizedSmoothedCP

        X_cal, y_cal, X_test, _ = cal_test_split
        rscp_no_ptt = RandomizedSmoothedCP(
            alpha=0.1, sigma=0.05, n_samples=20, ptt=False
        )
        rscp_ptt = RandomizedSmoothedCP(alpha=0.1, sigma=0.05, n_samples=20, ptt=True)

        rscp_no_ptt.calibrate(mock_model, X_cal, y_cal)
        rscp_ptt.calibrate(mock_model, X_cal, y_cal)

        sets_no = rscp_no_ptt.prediction_sets(X_test, mock_model)
        sets_ptt = rscp_ptt.prediction_sets(X_test, mock_model)

        avg_no = np.mean([len(s) for s in sets_no])
        avg_ptt = np.mean([len(s) for s in sets_ptt])
        # PTT should produce same or tighter sets
        assert avg_ptt <= avg_no + 0.1  # small tolerance

    def test_certified_radius_positive(self, mock_model, cal_test_split):
        from src.conformal.rscp import RandomizedSmoothedCP

        X_cal, y_cal, X_test, y_test = cal_test_split
        rscp = RandomizedSmoothedCP(alpha=0.1, sigma=0.2, n_samples=20)
        radii = rscp.certified_radius(mock_model, X_test[:10], y_test[:10])
        assert radii.shape == (10,)
        # With reasonable sigma and confident model, most radii should be positive
        assert np.sum(radii > 0) >= 5

    def test_uncalibrated_raises(self, mock_model, cal_test_split):
        from src.conformal.rscp import RandomizedSmoothedCP

        _, _, X_test, _ = cal_test_split
        rscp = RandomizedSmoothedCP()
        with pytest.raises(ValueError, match="calibrate"):
            rscp.prediction_sets(X_test, mock_model)


# ═══════════════════════════════════════════════════════════════
# 2. APS / RAPS TESTS
# ═══════════════════════════════════════════════════════════════


class TestAdaptiveCP:
    def test_aps_coverage(self, mock_model, cal_test_split):
        from src.conformal.multi_class_cp import AdaptiveConformalPredictor

        X_cal, y_cal, X_test, y_test = cal_test_split
        acp = AdaptiveConformalPredictor(alpha=0.1, method="APS")
        acp.calibrate(mock_model, X_cal, y_cal)
        sets = acp.prediction_sets(X_test, mock_model)
        coverage = np.mean([y_test[i] in sets[i] for i in range(len(y_test))])
        assert coverage >= 0.85

    def test_raps_tighter_than_aps(self, mock_model, cal_test_split):
        from src.conformal.multi_class_cp import AdaptiveConformalPredictor

        X_cal, y_cal, X_test, _ = cal_test_split
        aps = AdaptiveConformalPredictor(alpha=0.1, method="APS")
        raps = AdaptiveConformalPredictor(alpha=0.1, method="RAPS", penalty=0.1)
        aps.calibrate(mock_model, X_cal, y_cal)
        raps.calibrate(mock_model, X_cal, y_cal)
        avg_aps = aps.avg_set_size(X_test, mock_model)
        avg_raps = raps.avg_set_size(X_test, mock_model)
        # RAPS should produce same or smaller sets
        assert avg_raps <= avg_aps + 0.15

    def test_class_conditional_coverage(self, mock_model, cal_test_split):
        from src.conformal.multi_class_cp import ClassConditionalCP

        X_cal, y_cal, X_test, y_test = cal_test_split
        cc = ClassConditionalCP(alpha=0.1)
        cc.calibrate(mock_model, X_cal, y_cal)
        assert 0 in cc.q_hats and 1 in cc.q_hats
        sets = cc.prediction_sets(X_test, mock_model)
        assert len(sets) == len(X_test)

    def test_mondrian_per_group(self, mock_model, cal_test_split):
        from src.conformal.multi_class_cp import MondrianCP

        X_cal, y_cal, X_test, _ = cal_test_split
        # Group by sign of first feature
        mc = MondrianCP(alpha=0.1, group_fn=lambda x: "pos" if x[0] > 0 else "neg")
        mc.calibrate(mock_model, X_cal, y_cal)
        assert "pos" in mc.group_q_hats and "neg" in mc.group_q_hats
        sets = mc.prediction_sets(X_test, mock_model)
        assert all(len(s) >= 1 for s in sets)


# ═══════════════════════════════════════════════════════════════
# 3. POISON DEFENSE TESTS
# ═══════════════════════════════════════════════════════════════


class TestPoisonDefense:
    def test_robust_calibration_survives_poisoning(self, mock_model, cal_test_split):
        from src.conformal.poison_defense import RobustCalibration

        X_cal, y_cal, X_test, y_test = cal_test_split
        # Poison 20% of labels
        n_poison = int(len(y_cal) * 0.2)
        y_poisoned = y_cal.copy()
        y_poisoned[:n_poison] = 1 - y_poisoned[:n_poison]

        rc = RobustCalibration(alpha=0.1, n_partitions=7)
        rc.calibrate(mock_model, X_cal, y_poisoned)
        assert rc.q_hat is not None
        sets = rc.prediction_sets(X_test, mock_model)
        coverage = np.mean([y_test[i] in sets[i] for i in range(len(y_test))])
        # Even with 20% poisoning, robust cal should maintain decent coverage
        assert coverage >= 0.75

    def test_partition_q_hats_stored(self, mock_model, cal_test_split):
        from src.conformal.poison_defense import RobustCalibration

        X_cal, y_cal, _, _ = cal_test_split
        rc = RobustCalibration(alpha=0.1, n_partitions=5)
        rc.calibrate(mock_model, X_cal, y_cal)
        assert len(rc.partition_q_hats_) == 5
        assert all(0 < q < 2 for q in rc.partition_q_hats_)

    def test_integrity_monitor_no_drift(self):
        from src.conformal.poison_defense import CalibrationIntegrityMonitor

        baseline = np.random.normal(0.3, 0.1, 500)
        monitor = CalibrationIntegrityMonitor(baseline)
        # Same distribution → no drift
        recent = np.random.normal(0.3, 0.1, 100)
        result = monitor.detect_drift(recent)
        assert isinstance(result, dict)
        assert result["drift_detected"] is False

    def test_integrity_monitor_detects_drift(self):
        from src.conformal.poison_defense import CalibrationIntegrityMonitor

        baseline = np.random.normal(0.3, 0.1, 500)
        monitor = CalibrationIntegrityMonitor(baseline)
        # Shifted distribution → drift
        recent = np.random.normal(0.8, 0.1, 100)
        result = monitor.detect_drift(recent)
        assert result["drift_detected"] is True

    def test_integrity_full_check(self):
        from src.conformal.poison_defense import CalibrationIntegrityMonitor

        baseline = np.random.normal(0.3, 0.1, 500)
        monitor = CalibrationIntegrityMonitor(baseline)
        recent = np.random.normal(0.3, 0.1, 100)
        report = monitor.full_integrity_check(recent)
        assert "drift" in report
        assert "anomalies" in report
        assert "moment_shift" in report


# ═══════════════════════════════════════════════════════════════
# 4. ONLINE CP TESTS
# ═══════════════════════════════════════════════════════════════


class TestOnlineCP:
    def test_streaming_update(self, mock_model, sample_data):
        from src.conformal.online_cp import OnlineConformalPredictor

        X, y = sample_data
        ocp = OnlineConformalPredictor(alpha=0.1, gamma=0.01)
        # Process first 50 points
        for i in range(50):
            pset = ocp.update(mock_model, X[i], y[i])
            assert len(pset) >= 1
        assert ocp.q_hat is not None

    def test_rolling_coverage_converges(self, mock_model, sample_data):
        from src.conformal.online_cp import OnlineConformalPredictor

        X, y = sample_data
        ocp = OnlineConformalPredictor(alpha=0.1, gamma=0.005)
        ocp.update_batch(mock_model, X, y)
        cov = ocp.rolling_coverage(100)
        # After 500 points, rolling coverage should be in a reasonable range
        assert 0.5 < cov < 1.0

    def test_exchangeability_check(self, mock_model, sample_data):
        from src.conformal.online_cp import OnlineConformalPredictor

        X, y = sample_data
        ocp = OnlineConformalPredictor(alpha=0.1)
        ocp.update_batch(mock_model, X, y)
        result = ocp.check_exchangeability(lookback=200)
        assert "exchangeable" in result
        assert "runs_stat" in result

    def test_diagnostics(self, mock_model, sample_data):
        from src.conformal.online_cp import OnlineConformalPredictor

        X, y = sample_data
        ocp = OnlineConformalPredictor(alpha=0.1)
        ocp.update_batch(mock_model, X[:50], y[:50])
        diag = ocp.get_diagnostics()
        assert diag["n_processed"] == 50
        assert "current_alpha" in diag
        assert "buffer_size" in diag


# ═══════════════════════════════════════════════════════════════
# 5. CONFORMAL ENGINE ADAPTER TESTS
# ═══════════════════════════════════════════════════════════════


class TestConformalEngineAdapter:
    def test_split_backend_default(self, mock_model, sample_data):
        from src.risk_management_engine import ConformalEngine, ConformalBackend

        X, y = sample_data
        ce = ConformalEngine(alpha=0.05, backend=ConformalBackend.SPLIT)
        ce.calibrate(mock_model, X, y)
        assert ce.q_hat is not None

    def test_rscp_backend(self, mock_model, cal_test_split):
        from src.risk_management_engine import ConformalEngine, ConformalBackend

        X_cal, y_cal, X_test, _ = cal_test_split
        ce = ConformalEngine(
            alpha=0.1, backend=ConformalBackend.RSCP, sigma=0.05, n_samples=10
        )
        ce.calibrate(mock_model, X_cal, y_cal)
        sets = ce.prediction_sets(X_test, mock_model)
        assert len(sets) == len(X_test)

    def test_adaptive_backend(self, mock_model, cal_test_split):
        from src.risk_management_engine import ConformalEngine, ConformalBackend

        X_cal, y_cal, X_test, _ = cal_test_split
        ce = ConformalEngine(
            alpha=0.1, backend=ConformalBackend.ADAPTIVE, method="RAPS"
        )
        ce.calibrate(mock_model, X_cal, y_cal)
        sets = ce.prediction_sets(X_test, mock_model)
        assert all(len(s) >= 1 for s in sets)


# ═══════════════════════════════════════════════════════════════
# 6. RISK THERMOSTAT v2 TESTS
# ═══════════════════════════════════════════════════════════════


class TestRiskThermostatV2:
    def test_stable_state_on_clean(self):
        from src.risk_management_engine import RiskThermostat, SOCState

        rt = RiskThermostat(hysteresis_steps=1)
        # All singletons → clean
        sets = [[0]] * 100
        state = rt.evaluate(sets)
        assert state == SOCState.STABLE

    def test_severity_score_computed(self):
        from src.risk_management_engine import RiskThermostat

        rt = RiskThermostat(hysteresis_steps=1)
        sets = [[0, 1]] * 100  # all ambiguous
        rt.evaluate(sets)
        assert rt.severity > 0

    def test_hysteresis_prevents_flapping(self):
        from src.risk_management_engine import RiskThermostat, SOCState

        rt = RiskThermostat(hysteresis_steps=3, cooldown_seconds=0)
        # Single suspicious batch shouldn't flip state
        suspicious_sets = [[0, 1]] * 100
        state = rt.evaluate(suspicious_sets)
        # With hysteresis=3, one eval isn't enough
        assert state == SOCState.STABLE
        # After 3 consecutive suspicious batches → should transition
        rt.evaluate(suspicious_sets)
        state = rt.evaluate(suspicious_sets)
        assert state != SOCState.STABLE

    def test_multi_signal_severity(self):
        from src.risk_management_engine import RiskThermostat

        rt = RiskThermostat(hysteresis_steps=1)
        sets = [[0]] * 100
        rt.evaluate(sets, calibration_drift=0.5, disagreement=0.5)
        # Even with clean sets, high drift/disagreement raises severity
        assert rt.severity > 10

    def test_playbook_returns_action(self):
        from src.risk_management_engine import RiskThermostat

        rt = RiskThermostat(hysteresis_steps=1)
        sets = [[0]] * 100
        rt.evaluate(sets)
        pb = rt.playbook()
        assert "action" in pb

    def test_diagnostics(self):
        from src.risk_management_engine import RiskThermostat

        rt = RiskThermostat(hysteresis_steps=1)
        rt.evaluate([[0]] * 50)
        diag = rt.get_diagnostics()
        assert "state" in diag
        assert "severity" in diag
        assert "alert_debt" in diag
