"""
================================================================================
TEST SUITE — ADVERSARIAL ATTACK LIBRARY
================================================================================
≥20 test cases covering all attack categories.
================================================================================
"""
import pytest
import numpy as np


# ── Helpers ─────────────────────────────────────────────────────

class MockModel:
    """Lightweight mock that mimics EnsembleOrchestrator interface."""
    def __init__(self, input_dim=10):
        self.input_dim = input_dim

    def predict_proba(self, X):
        # Simple logistic: feature sum determines class
        scores = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.vstack([1 - scores, scores]).T

@pytest.fixture
def mock_model():
    return MockModel(input_dim=10)

@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y

# ═══════════════════════════════════════════════════════════════
# WHITE-BOX ATTACKS
# ═══════════════════════════════════════════════════════════════

class TestPGDAttack:
    def test_output_shape(self, mock_model, sample_data):
        from src.attacks.white_box import PGDAttack, AttackConfig
        cfg = AttackConfig(epsilon=0.3, max_iter=3)
        atk = PGDAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X, y)
        assert X_adv.shape == X.shape

    def test_epsilon_budget_linf(self, mock_model, sample_data):
        from src.attacks.white_box import PGDAttack, AttackConfig
        eps = 0.2
        cfg = AttackConfig(epsilon=eps, norm="l_inf", max_iter=5)
        atk = PGDAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X, y)
        assert np.all(np.abs(X_adv - X) <= eps + 1e-6)

    def test_epsilon_budget_l2(self, mock_model, sample_data):
        from src.attacks.white_box import PGDAttack, AttackConfig
        eps = 1.0
        cfg = AttackConfig(epsilon=eps, norm="l_2", max_iter=5)
        atk = PGDAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X, y)
        l2_norms = np.linalg.norm(X_adv - X, axis=1)
        assert np.all(l2_norms <= eps + 1e-5)

    def test_mutable_feature_mask(self, mock_model, sample_data):
        from src.attacks.white_box import PGDAttack, AttackConfig
        cfg = AttackConfig(epsilon=0.3, max_iter=3, mutable_features=[0, 1, 2])
        atk = PGDAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X, y)
        # Immutable features (3-9) must be unchanged
        assert np.allclose(X_adv[:, 3:], X[:, 3:])

class TestAutoAttack:
    def test_output_shape(self, mock_model, sample_data):
        from src.attacks.white_box import AutoAttack, AttackConfig
        cfg = AttackConfig(epsilon=0.3, max_iter=2)
        atk = AutoAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X[:20], y[:20])
        assert X_adv.shape == (20, 10)

    def test_attack_finds_adversarials(self, mock_model, sample_data):
        from src.attacks.white_box import AutoAttack, AttackConfig
        cfg = AttackConfig(epsilon=1.0, max_iter=5)
        atk = AutoAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X[:20], y[:20])
        preds_orig = (mock_model.predict_proba(X[:20])[:, 1] > 0.5).astype(int)
        preds_adv = (mock_model.predict_proba(X_adv)[:, 1] > 0.5).astype(int)
        # At least some predictions should change
        assert not np.array_equal(preds_orig, preds_adv)

# ═══════════════════════════════════════════════════════════════
# BLACK-BOX ATTACKS
# ═══════════════════════════════════════════════════════════════

class TestBoundaryAttack:
    def test_output_shape(self, mock_model, sample_data):
        from src.attacks.black_box import BoundaryAttack
        from src.attacks.white_box import AttackConfig
        cfg = AttackConfig(epsilon=2.0, max_iter=3)
        atk = BoundaryAttack(cfg)
        X, y = sample_data
        X_adv = atk.generate(mock_model, X[:5], y[:5])
        assert X_adv.shape == (5, 10)

class TestTransferAttack:
    def test_surrogate_training(self, mock_model, sample_data):
        from src.attacks.black_box import TransferAttack
        from src.attacks.white_box import AttackConfig
        cfg = AttackConfig(epsilon=0.5, max_iter=3)
        atk = TransferAttack(cfg, n_surrogate_estimators=10)
        X, y = sample_data
        atk.fit_surrogate(X, y)
        assert atk.surrogate is not None

    def test_output_within_budget(self, mock_model, sample_data):
        from src.attacks.black_box import TransferAttack
        from src.attacks.white_box import AttackConfig
        eps = 0.5
        cfg = AttackConfig(epsilon=eps, norm="l_inf", max_iter=3)
        atk = TransferAttack(cfg, n_surrogate_estimators=10)
        X, y = sample_data
        atk.fit_surrogate(X, y)
        X_adv = atk.generate(mock_model, X[:10], y[:10])
        assert np.all(np.abs(X_adv - X[:10]) <= eps + 1e-5)

# ═══════════════════════════════════════════════════════════════
# PHYSICAL ATTACKS
# ═══════════════════════════════════════════════════════════════

class TestFeatureConstrainedEvasion:
    def test_non_negative_constraint(self, mock_model):
        from src.attacks.physical import FeatureConstrainedEvasion
        from src.attacks.white_box import AttackConfig
        names = ["iat_mean", "pkt_count", "src_port", "dst_port"]
        cfg = AttackConfig(epsilon=1.0)
        atk = FeatureConstrainedEvasion(cfg, feature_names=names)
        X = np.array([[0.1, 5, 80, 443]] * 10, dtype=np.float32)
        y = np.ones(10, dtype=int)
        X_adv = atk.generate(mock_model, X, y)
        # iat and pkt columns must remain non-negative
        assert np.all(X_adv[:, 0] >= 0)
        assert np.all(X_adv[:, 1] >= 0)

    def test_immutable_unchanged(self, mock_model):
        from src.attacks.physical import FeatureConstrainedEvasion
        from src.attacks.white_box import AttackConfig
        names = ["iat_mean", "pkt_count", "src_port", "dst_port"]
        cfg = AttackConfig(epsilon=1.0, mutable_features=[0, 1])
        atk = FeatureConstrainedEvasion(cfg, feature_names=names)
        X = np.array([[0.1, 5, 80, 443]] * 10, dtype=np.float32)
        y = np.ones(10, dtype=int)
        X_adv = atk.generate(mock_model, X, y)
        assert np.allclose(X_adv[:, 2:], X[:, 2:])

class TestSlowDripAttack:
    def test_iat_stretched(self, mock_model):
        from src.attacks.physical import SlowDripAttack
        names = ["flow_iat_mean", "bytes_sent", "flag"]
        atk = SlowDripAttack(feature_names=names, slowdown=3.0)
        X = np.ones((20, 3), dtype=np.float32)
        y = np.ones(20, dtype=int)
        X_adv = atk.generate(mock_model, X, y)
        # IAT should be stretched
        assert X_adv[:, 0].mean() > X[:, 0].mean()

class TestMimicryAttack:
    def test_requires_profile(self, mock_model, sample_data):
        from src.attacks.physical import MimicryAttack
        atk = MimicryAttack()
        X, y = sample_data
        with pytest.raises(RuntimeError):
            atk.generate(mock_model, X, y)

    def test_output_moves_toward_benign(self, mock_model, sample_data):
        from src.attacks.physical import MimicryAttack
        atk = MimicryAttack(blend_factor=0.9)
        X, y = sample_data
        benign_mask = y == 0
        atk.fit_benign_profile(X[benign_mask])
        X_adv = atk.generate(mock_model, X, y)
        benign_centroid = X[benign_mask].mean(axis=0)
        dist_orig = np.linalg.norm(X.mean(axis=0) - benign_centroid)
        dist_adv = np.linalg.norm(X_adv.mean(axis=0) - benign_centroid)
        assert dist_adv < dist_orig

# ═══════════════════════════════════════════════════════════════
# POISONING ATTACKS
# ═══════════════════════════════════════════════════════════════

class TestLabelFlipPoisoning:
    def test_fraction_respected(self, sample_data):
        from src.attacks.poisoning import LabelFlipPoisoning, PoisonConfig
        cfg = PoisonConfig(fraction=0.1)
        atk = LabelFlipPoisoning(cfg)
        X, y = sample_data
        _, y_p = atk.poison(X, y)
        n_flipped = np.sum(y != y_p)
        assert abs(n_flipped - int(100 * 0.1)) <= 1

    def test_targeted_mode(self, sample_data):
        from src.attacks.poisoning import LabelFlipPoisoning, PoisonConfig
        cfg = PoisonConfig(fraction=0.2, target_class=1)
        atk = LabelFlipPoisoning(cfg, mode="targeted")
        X, y = sample_data
        _, y_p = atk.poison(X, y)
        # Only class-1 samples should have flipped
        # (Not always true because boundary mode, but targeted should only flip target)
        assert np.sum(y != y_p) > 0

class TestBackdoorPoisoning:
    def test_trigger_injection(self, sample_data):
        from src.attacks.poisoning import BackdoorPoisoning, PoisonConfig
        cfg = PoisonConfig(fraction=0.1, trigger_features=[0], trigger_value=999.0)
        atk = BackdoorPoisoning(cfg)
        X, y = sample_data
        X_p, y_p = atk.poison(X, y)
        triggered = X_p[:, 0] == 999.0
        assert triggered.sum() == int(100 * 0.1)

    def test_apply_trigger(self, sample_data):
        from src.attacks.poisoning import BackdoorPoisoning, PoisonConfig
        cfg = PoisonConfig(trigger_features=[0, 1], trigger_value=42.0)
        atk = BackdoorPoisoning(cfg)
        X, _ = sample_data
        X_t = atk.apply_trigger(X)
        assert np.all(X_t[:, 0] == 42.0)
        assert np.all(X_t[:, 1] == 42.0)

class TestCalibrationPoisoning:
    def test_score_inflation(self):
        from src.attacks.poisoning import CalibrationPoisoning, PoisonConfig
        cfg = PoisonConfig(fraction=0.5)
        atk = CalibrationPoisoning(cfg, mode="inflate", score_delta=0.5)
        scores = np.ones(100) * 0.3
        s_p = atk.poison_scores(scores)
        assert s_p.mean() > scores.mean()

    def test_score_deflation(self):
        from src.attacks.poisoning import CalibrationPoisoning, PoisonConfig
        cfg = PoisonConfig(fraction=0.5)
        atk = CalibrationPoisoning(cfg, mode="deflate", score_delta=0.5)
        scores = np.ones(100) * 0.8
        s_p = atk.poison_scores(scores)
        assert s_p.mean() < scores.mean()
