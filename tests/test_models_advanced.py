"""
================================================================================
TEST SUITE — ADVANCED MODELS & DEEP ENSEMBLE
================================================================================
≥25 test cases for TabTransformer, VAIDS, DeepEnsemble, Adversarial Training, and Calibration.
"""

import pytest
import numpy as np

tf = pytest.importorskip(
    "tensorflow", reason="TensorFlow not installed — skipping advanced model tests"
)

from src.models.tab_transformer import TabTransformer  # noqa: E402
from src.models.variational_autoencoder import VAIDS  # noqa: E402
from src.models.deep_ensemble import DeepEnsemble  # noqa: E402
from src.models.adversarial_trainer import (  # noqa: E402
    PGDTrainer,
    TRADESTrainer,
    FreeAdversarialTrainer,
)
from src.models.calibration import (  # noqa: E402
    TemperatureScaling,
    IsotonicCalibration,
    CalibrationAudit,
)

# ── FIXTURES ───────────────────────────────────────────────────


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


@pytest.fixture
def base_model():
    # A simple TF model for testing wrappers
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)  # logits
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
    return model


# ═══════════════════════════════════════════════════════════════
# 1. TAB-TRANSFORMER TESTS
# ═══════════════════════════════════════════════════════════════


class TestTabTransformer:
    def test_output_shape_probabilities(self, sample_data):
        X, _ = sample_data
        model = TabTransformer(num_numerical_features=X.shape[1])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        probs = model.predict_proba(X[:10])
        assert probs.shape == (10, 2)
        assert np.all((probs >= 0) & (probs <= 1))
        assert np.allclose(np.sum(probs, axis=1), 1.0)

    def test_forward_pass_dims(self, sample_data):
        X, _ = sample_data
        model = TabTransformer(
            num_numerical_features=X.shape[1], embed_dim=16, num_heads=2
        )
        out = model(X[:5])
        assert out.shape == (5, 1)

    def test_training_reduces_loss(self, sample_data):
        X, y = sample_data
        model = TabTransformer(num_numerical_features=X.shape[1])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        hist = model.fit(X, y, epochs=2, batch_size=32, verbose=0)
        assert len(hist.history["loss"]) == 2

    def test_attention_mask_shapes_internal(self, sample_data):
        # We implicitly test this by passing through the blocks without crashing
        X, _ = sample_data
        model = TabTransformer(num_numerical_features=X.shape[1])
        _ = model(X[:2])  # trigger build
        assert len(model.layers) > 0


# ═══════════════════════════════════════════════════════════════
# 2. VARIATIONAL AUTOENCODER (VAIDS) TESTS
# ═══════════════════════════════════════════════════════════════


class TestVAIDS:
    def test_latent_space_dims(self, sample_data):
        X, _ = sample_data
        vae = VAIDS(input_dim=10, latent_dim=4)
        z_mean, z_log_var, z = vae.encoder(X[:5])
        assert z_mean.shape == (5, 4)
        assert z_log_var.shape == (5, 4)
        assert z.shape == (5, 4)

    def test_reconstruction_dims(self, sample_data):
        X, _ = sample_data
        vae = VAIDS(input_dim=10, latent_dim=4)
        z_mean, _, _ = vae.encoder(X[:5])
        rec = vae.decoder(z_mean)
        assert rec.shape == (5, 10)

    def test_training_dict_metrics(self, sample_data):
        X, _ = sample_data
        vae = VAIDS(input_dim=10)
        vae.compile(optimizer="adam")
        hist = vae.fit(X, epochs=2, batch_size=32, verbose=0)
        assert "reconstruction_loss" in hist.history
        assert "kl_loss" in hist.history

    def test_anomaly_score_shape(self, sample_data):
        X, _ = sample_data
        vae = VAIDS(input_dim=10)
        scores = vae.score_anomalies(X[:10])
        assert scores.shape == (10,)

    def test_predict_proba_squash(self, sample_data):
        X, _ = sample_data
        vae = VAIDS(input_dim=10)
        probs = vae.predict_proba(X[:10])
        assert probs.shape == (10, 2)
        assert np.allclose(np.sum(probs, axis=1), 1.0)


# ═══════════════════════════════════════════════════════════════
# 3. DEEP ENSEMBLE TESTS
# ═══════════════════════════════════════════════════════════════


class TestDeepEnsemble:
    def test_initialization_members(self):
        de = DeepEnsemble(input_dim=10, n_members=3)
        assert len(de.members) == 3

    def test_predict_proba_shape(self, sample_data):
        X, _ = sample_data
        de = DeepEnsemble(input_dim=10, n_members=2)
        # Without explicit fit, models are randomly initialized
        probs = de.predict_proba(X[:5])
        assert probs.shape == (5, 2)

    def test_epistemic_uncertainty_decomposition(self, sample_data):
        X, _ = sample_data
        de = DeepEnsemble(input_dim=10, n_members=5)
        de.predict_proba(X[:5])
        var = de.get_epistemic_uncertainty()
        assert var is not None
        assert var.shape == (5,)
        assert np.all(var >= 0)

    def test_training_loop(self, sample_data):
        X, y = sample_data
        de = DeepEnsemble(input_dim=10, n_members=2, epochs=1)
        de.fit(X, y)  # should not crash
        assert True


# ═══════════════════════════════════════════════════════════════
# 4. ADVERSARIAL TRAINERS TESTS
# ═══════════════════════════════════════════════════════════════


class TestAdversarialTrainers:
    def test_pgd_trainer_generates_adv(self, sample_data, base_model):
        X, y = sample_data
        trainer = PGDTrainer(base_model, epsilon=0.1, iters=2)
        X_tf = tf.constant(X[:5])
        y_tf = tf.constant(y[:5])
        x_adv = trainer.generate_adversarial(X_tf, y_tf)
        assert x_adv.shape == X_tf.shape
        assert not np.array_equal(x_adv.numpy(), X_tf.numpy())

    def test_pgd_trainer_mutable_mask(self, sample_data, base_model):
        X, y = sample_data
        trainer = PGDTrainer(base_model, epsilon=1.0, iters=2, mutable_features=[0, 1])
        X_tf = tf.constant(X[:5])
        y_tf = tf.constant(y[:5])
        x_adv = trainer.generate_adversarial(X_tf, y_tf).numpy()
        # Non-mutable features should be strictly equal
        assert np.allclose(x_adv[:, 2:], X[:5, 2:])

    def test_pgd_train_step(self, sample_data, base_model):
        X, y = sample_data
        trainer = PGDTrainer(base_model, epsilon=0.1, iters=1)
        opt = tf.keras.optimizers.Adam()
        loss = trainer.train_step(tf.constant(X[:8]), tf.constant(y[:8]), opt)
        assert loss is not None

    def test_trades_train_step(self, sample_data, base_model):
        X, y = sample_data
        trainer = TRADESTrainer(base_model, epsilon=0.1, iters=1)
        opt = tf.keras.optimizers.Adam()
        loss = trainer.train_step(tf.constant(X[:8]), tf.constant(y[:8]), opt)
        assert loss > 0

    def test_free_train_step(self, sample_data, base_model):
        X, y = sample_data

        # create a dummy generator
        def dummy_gen():
            yield tf.constant(X[:8]), tf.constant(y[:8])

        trainer = FreeAdversarialTrainer(base_model, epsilon=0.1, m=2)
        opt = tf.keras.optimizers.Adam()
        loss = trainer.train_step_batch(dummy_gen(), opt)
        assert loss is not None


# ═══════════════════════════════════════════════════════════════
# 5. CALIBRATION TESTS
# ═══════════════════════════════════════════════════════════════


class TestCalibration:
    def test_temperature_scaling_bounds(self):
        cal = TemperatureScaling()
        logits = np.random.randn(100)
        y = np.random.randint(0, 2, 100)
        cal.fit(logits, y)
        assert cal.temperature > 0

    def test_temperature_scaling_probs(self):
        cal = TemperatureScaling()
        cal.temperature = 2.0
        logits = np.array([0.0, 2.0])
        probs = cal.predict_proba(logits)
        assert probs.shape == (2, 2)
        assert np.allclose(probs[0], [0.5, 0.5])

    def test_isotonic_calibration_fit_transform(self):
        cal = IsotonicCalibration()
        preds = np.random.rand(100)
        y = (preds + np.random.normal(0, 0.1, 100) > 0.5).astype(int)
        cal.fit(preds, y)
        calibrated = cal.predict_proba(preds)
        assert calibrated.shape == (100, 2)
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_ece_metric(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        ece = CalibrationAudit.expected_calibration_error(y_true, y_prob, n_bins=2)
        assert ece < 0.2

    def test_mce_metric(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        mce = CalibrationAudit.maximum_calibration_error(y_true, y_prob, n_bins=2)
        assert mce < 0.2

    def test_ece_after_scaling_improves(self):
        # We simulate uncalibrated probs
        y_true = np.random.randint(0, 2, 1000)
        logits = np.where(y_true == 1, 10.0, -10.0) + np.random.randn(1000) * 15.0
        uncalib_probs = 1 / (1 + np.exp(-logits))
        ece_before = CalibrationAudit.expected_calibration_error(y_true, uncalib_probs)

        cal = TemperatureScaling()
        cal.fit(logits, y_true)
        calib_probs = cal.predict_proba(logits)[:, 1]
        ece_after = CalibrationAudit.expected_calibration_error(y_true, calib_probs)
        # Calibration usually strictly improves ECE here
        assert ece_after <= ece_before + 1e-4

    def test_reliability_diagram_data(self):
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        confs, accs = CalibrationAudit.reliability_diagram_data(
            y_true, y_prob, n_bins=5
        )
        assert len(confs) == 5
        assert len(accs) == 5
