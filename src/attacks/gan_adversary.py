"""
================================================================================
ADVERSARIAL GAN — Wasserstein GAN with Gradient Penalty
================================================================================
Generator produces adversarial flow records that fool the detector.
Discriminator doubles as a secondary anomaly detector.
================================================================================
"""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
except ImportError:  # pragma: no cover
    tf = None  # type: ignore[assignment]
    layers = None  # type: ignore[assignment]
    Model = object  # type: ignore[assignment,misc]

from typing import Optional, List


class AdversarialGAN:
    """WGAN-GP for generating adversarial network traffic."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        gp_weight: float = 10.0,
        mutable_features: Optional[List[int]] = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.mutable_features = mutable_features or list(range(input_dim))

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        self.g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)

    def _build_generator(self) -> Model:
        z = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation="relu")(z)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.input_dim, activation="tanh")(x)
        return Model(z, x, name="generator")

    def _build_discriminator(self) -> Model:
        inp = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(256, activation="leaky_relu")(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="leaky_relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(1)(x)  # no activation (Wasserstein)
        return Model(inp, x, name="discriminator")

    def _gradient_penalty(self, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
        alpha = tf.random.uniform([tf.shape(real)[0], 1], 0.0, 1.0)
        interp = real + alpha * (fake - real)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = self.discriminator(interp, training=True)
        grads = tape.gradient(pred, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        return tf.reduce_mean(tf.square(norm - 1.0))

    @tf.function
    def _train_step(self, real_data: tf.Tensor):
        batch = tf.shape(real_data)[0]
        # ── Discriminator ─────────
        z = tf.random.normal([batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator(z, training=True)
            d_real = self.discriminator(real_data, training=True)
            d_fake = self.discriminator(fake, training=True)
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            gp = self._gradient_penalty(real_data, fake)
            d_loss += self.gp_weight * gp
        d_grad = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # ── Generator ─────────────
        z = tf.random.normal([batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator(z, training=True)
            g_out = self.discriminator(fake, training=True)
            g_loss = -tf.reduce_mean(g_out)
        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return d_loss, g_loss

    def fit(self, X_real: np.ndarray, epochs: int = 50, batch_size: int = 256):
        dataset = (
            tf.data.Dataset.from_tensor_slices(X_real.astype(np.float32))
            .shuffle(10_000)
            .batch(batch_size)
        )
        self.history = {"d_loss": [], "g_loss": []}
        for ep in range(epochs):
            for batch in dataset:
                d_l, g_l = self._train_step(batch)
            self.history["d_loss"].append(float(d_l))
            self.history["g_loss"].append(float(g_l))

    def generate(
        self,
        model=None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        n_samples: int = 1000,
    ) -> np.ndarray:
        z = np.random.randn(n_samples, self.latent_dim).astype(np.float32)
        return self.generator.predict(z, verbose=0)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Use discriminator as anomaly detector (high = real, low = anomalous)."""
        return self.discriminator.predict(X.astype(np.float32), verbose=0).flatten()
