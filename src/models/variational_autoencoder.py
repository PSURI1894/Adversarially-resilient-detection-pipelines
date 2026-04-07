"""
Variational Autoencoder for Intrusion Detection (VAIDS)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAIDS(Model):
    def __init__(self, input_dim, latent_dim=8, intermediate_dims=[64, 32], **kwargs):
        super(VAIDS, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_inputs = tf.keras.Input(shape=(input_dim,))
        x = encoder_inputs
        for dim in intermediate_dims:
            x = layers.Dense(dim, activation="relu")(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = latent_inputs
        for dim in reversed(intermediate_dims):
            x = layers.Dense(dim, activation="relu")(x)
        decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        # Tracker
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Semi-supervised setup: pass data as X or (X, y_ignore)
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_mean(tf.square(data - reconstruction), axis=-1), axis=-1
                )
                * self.input_dim
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def score_anomalies(self, X):
        z_mean, _, _ = self.encoder.predict(X, verbose=0)
        reconstruction = self.decoder.predict(z_mean, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstruction), axis=1)
        return reconstruction_errors

    def predict_proba(self, X):
        """Converts anomaly scores to pseudo-probabilities via squashing for ensemble compatibility."""
        scores = self.score_anomalies(X)
        # Scaled sigmoid squash
        prob_malicious = 1 / (
            1 + np.exp(-(scores - scores.mean()) / (scores.std() + 1e-6))
        )
        return np.vstack([1 - prob_malicious, prob_malicious]).T
