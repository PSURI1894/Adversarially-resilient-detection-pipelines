"""
DeepEnsemble Implementation
Based on Lakshminarayanan et al. (2017) - Simple and Scalable Predictive Uncertainty Estimation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def build_base_network(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    # Output predicting mean and variance (heteroscedastic)
    # Actually for binary classification, we just output the logit.
    # The ensemble handles the variance across members.
    logits = layers.Dense(1, name='logits')(x)
    return Model(inputs, logits)

class DeepEnsemble:
    def __init__(self, input_dim, n_members=5, epochs=10, batch_size=256):
        self.input_dim = input_dim
        self.n_members = n_members
        self.epochs = epochs
        self.batch_size = batch_size
        self.members = []
        
        for _ in range(n_members):
            model = build_base_network(input_dim)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            self.members.append(model)
            
    def fit(self, X, y):
        # Train each member on a shuffled version of the dataset (or bootstrap sample)
        for i, model in enumerate(self.members):
            print(f"Training ensemble member {i+1}/{self.n_members}")
            # Bootstrapping
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            model.fit(X_boot, y_boot, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            
    def predict_proba(self, X):
        predictions = []
        for model in self.members:
            logits = model.predict(X, verbose=0)
            probs = tf.nn.sigmoid(logits).numpy()
            predictions.append(probs)
            
        predictions = np.stack(predictions, axis=1) # (batch, members, 1)
        mean_pred = np.mean(predictions, axis=1) # (batch, 1)
        
        # We can also compute epistemic uncertainty as variance
        self.last_epistemic_uncertainty = np.var(predictions, axis=1)
        
        return np.hstack([1 - mean_pred, mean_pred])
        
    def get_epistemic_uncertainty(self):
        """Returns the variance across member predictions from the last predict_proba call."""
        if hasattr(self, 'last_epistemic_uncertainty'):
            return self.last_epistemic_uncertainty
        return None
