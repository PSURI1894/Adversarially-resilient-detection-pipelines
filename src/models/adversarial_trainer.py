"""
Adversarial Training Loops
Includes PGD, TRADES, and Free-m Adversarial Training
"""

import tensorflow as tf
import numpy as np

class BaseAdversarialTrainer:
    def __init__(self, model, epsilon=0.1, mutable_features=None):
        self.model = model
        self.epsilon = epsilon
        self.mutable_features = mutable_features

    def apply_feature_mask(self, x_adv, x_orig):
        if self.mutable_features is None:
            return x_adv
            
        mask = np.zeros(x_orig.shape[1], dtype=bool)
        mask[self.mutable_features] = True
        
        # We need to do this in TF for gradient preservation or outside if eager
        # Assuming eager mode with tf.where
        mask_tf = tf.constant(mask, dtype=tf.bool)
        return tf.where(mask_tf, x_adv, x_orig)


class PGDTrainer(BaseAdversarialTrainer):
    """PGD Adversarial Training."""
    def __init__(self, model, epsilon=0.1, alpha=0.02, iters=5, mutable_features=None):
        super().__init__(model, epsilon, mutable_features)
        self.alpha = alpha
        self.iters = iters

    def generate_adversarial(self, x, y):
        x_adv = tf.identity(x)
        # Random init inside epsilon ball
        noise = tf.random.uniform(tf.shape(x), minval=-self.epsilon, maxval=self.epsilon)
        x_adv = x_adv + noise
        x_adv = self.apply_feature_mask(x_adv, x)
        
        # PGD loop
        for _ in range(self.iters):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = self.model(x_adv, training=False)
                loss = tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(tf.cast(y, tf.float32), -1), logits, from_logits=True
                )
            
            grad = tape.gradient(loss, x_adv)
            x_adv = x_adv + self.alpha * tf.sign(grad)
            
            # Project back to epsilon ball
            eta = tf.clip_by_value(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + eta
            x_adv = self.apply_feature_mask(x_adv, x)
            
        return x_adv

    def train_step(self, x, y, optimizer):
        x_adv = self.generate_adversarial(x, y)
        
        with tf.GradientTape() as tape:
            # We can optionally compute loss on 50% clean, 50% adversarial
            logits_adv = self.model(x_adv, training=True)
            loss_adv = tf.keras.losses.binary_crossentropy(
                tf.expand_dims(tf.cast(y, tf.float32), -1), logits_adv, from_logits=True
            )
            loss = tf.reduce_mean(loss_adv)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class TRADESTrainer(BaseAdversarialTrainer):
    """TRADES: KL Divergence regularization."""
    def __init__(self, model, epsilon=0.1, alpha=0.02, iters=5, beta=6.0, mutable_features=None):
        super().__init__(model, epsilon, mutable_features)
        self.alpha = alpha
        self.iters = iters
        self.beta = beta

    def train_step(self, x, y, optimizer):
        # Obtain clean logits
        logits_clean = self.model(x, training=False)
        probs_clean = tf.nn.sigmoid(logits_clean)

        # Generate adversarial example that maximizes KL divergence
        x_adv = tf.identity(x)
        noise = tf.random.uniform(tf.shape(x), minval=-self.epsilon, maxval=self.epsilon)
        x_adv = x_adv + noise
        x_adv = self.apply_feature_mask(x_adv, x)

        for _ in range(self.iters):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits_adv = self.model(x_adv, training=False)
                probs_adv = tf.nn.sigmoid(logits_adv)
                
                # KL Divergence for binary classification (Bernoulli)
                kl = probs_clean * tf.math.log(probs_clean / (probs_adv + 1e-8) + 1e-8) + \
                     (1 - probs_clean) * tf.math.log((1 - probs_clean) / (1 - probs_adv + 1e-8) + 1e-8)
                loss_kl = tf.reduce_sum(kl)
            
            grad = tape.gradient(loss_kl, x_adv)
            x_adv = x_adv + self.alpha * tf.sign(grad)
            eta = tf.clip_by_value(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + eta
            x_adv = self.apply_feature_mask(x_adv, x)

        # Optimization step over clean and adv combined loss
        with tf.GradientTape() as tape:
            logits_clean_train = self.model(x, training=True)
            logits_adv_train = self.model(x_adv, training=True)
            
            ce_loss = tf.keras.losses.binary_crossentropy(
                tf.expand_dims(tf.cast(y, tf.float32), -1), logits_clean_train, from_logits=True
            )
            
            probs_clean_train = tf.nn.sigmoid(logits_clean_train)
            probs_adv_train = tf.nn.sigmoid(logits_adv_train)
            kl_train = probs_clean_train * tf.math.log(probs_clean_train / (probs_adv_train + 1e-8) + 1e-8) + \
                       (1 - probs_clean_train) * tf.math.log((1 - probs_clean_train) / (1 - probs_adv_train + 1e-8) + 1e-8)
            
            total_loss = tf.reduce_mean(ce_loss) + self.beta * tf.reduce_mean(kl_train)
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss


class FreeAdversarialTrainer(BaseAdversarialTrainer):
    """Free-m Adversarial Training."""
    def __init__(self, model, epsilon=0.1, m=4, mutable_features=None):
        super().__init__(model, epsilon, mutable_features)
        self.m = m

    def train_step_batch(self, dataset_iterator, optimizer):
        # We need a batch to replay
        try:
            x, y = next(dataset_iterator)
        except StopIteration:
            return None
            
        eta = tf.zeros_like(x)
        total_loss = 0.0

        for _ in range(self.m):
            x_adv = x + eta
            x_adv = self.apply_feature_mask(x_adv, x)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_adv)
                logits = self.model(x_adv, training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(tf.cast(y, tf.float32), -1), logits, from_logits=True
                ))
            
            grads_model = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads_model, self.model.trainable_variables))
            
            grad_x = tape.gradient(loss, x_adv)
            eta = eta + self.epsilon * tf.sign(grad_x)
            eta = tf.clip_by_value(eta, -self.epsilon, self.epsilon)
            
            del tape
            total_loss += loss.numpy()
            
        return total_loss / self.m
