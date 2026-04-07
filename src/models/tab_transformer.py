"""
TabTransformer Implementation
Attention-based tabular model (Huang et al., 2020)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class TabTransformer(Model):
    def __init__(
        self,
        num_numerical_features,
        num_categorical_features=0,
        embed_dim=32,
        num_heads=4,
        num_transformer_blocks=3,
        mlp_hidden_dims=[128, 64],
        dropout_rate=0.1,
        **kwargs,
    ):
        super(TabTransformer, self).__init__(**kwargs)

        self.num_numerical = num_numerical_features
        self.num_categorical = num_categorical_features

        # We don't have explicit categorical columns in the preprocessed data,
        # but following the architecture strictly, we treat features with an embedding projection.
        # Here we just project all continuous numerical features.
        self.feature_embeddings = layers.Dense(embed_dim)
        self.positional_encoding = layers.Embedding(num_numerical_features, embed_dim)

        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(embed_dim, num_heads, embed_dim * 2, dropout_rate)
            )

        self.flatten = layers.Flatten()

        self.mlp = tf.keras.Sequential()
        for dim in mlp_hidden_dims:
            self.mlp.add(layers.Dense(dim, activation="relu"))
            self.mlp.add(layers.Dropout(dropout_rate))

        self.out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, num_features)

        # Project each feature scalar to embed_dim vector -> (batch, num_features, embed_dim)
        x = tf.expand_dims(inputs, -1)
        x = self.feature_embeddings(x)

        # Add index-based positional encoding
        positions = tf.range(start=0, limit=self.num_numerical, delta=1)
        positions = self.positional_encoding(positions)
        x = x + tf.expand_dims(positions, 0)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.flatten(x)
        x = self.mlp(x, training=training)

        return self.out(x)

    def predict_proba(self, X):
        preds = self.predict(X, verbose=0)
        return tf.concat([1 - preds, preds], axis=1).numpy()


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
