import tensorflow as tf
import numpy as np

class SimpleTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=64, num_heads=2, num_layers=2):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(vocab_size, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ]
        
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def positional_encoding(self, position, d_model):
        angles = np.arange(position)[:, np.newaxis] / np.power(
            10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model
        )
        
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        
        # Embedding + Positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
            
        # Final layer
        return self.final_layer(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x):
        # Self-attention
        attention_output = self.attention(x, x)
        x1 = self.layernorm1(x + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x1)
        return self.layernorm2(x1 + ffn_output) 