import tensorflow as tf
import numpy as np

class SimpleTransformer(tf.keras.Model):
    """
    A simplified version of the Transformer architecture for language modeling.
    
    How it works:
    1. Takes sequences of word numbers as input
    2. Converts numbers to rich word representations (embeddings)
    3. Adds position information to maintain word order
    4. Processes through multiple transformer blocks that:
       - Use attention to focus on relevant words
       - Learn relationships between words
       - Transform word representations
    5. Outputs predictions for the next word
    
    Components:
    - Embedding layer: Converts word numbers to vectors
    - Positional encoding: Adds position information
    - Transformer blocks: Process and transform word representations
    - Final layer: Converts to word probabilities
    
    Args:
        vocab_size (int): Size of vocabulary
        d_model (int): Size of word vectors
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer blocks
    """
    def __init__(self, vocab_size, d_model=64, num_heads=2, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(vocab_size, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ]
        
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
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
    """
    Core processing block of the Transformer model.
    
    How it works:
    1. Multi-Head Attention:
       - Allows model to focus on different parts of input
       - Like having multiple "perspectives" on the text
       - Example: One head might focus on subject-verb agreement,
         another on topic-related words
    
    2. Feed-Forward Network:
       - Processes attended information
       - Transforms word representations
       - Learns complex patterns
    
    3. Layer Normalization:
       - Stabilizes learning
       - Helps training converge better
    
    Args:
        d_model (int): Size of word vectors
        num_heads (int): Number of attention heads
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, x):
        # Self-attention
        attention_output = self.attention(x, x)
        x1 = self.layernorm1(x + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x1)
        return self.layernorm2(x1 + ffn_output) 