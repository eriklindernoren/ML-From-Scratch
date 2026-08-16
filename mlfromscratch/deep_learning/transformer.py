"""
Transformer implementation from scratch using NumPy.

Based on the paper: "Attention Is All You Need" (Vaswani et al., 2017)

This implementation includes:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network
- Positional Encoding
- Transformer Encoder Layer
- Full Transformer Encoder for sequence classification

Reference: https://arxiv.org/abs/1706.03762
"""

from __future__ import print_function, division
import math
import numpy as np
import copy
from mlfromscratch.deep_learning.layers import Layer
from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils import train_test_split, to_categorical


class MultiHeadAttention(Layer):
    """Multi-Head Attention mechanism.

    Parameters:
    -----------
    d_model: int
        The dimension of the model (must be divisible by n_heads)
    n_heads: int
        Number of attention heads
    input_shape: tuple
        Expected input shape (seq_length, d_model)
    """
    def __init__(self, d_model, n_heads, input_shape=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.input_shape = input_shape
        self.trainable = True

        # Weight matrices for Q, K, V projections
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

    def initialize(self, optimizer):
        # Initialize weights with Xavier initialization
        limit_q = 1 / math.sqrt(self.d_model)
        self.W_q = np.random.uniform(-limit_q, limit_q, (self.d_model, self.d_model))
        self.W_k = np.random.uniform(-limit_q, limit_q, (self.d_model, self.d_model))
        self.W_v = np.random.uniform(-limit_q, limit_q, (self.d_model, self.d_model))
        self.W_o = np.random.uniform(-limit_q, limit_q, (self.d_model, self.d_model))

        # Optimizers for each weight
        self.W_q_opt = copy.copy(optimizer)
        self.W_k_opt = copy.copy(optimizer)
        self.W_v_opt = copy.copy(optimizer)
        self.W_o_opt = copy.copy(optimizer)

    def parameters(self):
        return 4 * self.d_model * self.d_model

    def _split_heads(self, X, batch_size):
        """Split the last dimension into (n_heads, d_k)."""
        # X shape: (batch_size, seq_length, d_model)
        X = X.reshape(batch_size, -1, self.n_heads, self.d_k)
        # Transpose to: (batch_size, n_heads, seq_length, d_k)
        return X.transpose(0, 2, 1, 3)

    def _scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention."""
        # Q, K, V shape: (batch_size, n_heads, seq_length, d_k)
        d_k = Q.shape[-1]

        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)

        # Apply mask if provided (for masked attention)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Apply softmax
        attention_weights = self._softmax(scores)

        # Compute weighted values
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def _softmax(self, X):
        """Numerically stable softmax."""
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def forward_pass(self, X, training=True):
        """Forward pass for multi-head attention.

        Parameters:
        -----------
        X: numpy array of shape (batch_size, seq_length, d_model)
        """
        self.layer_input = X
        batch_size = X.shape[0]

        # Linear projections
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)

        # Split into multiple heads
        Q = self._split_heads(Q, batch_size)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)

        # Apply scaled dot-product attention
        self.attention_output, self.attention_weights = self._scaled_dot_product_attention(Q, K, V)

        # Concatenate heads
        # Shape: (batch_size, n_heads, seq_length, d_k) -> (batch_size, seq_length, d_model)
        self.attention_output = self.attention_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # Final linear projection
        output = np.matmul(self.attention_output, self.W_o)

        return output

    def backward_pass(self, accum_grad):
        """Backward pass for multi-head attention."""
        batch_size = accum_grad.shape[0]

        # Gradient for W_o
        grad_W_o = self.attention_output.T.dot(accum_grad)

        # Gradient flowing back through W_o
        grad_attention_output = accum_grad.dot(self.W_o.T)

        # Reshape and transpose for multi-head
        grad_attention_output = grad_attention_output.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Simplified backward pass (approximation)
        # In practice, you'd compute gradients through softmax and matmul
        grad_V = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), grad_attention_output)
        grad_K = np.matmul(grad_attention_output.transpose(0, 1, 3, 2), self.attention_output)
        grad_Q = np.matmul(grad_attention_output, self.attention_output)

        # Update weights
        if self.trainable:
            self.W_o = self.W_o_opt.update(self.W_o, grad_W_o)

        # Return gradient to input (simplified)
        return accum_grad

    def output_shape(self):
        return self.input_shape


class PositionwiseFeedForward(Layer):
    """Position-wise Feed-Forward Network.

    Parameters:
    -----------
    d_model: int
        The dimension of the model
    d_ff: int
        The dimension of the hidden layer (typically 4 * d_model)
    input_shape: tuple
        Expected input shape
    """
    def __init__(self, d_model, d_ff, input_shape=None):
        self.d_model = d_model
        self.d_ff = d_ff
        self.input_shape = input_shape
        self.trainable = True

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def initialize(self, optimizer):
        # Initialize weights
        limit1 = 1 / math.sqrt(self.d_model)
        limit2 = 1 / math.sqrt(self.d_ff)

        self.W1 = np.random.uniform(-limit1, limit1, (self.d_model, self.d_ff))
        self.b1 = np.zeros((1, self.d_ff))
        self.W2 = np.random.uniform(-limit2, limit2, (self.d_ff, self.d_model))
        self.b2 = np.zeros((1, self.d_model))

        # Optimizers
        self.W1_opt = copy.copy(optimizer)
        self.b1_opt = copy.copy(optimizer)
        self.W2_opt = copy.copy(optimizer)
        self.b2_opt = copy.copy(optimizer)

    def parameters(self):
        return (self.d_model * self.d_ff + self.d_ff +
                self.d_ff * self.d_model + self.d_model)

    def _relu(self, X):
        return np.maximum(0, X)

    def forward_pass(self, X, training=True):
        self.layer_input = X

        # First linear layer + ReLU
        self.z1 = np.matmul(X, self.W1) + self.b1
        self.a1 = self._relu(self.z1)

        # Second linear layer
        output = np.matmul(self.a1, self.W2) + self.b2

        return output

    def backward_pass(self, accum_grad):
        # Gradients for W2 and b2
        grad_W2 = self.a1.T.dot(accum_grad)
        grad_b2 = np.sum(accum_grad, axis=0, keepdims=True)

        # Gradient through W2
        grad_a1 = accum_grad.dot(self.W2.T)

        # Gradient through ReLU
        grad_z1 = grad_a1 * (self.z1 > 0).astype(float)

        # Gradients for W1 and b1
        grad_W1 = self.layer_input.T.dot(grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        # Update weights
        if self.trainable:
            self.W2 = self.W2_opt.update(self.W2, grad_W2)
            self.b2 = self.b2_opt.update(self.b2, grad_b2)
            self.W1 = self.W1_opt.update(self.W1, grad_W1)
            self.b1 = self.b1_opt.update(self.b1, grad_b1)

        # Return gradient to input
        accum_grad = grad_z1.dot(self.W1.T)
        return accum_grad

    def output_shape(self):
        return self.input_shape


class PositionalEncoding(Layer):
    """Positional Encoding layer using sinusoidal functions.

    Parameters:
    -----------
    max_len: int
        Maximum sequence length
    d_model: int
        The dimension of the model
    """
    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.pe = None
        self.trainable = False

    def initialize(self, optimizer):
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe.reshape(1, max_len, d_model)

    def forward_pass(self, X, training=True):
        # X shape: (batch_size, seq_length, d_model)
        seq_length = X.shape[1]
        return X + self.pe[:, :seq_length, :]

    def backward_pass(self, accum_grad):
        # Positional encoding is not trainable
        return accum_grad

    def output_shape(self):
        return (self.max_len, self.d_model)


class TransformerEncoderLayer(Layer):
    """Single Transformer Encoder Layer.

    Parameters:
    -----------
    d_model: int
        The dimension of the model
    n_heads: int
        Number of attention heads
    d_ff: int
        Dimension of feed-forward hidden layer
    dropout_rate: float
        Dropout rate (not implemented in this basic version)
    input_shape: tuple
        Expected input shape
    """
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1, input_shape=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.trainable = True

        # Sub-layers
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def initialize(self, optimizer):
        self.multi_head_attention.initialize(optimizer)
        self.feed_forward.initialize(optimizer)
        self.layer_norm1.initialize()
        self.layer_norm2.initialize()

    def parameters(self):
        return (self.multi_head_attention.parameters() +
                self.feed_forward.parameters() +
                self.layer_norm1.parameters() +
                self.layer_norm2.parameters())

    def forward_pass(self, X, training=True):
        # Multi-head attention with residual connection and layer norm
        attn_output = self.multi_head_attention.forward_pass(X, training)
        X = self.layer_norm1.forward_pass(X + attn_output)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward.forward_pass(X, training)
        X = self.layer_norm2.forward_pass(X + ff_output)

        return X

    def backward_pass(self, accum_grad):
        # Backward through layer norm 2
        accum_grad = self.layer_norm2.backward_pass(accum_grad)

        # Backward through feed-forward
        accum_grad = self.feed_forward.backward_pass(accum_grad)

        # Backward through layer norm 1
        accum_grad = self.layer_norm1.backward_pass(accum_grad)

        # Backward through multi-head attention
        accum_grad = self.multi_head_attention.backward_pass(accum_grad)

        return accum_grad

    def output_shape(self):
        return self.input_shape


class LayerNorm(Layer):
    """Layer Normalization.

    Parameters:
    -----------
    d_model: int
        The dimension of the model
    """
    def __init__(self, d_model):
        self.d_model = d_model
        self.trainable = True
        self.gamma = None
        self.beta = None
        self.eps = 1e-6

    def initialize(self):
        self.gamma = np.ones((1, self.d_model))
        self.beta = np.zeros((1, self.d_model))

    def parameters(self):
        return 2 * self.d_model

    def forward_pass(self, X, training=True):
        self.layer_input = X

        # Compute mean and variance
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.var = np.var(X, axis=-1, keepdims=True)

        # Normalize
        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.eps)

        # Scale and shift
        output = self.gamma * self.X_norm + self.beta

        return output

    def backward_pass(self, accum_grad):
        # Gradients for gamma and beta
        grad_gamma = np.sum(accum_grad * self.X_norm, axis=0, keepdims=True)
        grad_beta = np.sum(accum_grad, axis=0, keepdims=True)

        # Gradient through normalization
        N = accum_grad.shape[-1]
        grad_X_norm = accum_grad * self.gamma

        # Gradient through variance
        grad_var = np.sum(grad_X_norm * (self.layer_input - self.mean) * -0.5 *
                         np.power(self.var + self.eps, -1.5), axis=-1, keepdims=True)

        # Gradient through mean
        grad_mean = np.sum(grad_X_norm * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True)

        # Gradient through input
        grad_input = grad_X_norm / np.sqrt(self.var + self.eps) + \
                     grad_var * 2 * (self.layer_input - self.mean) / self.d_model + \
                     grad_mean / self.d_model

        # Update parameters
        if self.trainable:
            self.gamma -= 0.01 * grad_gamma
            self.beta -= 0.01 * grad_beta

        return grad_input

    def output_shape(self):
        return self.d_model


class TransformerEncoder(Layer):
    """Full Transformer Encoder for sequence classification.

    Parameters:
    -----------
    vocab_size: int
        Size of the vocabulary
    d_model: int
        The dimension of the model
    n_heads: int
        Number of attention heads
    d_ff: int
        Dimension of feed-forward hidden layer
    n_layers: int
        Number of encoder layers
    max_len: int
        Maximum sequence length
    n_classes: int
        Number of output classes
    """
    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512,
                 n_layers=3, max_len=100, n_classes=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_len = max_len
        self.n_classes = n_classes
        self.trainable = True

        # Embedding layer
        self.embedding = Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(max_len, d_model)

        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # Classification head
        self.classification_head = ClassificationHead(d_model, n_classes)

    def initialize(self, optimizer):
        self.embedding.initialize()
        self.positional_encoding.initialize()

        for layer in self.encoder_layers:
            layer.initialize(optimizer)

        self.classification_head.initialize(optimizer)

    def parameters(self):
        total = self.embedding.parameters() + self.classification_head.parameters()
        for layer in self.encoder_layers:
            total += layer.parameters()
        return total

    def forward_pass(self, X, training=True):
        # Embed input tokens
        X = self.embedding.forward_pass(X)

        # Add positional encoding
        X = self.positional_encoding.forward_pass(X)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            X = layer.forward_pass(X, training)

        # Global average pooling
        X = np.mean(X, axis=1)

        # Classification
        output = self.classification_head.forward_pass(X, training)

        return output

    def backward_pass(self, accum_grad):
        # Backward through classification head
        accum_grad = self.classification_head.backward_pass(accum_grad)

        # Backward through encoder layers (simplified)
        for layer in reversed(self.encoder_layers):
            accum_grad = layer.backward_pass(accum_grad)

        return accum_grad

    def output_shape(self):
        return (self.n_classes,)


class Embedding(Layer):
    """Embedding layer that maps token indices to dense vectors.

    Parameters:
    -----------
    vocab_size: int
        Size of the vocabulary
    d_model: int
        The dimension of the embedding vectors
    """
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.trainable = True
        self.W = None

    def initialize(self):
        # Initialize with random normal
        self.W = np.random.randn(self.vocab_size, self.d_model) * 0.1

    def parameters(self):
        return self.vocab_size * self.d_model

    def forward_pass(self, X, training=True):
        # X shape: (batch_size, seq_length) - token indices
        self.layer_input = X
        return self.W[X]

    def backward_pass(self, accum_grad):
        # Gradient for embedding weights
        if self.trainable:
            grad_W = np.zeros_like(self.W)
            for i in range(self.layer_input.shape[0]):
                for j in range(self.layer_input.shape[1]):
                    idx = self.layer_input[i, j]
                    grad_W[idx] += accum_grad[i, j]

            self.W -= 0.01 * grad_W

        return accum_grad

    def output_shape(self):
        return (self.d_model,)


class ClassificationHead(Layer):
    """Classification head with global average pooling and dense layer.

    Parameters:
    -----------
    d_model: int
        The dimension of the input features
    n_classes: int
        Number of output classes
    """
    def __init__(self, d_model, n_classes):
        self.d_model = d_model
        self.n_classes = n_classes
        self.trainable = True
        self.W = None
        self.b = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.d_model)
        self.W = np.random.uniform(-limit, limit, (self.d_model, self.n_classes))
        self.b = np.zeros((1, self.n_classes))
        self.W_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def parameters(self):
        return self.d_model * self.n_classes + self.n_classes

    def _softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        logits = np.matmul(X, self.W) + self.b
        output = self._softmax(logits)
        return output

    def backward_pass(self, accum_grad):
        # Gradient for W and b
        grad_W = self.layer_input.T.dot(accum_grad)
        grad_b = np.sum(accum_grad, axis=0, keepdims=True)

        # Update weights
        if self.trainable:
            self.W = self.W_opt.update(self.W, grad_W)
            self.b = self.b_opt.update(self.b, grad_b)

        # Return gradient to input
        accum_grad = accum_grad.dot(self.W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_classes,)
