"""
Example: Transformer for Sequence Classification

This example demonstrates how to use the Transformer encoder
for text classification tasks.

The Transformer architecture was introduced in the paper:
"Attention Is All You Need" (Vaswani et al., 2017)
"""

import numpy as np
from mlfromscratch.deep_learning.transformer import TransformerEncoder
from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils import train_test_split, to_categorical


def generate_synthetic_data(n_samples=1000, vocab_size=100, seq_length=20, n_classes=2):
    """Generate synthetic sequence data for demonstration."""
    # Generate random sequences
    X = np.random.randint(0, vocab_size, size=(n_samples, seq_length))

    # Generate labels based on simple rule (e.g., sum of first half vs second half)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        first_half_sum = np.sum(X[i, :seq_length//2])
        second_half_sum = np.sum(X[i, seq_length//2:])
        y[i] = 1 if first_half_sum > second_half_sum else 0

    return X, y


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    y_pred_classes = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred_classes)


def main():
    print("=" * 60)
    print("Transformer for Sequence Classification")
    print("=" * 60)

    # Hyperparameters
    vocab_size = 100
    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 2
    max_len = 20
    n_classes = 2
    learning_rate = 0.001
    n_epochs = 10
    batch_size = 32

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(
        n_samples=1000,
        vocab_size=vocab_size,
        seq_length=max_len,
        n_classes=n_classes
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # One-hot encode labels
    y_train_encoded = to_categorical(y_train, n_classes)
    y_test_encoded = to_categorical(y_test, n_classes)

    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Sequence length: {max_len}")
    print(f"   Vocabulary size: {vocab_size}")

    # Create Transformer model
    print("\n2. Creating Transformer model...")
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_len=max_len,
        n_classes=n_classes
    )

    # Initialize optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.initialize(optimizer)

    print(f"   Model parameters: {model.parameters():,}")
    print(f"   Architecture:")
    print(f"     - Embedding: {vocab_size} x {d_model}")
    print(f"     - {n_layers} Transformer layers")
    print(f"     - Attention heads: {n_heads}")
    print(f"     - Feed-forward dim: {d_ff}")

    # Training loop
    print("\n3. Training model...")
    loss_fn = CrossEntropy()

    for epoch in range(n_epochs):
        # Mini-batch training
        epoch_loss = 0
        n_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_encoded[i:i+batch_size]

            # Forward pass
            y_pred = model.forward_pass(X_batch, training=True)

            # Compute loss
            loss = loss_fn.loss(y_batch, y_pred)
            epoch_loss += loss
            n_batches += 1

            # Backward pass (simplified - just for demonstration)
            # In a full implementation, you'd compute gradients properly
            grad = loss_fn.gradient(y_batch, y_pred)
            model.backward_pass(grad)

        # Evaluate on test set
        y_test_pred = model.forward_pass(X_test, training=False)
        test_acc = accuracy(y_test, y_test_pred)

        avg_loss = epoch_loss / n_batches
        print(f"   Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_acc:.4f}")

    print("\n4. Training complete!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    model = main()
