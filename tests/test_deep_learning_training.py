import unittest

import numpy as np

from mlfromscratch.deep_learning.layers import Dense
from mlfromscratch.deep_learning.loss_functions import SquareLoss
from mlfromscratch.deep_learning.neural_network import NeuralNetwork
from mlfromscratch.deep_learning.optimizers import Adam, AdamW


class DeepLearningTrainingTests(unittest.TestCase):
    def test_adam_bias_correction_advances_each_update(self):
        optimizer = Adam(learning_rate=0.1)
        weight = np.array([0.0])

        weight = optimizer.update(weight, np.array([1.0]))
        first_update = weight.copy()
        weight = optimizer.update(weight, np.array([2.0]))

        self.assertAlmostEqual(float(first_update[0]), -0.1, places=6)
        self.assertLess(float(weight[0]), float(first_update[0]))
        self.assertEqual(optimizer.t, 2)

    def test_adamw_decouples_weight_decay(self):
        optimizer = AdamW(learning_rate=0.1, weight_decay=0.2)
        updated = optimizer.update(np.array([2.0]), np.array([0.0]))

        np.testing.assert_allclose(updated, np.array([1.96]))

    def test_seeded_fit_shuffles_reproducibly(self):
        X = np.arange(12, dtype=float).reshape(-1, 1)
        y = 2 * X + 1

        def train_once():
            np.random.seed(13)
            model = NeuralNetwork(optimizer=Adam(learning_rate=0.01), loss=SquareLoss)
            model.add(Dense(1, input_shape=(1,)))
            history = model.fit(X, y, n_epochs=3, batch_size=4, seed=21)
            return model.layers[0].W.copy(), history[0]

        weights_a, losses_a = train_once()
        weights_b, losses_b = train_once()

        np.testing.assert_allclose(weights_a, weights_b)
        np.testing.assert_allclose(losses_a, losses_b)

    def test_early_stopping_restores_best_weights(self):
        class Layer:
            W = np.array([0.0])

        class ControlledNetwork(NeuralNetwork):
            def __init__(self):
                super(ControlledNetwork, self).__init__(Adam(), SquareLoss)
                self.layers = [Layer()]
                self.validation_losses = iter([0.5, 0.4, 0.41, 0.42])

            def train_on_batch(self, X, y):
                self.layers[0].W += 1
                return 1.0, 0.0

            def test_on_batch(self, X, y):
                return next(self.validation_losses), 0.0

        model = ControlledNetwork()
        X = np.ones((2, 1))
        y = np.ones((2, 1))
        model.fit(
            X, y, n_epochs=10, batch_size=2, shuffle=False,
            validation_data=(X, y), patience=1
        )

        self.assertEqual(len(model.errors["validation"]), 3)
        np.testing.assert_array_equal(model.layers[0].W, np.array([2.0]))


if __name__ == "__main__":
    unittest.main()
