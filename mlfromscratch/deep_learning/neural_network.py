from __future__ import print_function, division
import copy
from terminaltables import AsciiTable
import numpy as np
import progressbar
from mlfromscratch.utils import batch_iterator
from mlfromscratch.utils.misc import bar_widgets


class NeuralNetwork():
    """Neural Network. Deep Learning base model.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.accuracies = {"training": [], "validation": []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.val_set = None
        if validation_data is not None:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}

    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers. """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to the network
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size, shuffle=True, seed=None,
            validation_data=None, patience=None, min_delta=0.0,
            restore_best_weights=True):
        """Train the model, optionally stopping when validation loss plateaus.

        Set ``seed`` to make the per-epoch shuffling reproducible. Validation data
        passed here takes precedence over data supplied to the constructor.
        """
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if len(X) == 0:
            raise ValueError("X and y must contain at least one sample")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if patience is not None and patience < 0:
            raise ValueError("patience must be non-negative")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        val_set = self.val_set
        if validation_data is not None:
            X_val, y_val = validation_data
            if len(X_val) != len(y_val):
                raise ValueError("Validation X and y must contain the same number of samples")
            val_set = {"X": X_val, "y": y_val}
        if patience is not None and val_set is None:
            raise ValueError("Early stopping requires validation data")

        rng = np.random.RandomState(seed)
        best_loss = np.inf
        best_weights = None
        epochs_without_improvement = 0

        for _ in self.progressbar(range(n_epochs)):
            if shuffle:
                indices = rng.permutation(len(X))
                X_epoch, y_epoch = X[indices], y[indices]
            else:
                X_epoch, y_epoch = X, y

            total_loss = 0.0
            total_accuracy = 0.0
            for X_batch, y_batch in batch_iterator(X_epoch, y_epoch, batch_size=batch_size):
                loss, accuracy = self.train_on_batch(X_batch, y_batch)
                batch_samples = len(X_batch)
                total_loss += loss * batch_samples
                total_accuracy += accuracy * batch_samples

            self.errors["training"].append(total_loss / len(X))
            self.accuracies["training"].append(total_accuracy / len(X))

            if val_set is not None:
                val_loss, val_accuracy = self.test_on_batch(val_set["X"], val_set["y"])
                self.errors["validation"].append(val_loss)
                self.accuracies["validation"].append(val_accuracy)

                if patience is not None:
                    if val_loss < best_loss - min_delta:
                        best_loss = val_loss
                        epochs_without_improvement = 0
                        best_weights = self._snapshot_weights()
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            break

        if restore_best_weights and best_weights is not None:
            self._restore_weights(best_weights)

        return self.errors["training"], self.errors["validation"]

    def _snapshot_weights(self):
        """Copy model parameters and batch-normalization inference statistics."""
        state_names = ("W", "w0", "U", "V", "gamma", "beta",
                       "running_mean", "running_var")
        weights = []
        for layer in self.layers:
            state = {
                name: np.copy(getattr(layer, name))
                for name in state_names
                if hasattr(layer, name) and getattr(layer, name) is not None
            }
            state["optimizers"] = {
                name: copy.deepcopy(value.__dict__)
                for name, value in vars(layer).items()
                if name.endswith("_opt")
            }
            weights.append(state)
        return weights

    def _restore_weights(self, weights):
        for layer, state in zip(self.layers, weights):
            for name, value in state.items():
                if name == "optimizers":
                    for optimizer_name, optimizer_state in value.items():
                        optimizer = getattr(layer, optimizer_name)
                        optimizer.__dict__.clear()
                        optimizer.__dict__.update(copy.deepcopy(optimizer_state))
                    continue
                setattr(layer, name, value)

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def summary(self, name="Model Summary"):
        # Print model name
        print (AsciiTable([[name]]).table)
        # Network input shape (first layer's input shape)
        print ("Input Shape: %s" % str(self.layers[0].input_shape))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        # Print network configuration table
        print (AsciiTable(table_data).table)
        print ("Total Parameters: %d\n" % tot_params)

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward_pass(X, training=False)
