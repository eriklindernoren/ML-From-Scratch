# -*- coding: utf-8 -*-

import numpy as np
from ..base import Layer
from ztlearn.utils import clip_gradients as cg
from ztlearn.dl.initializers import InitializeWeights as init
from ztlearn.dl.activations import ActivationFunction as activate
from ztlearn.dl.optimizers import OptimizationFunction as optimizer


class GRU(Layer):

    def __init__(self, h_units, activation = 'tanh', input_shape = None, gate_activation = 'sigmoid'):
        self.h_units = h_units # number of hidden states
        self.activation = activation
        self.input_shape = input_shape
        self.gate_activation = gate_activation

        self.init_method = None # just added
        self.optimizer_kwargs = None # just added

        # gate weights
        self.W_update = None
        self.W_reset = None
        self.W_states = None

        # gate bias
        self.b_update = None
        self.b_reset = None
        self.b_states = None

        # final output to nodes weights
        self.W_final = None

        # final output to nodes bias
        self.b_final = None

    def prep_layer(self):
        _, input_dim = self.input_shape
        z_dim = self.h_units + input_dim # concatenate (h_units, vocabulary_size) vector

        # gate weights
        self.W_update = init(self.init_method).initialize_weights((z_dim, self.h_units))
        self.W_reset = init(self.init_method).initialize_weights((z_dim, self.h_units))
        self.W_cell = init(self.init_method).initialize_weights((z_dim, self.h_units))
        self.W_states = init(self.init_method).initialize_weights((z_dim, self.h_units))

        # gate hidden bias
        self.b_update = np.zeros((self.h_units,))
        self.b_reset = np.zeros((self.h_units,))
        self.b_cell = np.zeros((self.h_units,))
        self.b_states = np.zeros((self.h_units,))

        # final output to nodes weights (input_dim is the vocab size and also the ouput size)
        self.W_final = init(self.init_method).initialize_weights((self.h_units, input_dim))

        # final output to nodes bias (input_dim is the vocab size and also the ouput size)
        self.b_final = np.zeros((input_dim,))

    @property
    def weight_initializer(self):
        return self.init_method

    @weight_initializer.setter
    def weight_initializer(self, init_method):
        self.init_method = init_method

    @property
    def weight_optimizer(self):
        return self.optimizer_kwargs

    @weight_optimizer.setter
    def weight_optimizer(self, optimizer_kwargs = {}):
        self.optimizer_kwargs = optimizer_kwargs

    @property
    def layer_activation(self):
        return self.activation

    @layer_activation.setter
    def layer_activation(self, activation):
        self.activation = activation

    @property
    def output_shape(self):
        return self.input_shape

    def pass_forward(self, inputs, train_mode = True):
        self.inputs = inputs
        batch_size, time_steps, input_dim = inputs.shape

        self.update = np.zeros((batch_size, time_steps, self.h_units))
        self.reset = np.zeros((batch_size, time_steps, self.h_units))
        self.cell = np.zeros((batch_size, time_steps, self.h_units))
        self.states = np.zeros((batch_size, time_steps, self.h_units))
        self.final = np.zeros((batch_size, time_steps, input_dim))

        self.z = np.concatenate((self.inputs, self.states), axis = 2)
        self.z_tilde = np.zeros_like(self.z)

        for t in range(time_steps):
            self.update[:, t] = activate(self.gate_activation)._forward(np.dot(self.z[:, t], self.W_update) + self.b_update)
            self.reset[:, t] = activate(self.gate_activation)._forward(np.dot(self.z[:, t], self.W_reset) + self.b_reset)
            self.z_tilde[:, t] = np.concatenate((self.reset[:, t] * self.states[:, t-1], self.inputs[:, t]), axis = 1)
            self.cell[:, t] = activate(self.activation)._forward(np.dot(self.z_tilde[:, t-1], self.W_cell) + self.b_cell)
            self.states[:, t] = (1. - self.update[:, t]) * self.states[:, t-1]  + self.update[:, t] * self.cell[:, t]

            # logits
            self.final[:, t] = np.dot(self.states[:, t], self.W_final) + self.b_final

        if not train_mode:
            return activate('softmax')._forward(self.final) # if mode is not training

        return self.final

    def pass_backward(self, grad):
        _, time_steps, _ = grad.shape

        dW_update = np.zeros_like(self.W_update)
        dW_reset = np.zeros_like(self.W_reset)
        dW_cell = np.zeros_like(self.W_cell)
        dW_final = np.zeros_like(self.W_final)

        db_update = np.zeros_like(self.b_update)
        db_reset = np.zeros_like(self.b_reset)
        db_cell = np.zeros_like(self.b_cell)
        db_final = np.zeros_like(self.b_final)

        dstates = np.zeros_like(self.states)
        dstate_a = np.zeros_like(self.states)
        dstate_b = np.zeros_like(self.states)
        dstate_c = np.zeros_like(self.states)
        dstates_next = np.zeros_like(self.states)
        dstates_prime = np.zeros_like(self.states)

        dz_cell = np.zeros_like(self.cell)
        dcell = np.zeros_like(self.cell)

        dz_reset = np.zeros_like(self.reset)
        dreset = np.zeros_like(self.reset)

        dz_update = np.zeros_like(self.update)
        dupdate = np.zeros_like(self.update)

        next_grad = np.zeros_like(grad)

        for t in np.arange(time_steps)[::-1]: # reversed

            dW_final += np.dot(self.states[:, t].T, grad[:, t])
            db_final += np.sum(grad[:, t], axis = 0)

            dstates[:, t] = np.dot(grad[:, t], self.W_final.T)
            dstates[:, t] += dstates_next[:, t]
            next_grad = np.dot(dstates, self.W_final)

            dcell[:, t] = self.update[:, t] * dstates[:, t]
            dstate_a[:, t] = (1. - self.update[:, t]) * dstates[:, t]
            dupdate[:, t] = self.cell[:, t] * dstates[:, t] - self.states[:, t-1] * dstates[:, t]

            dcell[:, t] = activate(self.activation)._backward(self.cell[:, t]) * dcell[:, t]
            dW_cell += np.dot(self.z_tilde[:, t-1].T, dcell[:, t])
            db_cell += np.sum(dcell[:, t], axis = 0)
            dz_cell = np.dot(dcell[:, t], self.W_cell.T)

            dstates_prime[:, t] = dz_cell[:, :self.h_units]
            dstate_b[:, t] = self.reset[:, t] * dstates_prime[:, t]

            dreset[:, t] = self.states[:, t-1] * dstates_prime[:, t]
            dreset[:, t] = activate(self.gate_activation)._backward(self.reset[:, t]) * dreset[:, t]
            dW_reset += np.dot(self.z[:, t].T, dreset[:, t])
            db_reset += np.sum(dreset[:, t], axis = 0)
            dz_reset = np.dot(dreset[:, t], self.W_reset.T)

            dupdate[:, t] = activate(self.gate_activation)._backward(self.update[:, t]) * dupdate[:, t]
            dW_update += np.dot(self.z[:, t].T, dupdate[:, t])
            db_update += np.sum(dupdate[:, t], axis = 0)
            dz_update = np.dot(dupdate[:, t], self.W_update.T)

            dz = dz_reset + dz_update
            dstate_c[:, t] = dz[:, :self.h_units]

            dstates_next = dstate_a + dstate_b + dstate_c

        # optimize weights and bias
        self.W_final = optimizer(self.optimizer_kwargs)._update(self.W_final, cg(dW_final))
        self.b_final = optimizer(self.optimizer_kwargs)._update(self.b_final, cg(db_final))

        self.W_cell = optimizer(self.optimizer_kwargs)._update(self.W_cell, cg(dW_cell))
        self.b_cell = optimizer(self.optimizer_kwargs)._update(self.b_cell, cg(db_cell))

        self.W_reset = optimizer(self.optimizer_kwargs)._update(self.W_reset, cg(dW_reset))
        self.b_reset = optimizer(self.optimizer_kwargs)._update(self.b_reset, cg(db_reset))

        self.W_update = optimizer(self.optimizer_kwargs)._update(self.W_update, cg(dW_update))
        self.b_update = optimizer(self.optimizer_kwargs)._update(self.b_update, cg(db_update))

        return next_grad
