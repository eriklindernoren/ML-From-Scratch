# -*- coding: utf-8 -*-

import numpy as np
from ..base import Layer
from ztlearn.utils import clip_gradients as cg
from ztlearn.dl.initializers import InitializeWeights as init
from ztlearn.dl.activations import ActivationFunction as activate
from ztlearn.dl.optimizers import OptimizationFunction as optimizer


class LSTM(Layer):

    # (time_steps, input_dim) = input_shape
    # input_dim ==> vocabulary size

    def __init__(self, h_units, activation = 'tanh', input_shape = None, gate_activation = 'sigmoid'):
        self.h_units = h_units # number of hidden states
        self.activation = activation
        self.input_shape = input_shape
        self.gate_activation = gate_activation

        self.init_method = None
        self.optimizer_kwargs = None

        # gate weights
        self.W_input = None
        self.W_forget = None
        self.W_output = None

        # gate bias
        self.b_input = None
        self.b_forget = None
        self.b_output = None

        # cell weights
        self.W_cell = None

        # cell bias
        self.b_cell = None

        # final output weights
        self.W_final = None

        # final output bias
        self.b_final = None

    def prep_layer(self):
        _, input_dim = self.input_shape
        z_dim = self.h_units + input_dim # concatenate (h_units, vocabulary_size) vector

        # gate weights
        self.W_input = init(self.init_method).initialize_weights((z_dim, self.h_units))
        self.W_forget = init(self.init_method).initialize_weights((z_dim, self.h_units))
        self.W_output = init(self.init_method).initialize_weights((z_dim, self.h_units))

        # gate bias
        self.b_input = np.zeros((self.h_units,))
        self.b_forget = np.zeros((self.h_units,))
        self.b_output = np.zeros((self.h_units,))

        # cell weights
        self.W_cell = init(self.init_method).initialize_weights((z_dim, self.h_units))

        # cell bias
        self.b_cell = np.zeros((self.h_units,))

        # final output weights
        self.W_final = init(self.init_method).initialize_weights((self.h_units, input_dim))

        # final output bias
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

        self.forget = np.zeros((batch_size, time_steps, self.h_units))
        self.input = np.zeros((batch_size, time_steps, self.h_units))
        self.output = np.zeros((batch_size, time_steps, self.h_units))
        self.states = np.zeros((batch_size, time_steps, self.h_units))
        self.cell_tilde = np.zeros((batch_size, time_steps, self.h_units))
        self.cell = np.zeros((batch_size, time_steps, self.h_units))
        self.final = np.zeros((batch_size, time_steps, input_dim))

        self.z = np.concatenate((self.inputs, self.states), axis=2)

        for t in range(time_steps):
            self.forget[:, t] = activate(self.gate_activation)._forward(np.dot(self.z[:, t], self.W_forget) + self.b_forget)
            self.input[:, t] = activate(self.gate_activation)._forward(np.dot(self.z[:, t], self.W_input) + self.b_input)
            self.cell_tilde[:, t] = activate(self.activation)._forward(np.dot(self.z[:, t], self.W_cell) + self.b_cell)
            self.cell[:, t] = self.forget[:, t] * self.cell[:, t-1] + self.input[:, t] * self.cell_tilde[:, t]
            self.output[:, t] = activate(self.gate_activation)._forward(np.dot(self.z[:, t], self.W_output) + self.b_output)
            self.states[:, t] = self.output[:, t] * activate(self.activation)._forward(self.cell[:, t])

            # logits
            self.final[:, t] = np.dot(self.states[:, t], self.W_final) + self.b_final

        if not train_mode:
            return activate('softmax')._forward(self.final) # if mode is not training

        return self.final

    def pass_backward(self, grad):
        _, time_steps, _ = grad.shape

        dW_forget = np.zeros_like(self.W_forget)
        dW_input = np.zeros_like(self.W_input)
        dW_output = np.zeros_like(self.W_output)
        dW_cell = np.zeros_like(self.W_cell)
        dW_final = np.zeros_like(self.W_final)

        db_forget = np.zeros_like(self.b_forget)
        db_input = np.zeros_like(self.b_input)
        db_output = np.zeros_like(self.b_output)
        db_cell = np.zeros_like(self.b_cell)
        db_final = np.zeros_like(self.b_final)

        dstates = np.zeros_like(self.states)
        dcell = np.zeros_like(self.cell)
        dcell_tilde = np.zeros_like(self.cell_tilde)
        dforget = np.zeros_like(self.forget)
        dinput = np.zeros_like(self.input)
        doutput = np.zeros_like(self.output)

        dcell_next = np.zeros_like(self.cell)
        dstates_next = np.zeros_like(self.states)

        next_grad = np.zeros_like(grad)

        for t in np.arange(time_steps)[::-1]: # reversed

            dW_final += np.dot(self.states[:, t].T, grad[:, t])
            db_final += np.sum(grad[:, t], axis = 0)

            dstates[:, t] = np.dot(grad[:, t], self.W_final.T)
            dstates[:, t] += dstates_next[:, t]
            next_grad = np.dot(dstates, self.W_final)

            doutput[:,t] = activate(self.activation)._forward(self.cell[:, t]) * dstates[:, t]
            doutput[:,t] = activate(self.gate_activation)._backward(self.output[:, t]) * doutput[:,t]
            dW_output += np.dot(self.z[:, t].T, doutput[:, t])
            db_output += np.sum(doutput[:, t], axis = 0)

            dcell[:, t] += self.output[:, t] * dstates[:, t] * activate(self.activation)._backward(self.cell[:, t])
            dcell[:, t] += dcell_next[:, t]
            dcell_tilde[:, t] = dcell[:, t] * self.input[:, t]
            dcell_tilde[:, t] = dcell_tilde[:, t] * activate(self.activation)._backward(dcell_tilde[:, t])
            dW_cell += np.dot(self.z[:, t].T, dcell[:, t])
            db_cell += np.sum(dcell[:, t], axis = 0)

            dinput[:, t] = self.cell_tilde[:, t] * dcell[:, t]
            dinput[:, t] = activate(self.gate_activation)._backward(self.input[:, t]) * dinput[:, t]
            dW_input += np.dot(self.z[:, t].T, dinput[:, t])
            db_input += np.sum(dinput[:, t], axis = 0)

            dforget[:, t] = self.cell[:, t-1] * dcell[:, t]
            dforget[:, t] = activate(self.gate_activation)._backward(self.forget[:, t]) * dforget[:, t]
            dW_forget += np.dot(self.z[:, t].T, dforget[:, t])
            db_forget += np.sum(dforget[:, t], axis = 0)

            dz_forget = np.dot(dforget[:, t], self.W_forget.T)
            dz_input = np.dot(dinput[:, t], self.W_input.T)
            dz_output = np.dot(doutput[:, t], self.W_output.T)
            dz_cell = np.dot(dcell[:, t], self.W_cell.T)

            dz = dz_forget + dz_input + dz_output + dz_cell
            dstates_next[:, t] = dz[:,:self.h_units]
            dcell_next = self.forget * dcell

        # optimize weights and bias
        self.W_final = optimizer(self.optimizer_kwargs)._update(self.W_final, cg(dW_final))
        self.b_final = optimizer(self.optimizer_kwargs)._update(self.b_final, cg(db_final))

        self.W_forget = optimizer(self.optimizer_kwargs)._update(self.W_forget, cg(dW_forget))
        self.b_forget = optimizer(self.optimizer_kwargs)._update(self.b_forget, cg(db_forget))

        self.W_input = optimizer(self.optimizer_kwargs)._update(self.W_input, cg(dW_input))
        self.b_input = optimizer(self.optimizer_kwargs)._update(self.b_input, cg(db_input))

        self.W_output = optimizer(self.optimizer_kwargs)._update(self.W_output, cg(dW_output))
        self.b_output = optimizer(self.optimizer_kwargs)._update(self.b_output, cg(db_output))

        self.W_cell = optimizer(self.optimizer_kwargs)._update(self.W_cell, cg(dW_cell))
        self.b_cell = optimizer(self.optimizer_kwargs)._update(self.b_cell, cg(db_cell))

        return next_grad
