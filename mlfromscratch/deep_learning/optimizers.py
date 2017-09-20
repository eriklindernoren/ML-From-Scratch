import numpy as np
from mlfromscratch.utils import make_diagonal, normalize

# Optimizers for models that use gradient based methods for finding the 
# weights that minimizes the loss.
# A great resource for understanding these methods: 
# http://sebastianruder.com/optimizing-gradient-descent/index.html

class StochasticGradientDescent():
    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_updt

class NesterovAcceleratedGradient():
    def __init__(self, learning_rate=0.001, momentum=0.4):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_func):
        # Calculate the gradient of the loss a bit further down the slope from w
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.w_updt), -1, 1)
        # Initialize on first update
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))

        self.w_updt = self.momentum * self.w_updt + self.learning_rate * approx_future_grad
        # Move against the gradient to minimize loss
        return w - self.w_updt

class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)
        # Adaptive gradient with higher learning rate for sparse data
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)

class Adadelta():
    def __init__(self, rho=0.95, eps=1e-6):
        self.E_w_updt = None # Running average of squared parameter updates
        self.E_grad = None   # Running average of the squared gradient of w
        self.w_updt = None   # Parameter update
        self.eps = eps
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
            self.E_w_updt = np.zeros(np.shape(w))
            self.E_grad = np.zeros(np.shape(grad_wrt_w))

        # Update average of gradients at w
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)
        
        RMS_delta_w = np.sqrt(self.E_w_updt + self.eps)
        RMS_grad = np.sqrt(self.E_grad + self.eps)

        # Adaptive learning rate
        adaptive_lr = RMS_delta_w / RMS_grad

        # Calculate the update
        self.w_updt = adaptive_lr * grad_wrt_w

        # Update the running average of w updates
        self.E_w_updt = self.rho * self.E_w_updt + (1 - self.rho) * np.power(self.w_updt, 2)

        return w - self.w_updt

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate *  grad_wrt_w / np.sqrt(self.Eg + self.eps)

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))
        
        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt



