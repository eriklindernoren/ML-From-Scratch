import numpy as np
from data_manipulation import make_diagonal

# Optimizers for models that use gradient methods for finding the 
# weights that minimizes the loss.
# A good resource: 
# http://sebastianruder.com/optimizing-gradient-descent/index.html

class GradientDescent():
    def __init__(self, learning_rate=0.01, momentum):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_wrt_w):
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_updt

class GradientDescent_():
    def __init__(self, learning_rate=0.001, momentum=):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_func):
        # Initialize on first update
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + self.learning_rate * grad_func(w)
        # Move against the gradient to minimize loss
        return w -  self.w_updt

class NesterovAcceleratedGradient():
    def __init__(self, learning_rate=0.001, momentum=0.4):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_func):
        # Calculate the gradient of the loss a bit further down the slope from w
        approx_future_grad = grad_func(w - self.momentum * self.w_updt)
        # Initialize on first update
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))

        self.w_updt = self.momentum * self.w_updt + self.learning_rate * approx_future_grad
        # Move against the gradient to minimize loss
        return w - self.w_updt


class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = np.array([]) # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)
        # If not initialized
        if not self.G.any():
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_at_w, 2)
        # Adaptive gradient with higher learning rate for sparse data
        w_updt = self.learning_rate * np.linalg.pinv(np.sqrt(self.G + self.eps)).T * grad_at_w

        return w - w_updt


class Adadelta():
    def __init__(self, rho=0.95, eps=1e-6):
        self.E_w_updt = np.array([]) # Running average of squared parameter updates
        self.E_grad = np.array([]) # Running average of the squared gradient of w
        self.w_updt = np.array([]) # Parameter update
        self.eps = eps
        self.rho = rho

    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)
        # If not initialized
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))
            self.E_w_updt = np.zeros(np.shape(w))
            self.E_grad = np.zeros(np.shape(grad_at_w))

        # Update average of gradients at w
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_at_w, 2)
        
        RMS_delta_w = np.sqrt(self.E_w_updt + self.eps)
        RMS_grad = np.sqrt(self.E_grad + self.eps)

        # Adaptive learning rate
        adaptive_lr = RMS_delta_w * np.linalg.pinv(RMS_grad).T 

        # Calculate the update
        self.w_updt = adaptive_lr * grad_at_w

        # Update the running average of w updates
        self.E_w_updt = self.rho * self.E_w_updt + (1 - self.rho) * np.power(self.w_updt, 2)

        return w - self.w_updt

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = np.array([]) # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)
        # If not initialized
        if not self.Eg.any():
            self.Eg = np.zeros(np.shape(grad_at_w))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_at_w, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        self.w_updt = self.learning_rate * np.linalg.pinv(np.sqrt(self.Eg + self.eps)).T * grad_at_w

        return w - self.w_updt

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = np.array([])
        self.v = np.array([]) 
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)

        # If not initialized
        if not self.m.any():
            self.m = np.zeros(np.shape(grad_at_w))
            self.v = np.zeros(np.shape(grad_at_w))
        
        self.m = self.b1 * self.m + (1 - self.b1) * grad_at_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_at_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * np.linalg.pinv(np.sqrt(v_hat) + self.eps).T * m_hat

        return w - self.w_updt



