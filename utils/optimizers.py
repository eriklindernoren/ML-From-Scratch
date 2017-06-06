import numpy as np
from data_manipulation import make_diagonal

# Optimizers for models that use gradient methods for finding the 
# weights that minimizes the loss.
# A good resource: 
# http://sebastianruder.com/optimizing-gradient-descent/index.html


class GradientDescent():
    def __init__(self, learning_rate, momentum=0):
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
    def __init__(self, learning_rate, momentum=0):
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
    def __init__(self, learning_rate, momentum=0):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = np.array([])

    def update(self, w, grad_func):
        # Calculate the gradient of the loss a bit further down the slope from w
        grad_at_w = grad_func(w - self.momentum * self.w_updt)
        # Initialize on first update
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentum * self.w_updt + self.learning_rate * grad_at_w
        # Move against the gradient to minimize loss
        return w - self.w_updt


class Adagrad():
    def __init__(self, learning_rate, momentum=0):
        self.learning_rate = .1
        self.G = np.array([])
        self.err = 1e-8

    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)

        # If not initialized
        if not self.G.any():
            self.G = np.zeros(np.shape(w))
        
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_at_w, 2)

        # Adaptive gradient with higher learning rate for sparse data
        w_updt = self.learning_rate * np.linalg.pinv(np.sqrt(self.G + self.err)).T * grad_at_w

        return w - w_updt


class Adadelta():
    def __init__(self, learning_rate=0, momentum=0):
        self.Et = np.array([]) # Running average of theta
        self.Eg = np.array([]) # Running average of the gradient of theta
        self.w_updt = np.array([]) # Parameter update
        self.err = 1e-8
        self.gamma = 0.1


    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)

        # If not initialized
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))
            self.Et = np.zeros(np.shape(w))
            self.Eg = np.power(grad_at_w, 2)
        else:
            self.Et = self.gamma * self.Et + (1 - self.gamma) * np.power(self.w_updt, 2)
            self.Eg = self.gamma * self.Eg + (1 - self.gamma) * np.power(grad_at_w, 2)
        
        RMS_theta = np.sqrt(self.Et + self.err)
        RMS_grad = np.sqrt(self.Eg + self.err)

        # Adaptiv gradient with higher learning rate for sparse data
        self.w_updt = RMS_theta * np.linalg.pinv(RMS_grad).T * grad_at_w

        return w - self.w_updt

class RMSprop():
    def __init__(self, learning_rate=0.001, momentum=0):
        self.learning_rate = learning_rate
        self.Eg = np.array([]) # Running average of the gradient of theta
        self.err = 1e-8
        self.gamma = 0.9

    def update(self, w, grad_func):
        # Calculate the gradient of the loss at w
        grad_at_w = grad_func(w)

        # If not initialized
        if not self.Eg.any():
            self.Eg = np.power(grad_at_w, 2)
        else:
            self.Eg = self.gamma * self.Eg + (1 - self.gamma) * np.power(grad_at_w, 2)

        # Adaptiv gradient with higher learning rate for sparse data
        self.w_updt = self.learning_rate * np.linalg.pinv(np.sqrt(self.Eg + self.err)).T * grad_at_w

        return w - self.w_updt

class Adam():
    def __init__(self, learning_rate=0.001, momentum=0, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.err = 1e-8

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

        self.w_updt = self.learning_rate * np.linalg.pinv(np.sqrt(v_hat) + self.err).T * m_hat

        return w - self.w_updt



