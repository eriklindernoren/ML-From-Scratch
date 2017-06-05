import numpy as np

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