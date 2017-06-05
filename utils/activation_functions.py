import numpy as np

# Collection of activation functions
# Reference: https://en.wikipedia.org/wiki/Activation_function

class Sigmoid():
    def __init__(self): pass 

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.function(x) * (1 - self.function(x))

class TanH():
    def __init__(self): pass 

    def function(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.function(x), 2)

class ReLU():
    def __init__(self): pass 

    def function(self, x):
        # If x < 0: output = 0 else: output = x
        h = x
        h[x < 0] = 0
        return h

    def gradient(self, x):
        # If x < 0: output = 0 else: output = 1
        dx = np.zeros(np.shape(x))
        dx[x >= 0] = 1
        return dx

class LeakyReLU():
    def __init__(self): pass 

    def function(self, x):
        # If x < 0: output = 0.01*x else: output = x
        h = x
        h[h < 0] = 0.01 * h[h < 0]
        return h

    def gradient(self, x):
        # If x < 0: output = 0.01 else: output = 1
        dx = np.ones(np.shape(x))
        dx[x < 0] = 0.01
        return dx

class ExpLU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def function(self, x):
        # If x < 0: output = alpha*(exp(x)-1) else: output = x
        h = x
        h[h < 0] = self.alpha * (np.exp(h[h < 0]) - 1)
        return h

    def gradient(self, x):
        # If x < 0: output = f(alpha, x) + alpha else: output = 1
        dx = self.function(x) + self.alpha
        dx[x >= 0] = 1
        return dx

class SoftPlus():
    def __init__(self): pass 

    def function(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return Sigmoid().function(x)