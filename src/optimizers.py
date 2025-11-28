import numpy as np


class Optimizer:
    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameter, gradient):
        parameter -= self.learning_rate * gradient
        return parameter

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}   
        self.v = {}   
        self.t = 0   

    def step(self, parameter, gradient):
        param_id = id(parameter)

        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(parameter)
            self.v[param_id] = np.zeros_like(parameter)

        m = self.m[param_id]
        v = self.v[param_id]

        self.t += 1
        # Update biased first moment estimate
        m[:] = self.beta1 * m + (1 - self.beta1) * gradient
        v[:] = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)

        # Bias correction
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameter
