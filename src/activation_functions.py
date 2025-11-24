import numpy as np


# Activations function base class
class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def backward(self, x):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def zero_gradients(self):
        pass  # Activation functions have no gradients to zero

    def update_params(self, optimizer):
        pass  # Activation functions have no parameters to update


class Identity(ActivationFunction):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class ReLU(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        grad = np.where(x > 0, 1, 0)
        return grad


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)


class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        tanh_x = self.forward(x)
        return 1 - tanh_x**2


class LeakyReLU(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x) + 0.1 * np.minimum(0, x)

    def backward(self, x):
        grad = np.where(x > 0, 1, 0.1)
        return grad
