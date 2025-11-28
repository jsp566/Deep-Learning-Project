import numpy as np


# Activations function base class
class ActivationFunction:
    def forward(self, x, training=True):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def backward(self, x, regularization=None):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def zero_gradients(self):
        pass  # Activation functions have no gradients to zero

    def update_params(self, optimizer):
        pass  # Activation functions have no parameters to update


class Identity(ActivationFunction):
    def forward(self, x, training=True):
        self.inputs = x
        return x

    def backward(self, loss_grad, regularization=None):
        return np.ones_like(self.inputs) * loss_grad

class ReLU(ActivationFunction):
    def forward(self, x, training=True):
        self.inputs = x
        return np.maximum(0, x)

    def backward(self, loss_grad, regularization=None):
        grad = np.where(self.inputs > 0, 1, 0)
        return grad * loss_grad


class Sigmoid(ActivationFunction):
    def forward(self, x, training=True):
        self.inputs = x
        return 1 / (1 + np.exp(-x))

    def backward(self, loss_grad, regularization=None):
        sig = self.forward(self.inputs)
        return sig * (1 - sig) * loss_grad


class Tanh(ActivationFunction):
    def forward(self, x, training=True):
        self.inputs = x
        return np.tanh(x)

    def backward(self, loss_grad, regularization=None):
        tanh_x = self.forward(self.inputs)
        return (1 - tanh_x**2) * loss_grad


class LeakyReLU(ActivationFunction):
    def forward(self, x, training=True):
        self.inputs = x
        return np.maximum(0, x) + 0.1 * np.minimum(0, x)

    def backward(self, loss_grad, regularization=None):
        grad = np.where(self.inputs > 0, 1, 0.1)
        return grad * loss_grad
