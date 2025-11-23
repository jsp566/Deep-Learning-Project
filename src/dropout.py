import numpy as np

class Dropout:
    def __init__(self, drop_probability=0.5):
        self.drop_probability = drop_probability
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.drop_probability) / (1.0 - self.drop_probability)
            return x * self.mask
        else:
            return x

    def backward(self, grad_output):
        if self.mask is not None:
            return grad_output * self.mask
        else:
            raise ValueError("Cannot backpropagate before forward pass during training.")