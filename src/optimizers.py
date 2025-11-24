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

