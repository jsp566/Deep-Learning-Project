import numpy as np


class Optimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            layer.weights -= self.lr * layer.dweights
            layer.biases -= self.lr * layer.dbiases

