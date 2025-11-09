import numpy as np


class FFNN:
    def __init__(self, layers, loss_function, optimizer):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward_pass(self, batch_x):
        for layer in self.layers:
            batch_x = layer.forward(batch_x)
        return batch_x
    
    def backward_pass(self, batch_y, y_pred):

        loss_grad = self.loss_function.gradient(batch_y, y_pred)
        
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def train(self, batch_x, batch_y):

        y_pred = self.forward_pass(batch_x)

        loss = self.loss_function.compute(batch_y, y_pred)
        self.backward_pass(batch_y, y_pred)

        for layer in self.layers:
            layer.update_params(self.optimizer)
            layer.zero_gradients()

    def train_loop(self):
        # for each epoch:
        # Mini batch gradient descent
        # train

        raise NotImplementedError("Training loop not implemented yet.")

    def pred(self):
        # TBD
        raise NotImplementedError("Prediction method not implemented yet.")
