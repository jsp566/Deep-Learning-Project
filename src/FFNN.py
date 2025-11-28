import numpy as np


class FFNN:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, batch_x, training=True):
        for layer in self.layers:
            batch_x = layer.forward(batch_x, training=training)
        return batch_x
    
    def backward_pass(self, loss_grad, regularization=None):
        #loss_grad = self.loss_function.gradient(batch_y, y_pred)
        
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, regularization=regularization)


    def update_params(self, optimizer):
        for layer in self.layers:
            layer.update_params(optimizer)
            layer.zero_gradients()


    def pred(self, X):
        return self.forward_pass(X, training=False)