import numpy as np


# parameter værdier er np.random
class FFNN:
    def __init__(self, x, y, batch_size=32, epoch=5000, lr=0.01, hidden_layer_size=16):
        self.input = np.array(x)
        self.target = np.array(y)

        self.input_dim = x.shape[1]
        self.output_dim = y.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = lr
        self.weight_hidden = np.random.randn(self.input_dim, self.hidden_layer_size)
        self.bias_hidden = np.random.randn(self.hidden_layer_size)
        self.weight_output = np.random.randn(self.hidden_layer_size, self.output_dim)
        self.bias_output = np.random.randn(self.output_dim)

    def forward_pass(self):
        self.hidden1 = self.input @ self.weight_hidden + self.bias_hidden
        self.activation1 = self.ReLU(self.hidden1)
        self.hidden2 = self.activation1 @ self.weight_output + self.bias_output
        output = self.hidden2
        return output

    def backward_pass(self):
        ###TBD###
        # forward pass
        # Mini batch gradient descent
        # Backpropagation
        raise NotImplementedError("Backpropagation not implemented yet.")

    def train(self):
        # TBD
        # Train loop
        # update weights
        raise NotImplementedError("Training loop not implemented yet.")

    def pred(self):
        # TBD
        raise NotImplementedError("Prediction method not implemented yet.")
