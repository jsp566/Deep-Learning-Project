import numpy as np

class Layer:
    def __init__(self, input_size, output_size, weight_initializer, bias_initializer):
        self.input_size = input_size
        self.output_size = output_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.weights = self.weight_initializer.initialize((input_size, output_size))
        self.biases = self.bias_initializer.initialize((output_size,))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, batch_inputs):
        self.inputs = batch_inputs
        self.z = np.dot(batch_inputs, self.weights) + self.biases
        return self.z
    
    def backward(self, loss_grad):
        m = loss_grad.shape[0]
        delta = loss_grad

        self.dweights += np.dot(self.inputs.T, delta) / m
        self.dbiases += np.sum(delta, axis=0) / m

        prev_loss_grad = np.dot(delta, self.weights.T)
        return prev_loss_grad


    def zero_gradients(self):
        self.dweights.fill(0)
        self.dbiases.fill(0)

    def update_params(self, optimizer):
        self.weights = optimizer.step(self.weights, self.dweights)
        self.biases = optimizer.step(self.biases, self.dbiases)
