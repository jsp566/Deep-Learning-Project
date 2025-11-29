import numpy as np

class Layer:
    def __str__(self):
        return f"Layer(input_size={self.input_size}, output_size={self.output_size}, weight_initializer={type(self.weight_initializer).__name__}, bias_initializer={type(self.bias_initializer).__name__})"

    def __init__(self, input_size, output_size, weight_initializer, bias_initializer):
        self.input_size = input_size
        self.output_size = output_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.weights = self.weight_initializer.initialize((input_size, output_size))
        self.biases = self.bias_initializer.initialize((output_size,))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, batch_inputs, training=True):
        self.inputs = batch_inputs
        self.z = np.dot(batch_inputs, self.weights) + self.biases
        return self.z
    
    def backward(self, loss_grad, regularization):
        m = loss_grad.shape[0]

        regularization_term = regularization.compute_gradient(self.weights)
        self.dweights = np.dot(self.inputs.T, loss_grad) / m + regularization_term
        self.dbiases = np.sum(loss_grad, axis=0) / m

        prev_loss_grad = np.dot(loss_grad, self.weights.T)
        return prev_loss_grad


    def zero_gradients(self):
        self.dweights.fill(0)
        self.dbiases.fill(0)

    def update_params(self, optimizer):
        optimizer.step(self.weights, self.dweights)
        optimizer.step(self.biases, self.dbiases)