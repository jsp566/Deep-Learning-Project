import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function, weight_initializer, bias_initializer):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.weights = self.weight_initializer.initialize((input_size, output_size))
        self.biases = self.bias_initializer.initialize((output_size,))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function.forward(self.z)
        return self.a
    
    def backward(self, dvalues):
        dz = dvalues * self.activation_function.backward(self.z)
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0)
        dinputs = np.dot(dz, self.weights.T)
        return dinputs

    

