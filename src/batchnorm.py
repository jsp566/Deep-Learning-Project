import numpy as np

class BatchNorm:
    def __str__(self):
        return f"BatchNorm(num_features={self.num_features}, momentum={self.momentum}, epsilon={self.epsilon})"

    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))

        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, batch_inputs, training=True):
        if training:
            self.inputs = batch_inputs
            self.batch_mean = np.mean(batch_inputs, axis=0)
            self.batch_var = np.var(batch_inputs, axis=0)

            self.normalized_inputs = (batch_inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            out = self.gamma * self.normalized_inputs + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            normalized_inputs = (batch_inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * normalized_inputs + self.beta

        return out

    def backward(self, loss_grad, regularization=None):
        m = loss_grad.shape[0]

        dbeta = np.sum(loss_grad, axis=0)
        dgamma = np.sum(loss_grad * self.normalized_inputs, axis=0)

        dnormalized = loss_grad * self.gamma

        dvar = np.sum(dnormalized * (self.inputs - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon) ** (-1.5), axis=0)
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (self.inputs - self.batch_mean), axis=0)

        dinputs = dnormalized / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.inputs - self.batch_mean) / m + dmean / m

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dinputs
    
    def zero_gradients(self):
        self.dgamma.fill(0)
        self.dbeta.fill(0)

    def update_params(self, optimizer):
        optimizer.step(self.gamma, self.dgamma)
        optimizer.step(self.beta, self.dbeta)