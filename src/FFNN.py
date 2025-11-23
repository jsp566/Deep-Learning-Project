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
    
    def backward_pass(self, loss_grad):

        #loss_grad = self.loss_function.gradient(batch_y, y_pred)
        
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def train(self, batch_x, batch_y):

        y_pred = self.forward_pass(batch_x)

        loss = self.loss_function.compute(batch_y, y_pred)
        self.backward_pass(batch_y, y_pred)

        self.update_params()

    def update_params(self):
        for layer in self.layers:
            layer.update_params(self.optimizer)
            layer.zero_gradients()

    def train_loop(self):
        # for each epoch:
        # Mini batch gradient descent
        # train
        raise NotImplementedError("Training loop not implemented yet.")
    
    def train_loop(self, X, y, epochs=10, batch_size=32, shuffle=True, verbose=True):
        n_samples = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
                X = X[indices]
                y = y[indices]

            epoch_loss = 0
            n_batches = int(np.ceil(n_samples / batch_size))

            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                batch_x = X[start:end]
                batch_y = y[start:end]

                # Run training step (forward + backward + update)
                y_pred = self.forward_pass(batch_x)
                epoch_loss += self.loss_function.compute(batch_y, y_pred)
                self.backward_pass(batch_y, y_pred)

                for layer in self.layers:
                    layer.update_params(self.optimizer)
                    layer.zero_gradients()

            epoch_loss /= n_batches
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}")


        return loss_history

    def pred(self):
        # TBD
        raise NotImplementedError("Prediction method not implemented yet.")
