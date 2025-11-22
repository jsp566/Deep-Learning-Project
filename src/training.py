import numpy as np


class Trainer:
    def __init__(self, model, loss_function, optimizer=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, X, y, epochs=10, batch_size=32, shuffle=True):
        num_samples = X.shape[0]
        history = {"loss": []}

        for epoch in range(epochs):
            if shuffle:
                idx = np.random.permutation(num_samples)
                X, y = X[idx], y[idx]

            epoch_loss = 0
            batches = num_samples // batch_size

            for i in range(batches):
                start = i * batch_size
                end = start + batch_size

                X_batch = X[start:end]
                y_batch = y[start:end]

                # Forward pass
                y_pred = self.model.forward(X_batch)

                # Compute loss
                loss = self.loss_function.compute(y_batch, y_pred)
                epoch_loss += loss

                # Backprop
                grad = self.loss_function.gradient(y_batch, y_pred)
                self.model.backward(grad)

                # Update parameters
                if self.optimizer is not None:
                    self.optimizer.update(self.model)

            epoch_loss /= batches
            history["loss"].append(epoch_loss)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        return history

    def predict(self, X):
        return self.model.forward(X)
