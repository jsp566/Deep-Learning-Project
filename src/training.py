import numpy as np


class Trainer:
    def __init__(self, model, loss_function, optimizer=None, logger=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger

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
                y_pred = self.model.forward_pass(X_batch)

                # Compute loss
                loss = self.loss_function.compute(y_batch, y_pred)
                epoch_loss += loss

                # Backprop
                grad = self.loss_function.gradient(y_batch, y_pred)
                self.model.backward_pass(grad, y_pred)

                # Update parameters
                if self.optimizer is not None:
                    self.optimizer.update(self.model)

            epoch_loss /= batches
            history["loss"].append(epoch_loss)
            # Log metrics
            if self.logger is not None:
                self.logger.log_metrics(epoch, y, self.model.forward_pass(X))

            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        if self.logger is not None:
            self.logger.finish()
        return history

    def predict(self, X):
        return self.model.forward_pass(X)
