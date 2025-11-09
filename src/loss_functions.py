import numpy as np


class LossFunction:
    def compute(self, y_true, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses.")


class MSELoss(LossFunction):
    def compute(self, y_true, y_pred):
        return 1 / len(y_true) * np.sum((y_true - y_pred) ** 2)


class CrossEntropyLoss(LossFunction):
    def compute(self, y_true, y_pred):
        # Adding a small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)
