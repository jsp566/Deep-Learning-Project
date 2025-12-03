import numpy as np
import matplotlib.pyplot as plt



class LossFunction:
    def compute(self, y_true, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def plot_loss_curve(self, loss_history):
        """
        Input:
            loss_history: list of floats, logged loss per epoch
        """
        plt.figure(figsize=(7, 4))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.show()



class Accuracy(LossFunction):
    def compute(self, y_true, y_pred):
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        pred_labels = np.argmax(y_pred, axis=1)
        correct = np.sum(y_true == pred_labels)
        return correct / len(y_true)
        


class MSELoss(LossFunction):
    def compute(self, y_true, y_pred):
        return 1 / len(y_true) * np.sum((y_true - y_pred) ** 2)
    
    def gradient(self, y_true, y_pred):
        return -2 / len(y_true) * (y_true - y_pred)


class CrossEntropyLoss(LossFunction):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute(self, y_true, y_pred):

        if self.label_smoothing > 0.0:
            num_classes = y_true.shape[1]
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        y_pred = self._softmax(y_pred)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)
    
    def gradient(self, y_true, y_pred):

        if self.label_smoothing > 0.0:
            num_classes = y_true.shape[1]
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        y_pred = self._softmax(y_pred)
        # Adding a small epsilon to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        grad = (y_pred - y_true) / len(y_true)


        return grad
    
class Metrics:
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        pred_labels = np.argmax(y_pred, axis=1)

        num_classes = y_pred.shape[1]
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for t, p in zip(y_true, pred_labels):
            cm[t, p] += 1

        return cm

    @staticmethod
    def plot_confusion_matrix(cm, class_names=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()

        num_classes = cm.shape[0]
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names if class_names else tick_marks)
        plt.yticks(tick_marks, class_names if class_names else tick_marks)

        # Text labels added later (see fix below)
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

