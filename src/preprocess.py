import numpy as np


class DataPreprocessor:

    def transform(self, X):
        raise NotImplementedError("This method should be overridden by subclasses.")


class ScaleToUnit(DataPreprocessor):
    """
    Scales data to [0,1] range.
    """
    def transform(self, X):
        return X / 255.0


class Standardize(DataPreprocessor):
    """
    Standardizes data to zero mean and unit variance.
    Can be applied per channel
    """
    def transform(self, X, mean=None, std=None):
        """
        If mean/std are provided, uses them; else computes from X.
        """
        if mean is None:
            mean = np.mean(X, axis=(0, 1, 2), keepdims=True) if X.ndim == 4 else np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=(0, 1, 2), keepdims=True) if X.ndim == 4 else np.std(X, axis=0)
        
        std = np.where(std == 0, 1e-15, std)
        return (X - mean) / std


class PerChannelStandardize(DataPreprocessor):
    """
    Standardizes each channel separately using provided mean and std.
    """
    def transform(self, X, mean, std):
        mean = np.array(mean).reshape(1, 1, 1, -1)
        std = np.array(std).reshape(1, 1, 1, -1)
        std = np.where(std == 0, 1e-15, std)
        return (X - mean) / std


class ZScoreNormalize(DataPreprocessor):
    def transform(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1e-15, std)
        return (X - mean) / std


