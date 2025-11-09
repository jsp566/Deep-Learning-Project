import numpy as np


class Initializer:
    def initialize(self, shape):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class NormalInitializer(Initializer):
    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stddev = stddev

    def initialize(self, shape):
        return np.random.randn(*shape) * self.stddev + self.mean
    
class ConstantInitializer(Initializer):
    def __init__(self, value=1):
        self.value = value

    def initialize(self, shape):
        return np.ones(shape) * self.value
    
class HeInitializer(Initializer):
    def __init__(self, a = 1.0):
        self.a = a
    
    def initialize(self, shape):
        stddev = np.sqrt(2. * self.a / shape[0])
        return np.random.randn(*shape) * stddev
    
class GlorotInitializer(Initializer):
    def __init__(self, a = 1.0):
        self.a = a
    
    def initialize(self, shape):
        stddev = np.sqrt(2. * self.a / (shape[0] + shape[1]))
        return np.random.randn(*shape) * stddev