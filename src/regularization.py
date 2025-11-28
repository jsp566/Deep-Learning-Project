import numpy as np

class Regularization:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def compute_penalty(self, model):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def compute_gradient(self, weights):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class L1Regularization(Regularization):
    def compute_penalty(self, model):
        penalty = sum(np.sum(np.abs(layer.weights)) for layer in model.layers if layer.__class__.__name__ == "Layer")
                
        return self.lambda_/2 * penalty
    
    def compute_gradient(self, weights):
        return self.lambda_ * np.sign(weights)
    
class L2Regularization(Regularization):
    def compute_penalty(self, model):
        
        penalty = sum(np.sum(layer.weights ** 2) for layer in model.layers if layer.__class__.__name__ == "Layer")
        return self.lambda_/2 * penalty
    
    def compute_gradient(self, weights):
        return self.lambda_ * weights
