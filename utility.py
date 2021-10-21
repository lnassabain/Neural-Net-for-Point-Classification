import numpy as np
from scipy.special import softmax # use built-in function to avoid numerical instability

class Utility:
    @staticmethod
    def identity(Z):
        return Z,1
    
    @staticmethod
    def tanh(Z):
        """
        Z : non activated outputs
        Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
        """
        A = np.empty(Z.shape)
        A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
        df = 1-A**2
        return A,df
    
    @staticmethod
    def sigmoid(Z):
        A = np.empty(Z.shape)
        A = 1.0 / (1 + np.exp(-Z))
        df = A * (1 - A)
        return A,df
    
    @staticmethod
    def relu(Z):
        A = np.empty(Z.shape)
        A = np.maximum(0,Z)
        df = (Z > 0).astype(int)
        return A,df
    
    @staticmethod
    def softmax(Z):
        return softmax(Z, axis=0) # from scipy.special
    
    @staticmethod
    def cross_entropy_cost(y_hat, y):
        n  = y_hat.shape[1]
        ce = -np.sum(y*np.log(y_hat+1e-9))/n
        return ce
    
    """
    Explication graphique du MSE:
    https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f
    """
    @staticmethod
    def MSE_cost(y_hat, y):
        mse = np.square(np.subtract(y_hat, y)).mean()
        return mse