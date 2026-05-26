import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        # as defined in Lec 9 slide 24
        return 1/(1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        # easily found by taking derivative of forward function wrt z
        return (np.exp(-z)) / ( (1 + np.exp(-z))**2 )

class ReLU:
    @staticmethod
    def forward(z):
        # as defined in Lec 9 slide 24
        return np.maximum(z, 0)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(float)
        # basically, if z > 0 , that means on the forward ReLU passed the value
        # aka the gradient is 1
        # but if z == 0, that means ReLu didn't activate on the forward, 
        # so gradient 0

class Linear:
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def gradient(x):
        return np.ones_like(x)
