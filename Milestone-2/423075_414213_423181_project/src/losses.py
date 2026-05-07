import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        # as define in Lec 2 slide 42
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        # simple deriv of the MS loss func
        return 2 * (y_pred - y_true)
    
class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        # from Lec 4 slide 48
        return -np.sum( y_true * np.log(y_pred) )

    @staticmethod
    def gradient(y_true, y_pred):
        # derivative of the above function wrt y_pred
        # not a sum because we are making gradient vector
        return - y_true / y_pred
