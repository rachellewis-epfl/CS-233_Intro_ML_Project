import numpy as np


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """

        # there should be no weight attribute until fit() computes it
        self.w = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """

        # add bias term (a column of 1s)
        X = np.concatenate([np.ones((training_data.shape[0], 1)), training_data], axis=1)
        y = training_labels

        # closed-form solution
        # @ means matrix multiplication
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y

        # X times closed form solution is the predicted labels 
        pred_labels = X @ self.w

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        # add bias term
        X = np.concatenate([np.ones((test_data.shape[0], 1)), test_data], axis=1)
        
        # simply use self.w weight to predict labels
        pred_labels = X @ self.w

        return pred_labels
