import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label

# filled in by Rachel Lewis

def m_mult(m1, m2):
    # matrix multiplication
    return np.dot(m1, m2)


def m_T(m):
    # matrix transpose
    return np.transpose(m)


def m_exp(m):
    # elementwise exponential
    return np.exp(m)


class LogisticRegression(object):
    """
    Multi-class logistic regression classifier.
    """

    def __init__(self, lr=0.0001, max_iters=5000, tol=1e-5):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.W = None           # weights
        self.C = None           # num of classes


    # from softmax formula in Lec4 slide 47
    def softmax(self, scores):
        """
        scores: shape (N, C)
        returns probabilities of shape (N, C)
        """
        # prevent overflow from large exp() nums: subtract row-wise max
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)

        # normalization to make probabilities over the row
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    
    # from Lec 4 "Multi-class logistic regression"
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): class labels of shape (N,)
        Returns:
            pred_labels (np.array): predicted labels of shape (N,)
        """
        #add bias (account for w_0)
        X = np.hstack([np.ones((training_data.shape[0], 1)), training_data])
        N, D = X.shape
        self.C = get_n_classes(training_labels)

        # one-hot labels, shape (N, C)
        Y = label_to_onehot(training_labels, self.C)

        # random initialization of W
        self.W = 0.01 * np.random.randn(D, self.C)

        for i in range(self.max_iters):
            # Gradient descent process: Lec 4 slide 49

            # X*W aka (w_k^T * x) in lecture softmax func
            scores = m_mult(X, self.W)
            Y_hat = self.softmax(scores)

            if (i % 50 == 0):
                eps = 1e-12
                loss = -np.sum(Y * np.log(Y_hat + eps)) / N
                print(f'loss: {loss}')

            # gradient from lec 4 slide 51
            # ∇_W R(W) = sum_i x_i (y_hat_i - y_i)^T
            # vectorized: X^T (Y_hat - Y)
            grad = m_mult( m_T(X), (Y_hat - Y) )
            # normalize so that the gradient doesn't scale w bigger data sets
            grad = grad / N 

            W_prev = self.W.copy()
            self.W = self.W - self.lr * grad

            fro_norm = np.linalg.norm(W_prev - self.W, ord='fro')
            if fro_norm < self.tol:
                break

        # return predictions on training data
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        #add bias (account for w_0)
        X = np.hstack([np.ones((test_data.shape[0], 1)), test_data])

        # class scores
        scores = m_mult(X, self.W)

        # probabilities
        Y_hat = self.softmax(scores)

        # pick class with highest probability
        pred_labels = onehot_to_label(Y_hat)
        return pred_labels