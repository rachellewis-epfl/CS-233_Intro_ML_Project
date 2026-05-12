import numpy as np

class MLP:


    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation functions, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        if len(activations) != len(dimensions) - 1:
            raise ValueError(
                "Number of activations must be len(dimensions) - 1."
            )

        self.dimensions = dimensions
        self.activations = activations
        self.n_layers = len(dimensions) - 1

        self.weights = {}
        self.biases = {}

        # Xavier/Glorot initialization to overcome gradient vanishing
        # recommended in Lec 9 slide 75
        # algorithm source: https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/xavier-glorot-initialization
        for i in range(1, len(dimensions)):
            fan_in = dimensions[i - 1]
            fan_out = dimensions[i]
            limit = np.sqrt(6 / (fan_in + fan_out))

            # uniform dist [-limit, limit]
            self.weights[i] = np.random.uniform(
                low= -limit,
                high=limit,
                size=(fan_in, fan_out)
            )

            # make bias vector for each neuron in the layer i
            self.biases[i] = np.zeros((1, fan_out))

        self.learning_rate = None



    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # output of layer "0" is just the input vectors
        z = {0: x}
        # dictionary of weighted values before applying activation func
        a = {}

        for i in range(1, self.n_layers + 1):
            # Lect 9 slide 41: z_l = f (W_l * z_l-1) (where l is current layer)
            # and ( W_l * z_l-1 ) is the "activation" of the layer,
            # so z[i] = f ( a[i] )
            
            a[i] = z[i - 1] @ self.weights[i] + self.biases[i]
            z[i] = self.activations[i - 1].forward(a[i])

        return z, a


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        ### WRITE YOUR CODE HERE


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        ### WRITE YOUR CODE HERE


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        ### WRITE YOUR CODE HERE

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        ### WRITE YOUR CODE HERE
