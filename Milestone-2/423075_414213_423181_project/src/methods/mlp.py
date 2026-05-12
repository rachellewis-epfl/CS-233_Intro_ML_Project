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
        
        z, _ = self.feed_forward(x)
        return z[self.n_layers]

    # Lect 9 slide 52 back prop algorithm
    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        ex = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.

        :return: tuple (dw, delta)
                 dw[i] = gradient of weights at layer i
                 delta[i] = error signal at layer i
                
        """

        batch_size = y_true.shape[0]
        dw = {}
        delta = {}

        L = self.n_layers
        y_pred = z[L]

        # Get output layer delta
        # Lec 9 slide 48: delta_k_l = d_l_i / d_a_k_l
        # aka delta_i = derivative of loss func "l" wrt a certain activation "a_i"

        # f( a[L] ) = y_pred where f is activation func
        # so delta[L] = deriv(loss wrt y_pred) * deriv(y_pred wrt a[L])
        delta[L] = loss.gradient(y_true, y_pred) * self.activations[L - 1].gradient(a[L])


        for i in range(L, 0, -1):
            # average over gradients of all samples
            dw[i] = (z[i - 1].T @ delta[i]) / batch_size

            if i > 1:
                delta[i - 1] = (
                    delta[i] @ self.weights[i].T
                ) * self.activations[i - 2].gradient(a[i - 1])

        return dw, delta


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.weights[index] -= self.learning_rate * dw
        self.biases[index] -= self.learning_rate * np.mean(
            delta,
            axis=0,
            keepdims=True
        )


    # train using mini-batch SGD (Lec 9 slide 60)
    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        self.learning_rate = learning_rate
        n_cases = x.shape[0]
        history = []

        for epoch in range(epochs):
            indices = np.arange(n_cases)
            np.random.shuffle(indices)

            x_shuffled = x[indices]
            y_shuffled = y_true[indices]

            for start in range(0, n_cases, batch_size):
                end = start + batch_size

                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                z, a = self.feed_forward(x_batch)
                dw, delta = self.back_prop(z, a, y_batch, loss)

                for i in range(1, self.n_layers + 1):
                    self.update_w_b(i, dw[i], delta[i])

            y_pred = self.predict(x)
            epoch_loss = loss.loss(y_true, y_pred)
            history.append(epoch_loss)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.6f}")

        return history
