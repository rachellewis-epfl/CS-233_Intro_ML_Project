import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        predicted_values = [self._predict_single_point(point) for point in test_data]
        return np.array(predicted_values)

    def _find_distances(self, point):
        # sqrt not needed since just the closest points need to be found
        return np.sum((self.training_data - point) ** 2, axis=1)

    def _predict_single_point(self, point):
        distances = self._find_distances(point)
        knn_idx = np.argsort(distances)[:self.k]
        knn_labels = self.training_labels[knn_idx]

        if self.task_kind == "regression":
            return np.mean(knn_labels)

        elif self.task_kind == "classification":
            knn_labels = knn_labels.astype(int)
            labels, counts = np.unique(knn_labels, return_counts=True)
            mode = np.max(counts)
            corresponding_labels = labels[counts == mode]

            if len(corresponding_labels) == 1:
                return corresponding_labels[0]

            else:
                best_label = None
                best_dist = None
                for label in corresponding_labels:
                    avg_dist = np.mean(distances[knn_idx][knn_labels == label])
                    if best_dist is None or avg_dist < best_dist:
                        best_dist = avg_dist
                        best_label = label
                return best_label
