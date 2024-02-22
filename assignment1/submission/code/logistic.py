"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = np.array([])  # TODO: change this
        self.lr = lr
        self.decay = .65
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train_raw: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # training on RICE only, so scale lable to -1 to 1 instead of 0 to 1
        y_train = y_train_raw.copy()
        y_train[y_train == 0] = -1
        def get_acc(pred, y_test): return np.sum(
            y_test == pred) / len(y_test)
        n_samples, n_features = X_train.shape
        self.w = np.random.randn(1, n_features)
        for epoch in range(self.epochs):
            # print(
            #     f"epoch: {epoch}, error: {1 - get_acc(self.predict(X_train), y_train_raw)}")
            for i, sample in enumerate(X_train):
                self.w += self.lr * \
                    self.sigmoid(np.dot(self.w, sample.T) * -
                                 y_train[i]) * sample * y_train[i]
            self.lr *= 1 / (1 + self.decay * epoch)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        res = self.sigmoid(np.dot(self.w, X_test.T))
        res[res >= self.threshold] = 1
        res[res < self.threshold] = 0
        return res
