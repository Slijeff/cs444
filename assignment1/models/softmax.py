"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = np.array([])  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = 128

    def softmax(self, x) -> np.ndarray:
        """
        Softmax a matrix along the rows
        """
        assert x.shape[1] == self.n_class
        mx = np.max(x, axis=-1, keepdims=True)
        numer = np.exp(x - mx)
        return numer / np.sum(numer, axis=-1, keepdims=True)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        n_samples, n_features = X_train.shape
        gradient = np.zeros_like(self.w)
        probs = self.softmax(np.dot(self.w, X_train.T).T)
        for i in range(n_samples):
            for c in range(self.n_class):
                gradient[c] += probs[i][c] * \
                    X_train[i] if c != y_train[i] else (
                        probs[i][c] - 1) * X_train[i]
        return gradient + self.reg_const * self.w

    def train(self, X_train_raw: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        def get_acc(pred, y_test): return np.sum(y_test == pred) / len(y_test)
        X_train = (X_train_raw - np.mean(X_train_raw)) / np.std(X_train_raw)
        n_samples, n_features = X_train.shape
        self.w = np.random.randn(self.n_class, n_features)
        for epoch in range(self.epochs):
            print(
                f"epoch: {epoch + 1} / {self.epochs}, error: {1 - get_acc(self.predict(X_train), y_train)}")
            for batch in range(n_samples // self.batch_size):
                start, end = self.batch_size * \
                    batch, self.batch_size * (batch + 1)
                batch_x, batch_y = X_train[start:end], y_train[start:end]
                self.w -= self.lr * self.calc_gradient(batch_x, batch_y)

        return

    def predict(self, X_test_raw: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        X_test = (X_test_raw - np.mean(X_test_raw)) / np.std(X_test_raw)
        return np.argmax(np.dot(self.w, X_test.T).T, axis=1)
