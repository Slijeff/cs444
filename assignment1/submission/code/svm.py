"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w: np.ndarray = np.array([])  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

        self.batch_size = 128
        self.decay = 0.65

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        gradient = np.zeros_like(self.w)  # n_class X n_features
        n_samples, n_features = X_train.shape
        for ith_sample in range(n_samples):
            score = np.dot(self.w, X_train[ith_sample])
            for ith_class in range(self.n_class):
                if ith_class != y_train[ith_sample] and score[y_train[ith_sample]] - score[ith_class] < 1:
                    gradient[ith_class] += X_train[ith_sample]
                    gradient[y_train[ith_sample]] -= X_train[ith_sample]
        # return gradient / n_samples
        return gradient

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        def get_acc(pred, y_test): return np.sum(
            y_test == pred) / len(y_test)
        n_samples, n_features = X_train.shape
        self.w = np.random.randn(self.n_class, n_features)
        for i in range(self.epochs):
            # print(
            #     f"epoch {i + 1} / {self.epochs}, error: {1 - get_acc(self.predict(X_train), y_train)}, lr: {self.lr}")
            start, end = 0, self.batch_size
            for batch in range(n_samples // self.batch_size):
                start, end = self.batch_size * \
                    batch, self.batch_size * (batch + 1)
                batch_x, batch_y = X_train[start:end], y_train[start:end]
                self.w = (1 - self.lr * self.reg_const / (n_samples // self.batch_size)
                          ) * self.w - self.lr * self.calc_gradient(batch_x, batch_y)
            # if i <= 15:
            #     self.lr *= 0.98
            self.lr *= 1 / (1 + self.decay * i)

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
        # TODO: implement me
        return np.argmax(np.dot(self.w, X_test.T).T, axis=1)
