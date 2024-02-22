"""Perceptron model."""

import numpy as np
from tqdm.notebook import trange


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        np.random.seed(441)
        self.w = np.array([])
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay = 0.65

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        n_samples, n_features = X_train.shape
        self.w = np.random.randn(self.n_class, n_features)
        with trange(self.epochs, leave=False) as pbar:
            for epoch in pbar:
                n_incorrect_samples = 0
                for i, sample in enumerate(X_train):
                    scores = np.dot(self.w, sample)
                    for c in range(self.n_class):
                        if c != y_train[i] and scores[c] > scores[y_train[i]]:
                            self.w[y_train[i]] += self.lr * X_train[i]
                            self.w[c] -= self.lr * X_train[i]
                            n_incorrect_samples += 1
                # weight decay (hugely improves accuracy)
                self.lr *= 1 / (1 + self.decay * epoch)
                pbar.set_postfix_str(
                    f"error: {n_incorrect_samples/n_samples}, lr: {self.lr}", refresh=True)

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
        return np.argmax(np.dot(self.w, X_test.T).T, axis=1)
