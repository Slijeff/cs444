"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1],
                                                        sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # TODO: You may set parameters for Adam optimizer here

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N):
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me
        return X.T @ de_dz, de_dz.T @ np.ones((de_dz.shape[0],)), de_dz @ W.T

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        xrelu = X.copy()
        xrelu[xrelu < 0] = 0
        return xrelu

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        xgrad = X.copy()
        xgrad[xgrad <= 0] = 0
        xgrad[xgrad > 0] = 1
        return xgrad

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-X))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid(X) * (1.0 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> float:
        # TODO implement this
        return np.mean((y - p) ** 2).astype(float)

    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return -2 * (y - p) / np.prod(p.shape)

    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        raise NotImplementedError("mse_sigmoid_grad not needed")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {
            'g(z)-0': X
        }
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        for i in range(1, self.num_layers):
            w_param_name = f'W{i}'
            b_param_name = f'b{i}'
            y = self.linear(self.params[w_param_name],
                            X, self.params[b_param_name])
            self.outputs[f'z-{i}'] = y
            y = self.relu(y)
            self.outputs[f'g(z)-{i}'] = y
            X = y

        y = self.linear(
            self.params[f'W{self.num_layers}'], X, self.params[f'b{self.num_layers}'])
        self.outputs[f'z-{self.num_layers}'] = y
        y = self.sigmoid(y)
        self.outputs[f'g(z)-{self.num_layers}'] = y
        return y

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        dedz = self.mse_grad(
            y,
            self.outputs[f'g(z)-{self.num_layers}']
        ) * self.sigmoid_grad(self.outputs[f'z-{self.num_layers}'])

        dedw, dedb, dedx = self.linear_grad(
            self.params[f'W{self.num_layers}'],
            self.outputs[f'g(z)-{self.num_layers - 1}'],
            np.array([]),
            dedz,
            0, 0
        )
        self.gradients[f'W{self.num_layers}'] = dedw
        self.gradients[f'b{self.num_layers}'] = dedb

        for layer in range(self.num_layers - 1, 0, -1):
            dedz = dedx * self.relu_grad(self.outputs[f'z-{layer}'])
            dedw, dedb, dedx = self.linear_grad(
                self.params[f'W{layer}'],
                self.outputs[f'g(z)-{layer - 1}'],
                np.array([]),
                dedz,
                0, 0
            )
            self.gradients[f'W{layer}'] = dedw
            self.gradients[f'b{layer}'] = dedb

        return self.mse(y, self.outputs[f'g(z)-{self.num_layers}'])

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        return

    def print_shapes(self) -> None:
        print("=====Parameter shapes: ======= ")
        for k, v in self.params.items():
            print(f'{k}: {v.shape}')
        print("=====Output shapes: =====")
        for k, v in self.outputs.items():
            print(f'{k}: {v.shape}')
        print("=====Gradient shapes: =====")
        for k, v in self.gradients.items():
            print(f'{k}: {v.shape}')
