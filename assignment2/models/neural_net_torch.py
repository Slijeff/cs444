import torch
from torch import nn


class TorchNeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int = 4,
        hidden_size: int = 256
    ):
        super().__init__()
        self.float()
        self.layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for layer in range(num_layers - 1):
            if layer != num_layers - 2:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(hidden_size, 3))
        self.layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(self.layers)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
