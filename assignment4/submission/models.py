import torch
from torch import nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.layers = nn.Sequential(
          nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1),
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.layers(x)
        ##########       END      ##########

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.layers = nn.Sequential(
          nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
          nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
          nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(negative_slope=0.2),
          nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2),
          nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
          nn.Tanh()
        )
        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.layers(x)
        ##########       END      ##########

        return x
