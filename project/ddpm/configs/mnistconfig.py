from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torchvision import transforms
from dataclasses import dataclass
from .dataconfig import DataConfig


@dataclass
class MNISTConfig(DataConfig):

    dataset: Subset = Subset(MNIST(
        root="./data",
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize(
                # (0.0981, ),
                # (0.2725, )
                (0.5, ),
                (0.5, )
            )
        ]),
        download=True
    ), range(1000))


mnist_config = MNISTConfig()
# print(mnist_config.image_mean)
# print(mnist_config.image_std)
# print(mnist_config.image_value_range)
