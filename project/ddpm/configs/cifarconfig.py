from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from dataclasses import dataclass
from .dataconfig import DataConfig
from torchvision import transforms


@dataclass
class CIFARConfig(DataConfig):

    dataset: Subset = Subset(
        CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1)
            ]),
            download=True
        ),
        range(10000))


cifar_config = CIFARConfig()
# print(cifar_config.image_shape)
