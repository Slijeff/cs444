from torch.utils.data import Subset
from torchvision import transforms
from dataclasses import dataclass
from .dataconfig import DataConfig

import glob
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torch.utils.data import Dataset
from PIL import Image

from dataset.animeface import AnimeFace


@dataclass
class AnimeConfig(DataConfig):

    dataset: Subset = Subset(AnimeFace(
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64, antialias=True),
            Lambda(lambda x : (x * 2) - 1)
        ]),
    ), range(10000))


anime_config = AnimeConfig()
# print(animeconfig.image_size)
# print(animeconfig.image_mean)
# print(animeconfig.image_std)
# print(animeconfig.image_value_range)
