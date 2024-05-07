from torch.utils.data import Subset
from torchvision import transforms
from dataclasses import dataclass
from .dataconfig import DataConfig

import glob
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torch.utils.data import Dataset
from PIL import Image

from dataset.cats import Cat


@dataclass
class CatConfig(DataConfig):

    dataset: Subset = Cat(
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            Lambda(lambda x : (x * 2) - 1)
        ]),
    )


cat_config = CatConfig()