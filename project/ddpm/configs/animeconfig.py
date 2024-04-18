from torch.utils.data import Subset
from torchvision import transforms
from dataclasses import dataclass
from dataconfig import DataConfig

import glob
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset
from PIL import Image


class AnimeFace(Dataset):
    def __init__(self, transform=None):
        self.img_path = "./data/animefaces256cleaner/*"
        self.images = glob.glob(self.img_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i])
        if self.transform:
            img = self.transform(img)

        return img, "dummylabel"


@dataclass
class AnimeConfig(DataConfig):

    dataset: Subset = Subset(AnimeFace(
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.Normalize(
                (0.7051, 0.6199, 0.6131),
                (0.2667, 0.2694, 0.2587)
            )
        ]),
    ), range(1000))


animeconfig = AnimeConfig()
# print(animeconfig.image_size)
# print(animeconfig.image_mean)
# print(animeconfig.image_std)
# print(animeconfig.image_value_range)
