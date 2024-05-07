import glob
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset
from PIL import Image


class Cat(Dataset):
    def __init__(self, transform=None):
        self.img_path = "./data/cats/*"
        self.images = glob.glob(self.img_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i])
        if self.transform:
            img = self.transform(img)

        return img, "dummylabel"