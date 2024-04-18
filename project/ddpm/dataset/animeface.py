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


if __name__ == "__main__":
    af = AnimeFace(
        Compose([
            ToTensor(),
            Resize(64)
        ])
    )
    x, _ = af[3]
    print(x.shape)
