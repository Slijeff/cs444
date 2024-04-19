import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def save_image_from_batch(x: torch.Tensor, path: str, nrow: int):
    grid = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
