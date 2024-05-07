from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from unet import Unet
from ddpm import DDPM
from configs.trainconfig import tc, TrainConfig
import torch
from utils import save_image_from_batch
import math
import random
from torch.utils.data import DataLoader

def interpolate(config):
    # torch.manual_seed(12138)
    ddpm = DDPM(
        model = Unet(dim=config.unet_features, channels=config.data.image_channels, dim_mults=(1,2,4,8,16), time_emb_dim=64),
        beta_schedule=config.beta_schedule,
        beta1=config.beta1,
        beta2=config.beta2,
        T=config.T,
        device=config.device
    ).to(config.device)

    ddpm.load_state_dict(torch.load(config.checkpoint_path,
                         map_location=torch.device(config.device)))

    dataloader = DataLoader(
        config.data.dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    n_images = 10
    ddpm.eval()
    ddpm.unet.eval()
    with torch.no_grad():
        for xs, _ in dataloader:
            img1, img2 = xs.to(config.device)
            res = ddpm.interpolate(img1, img2, n_images)
            save_image_from_batch(res, "./outputs/generate/interpolate.png", 2 + n_images)
            break


if __name__ == "__main__":
    interpolate(tc)