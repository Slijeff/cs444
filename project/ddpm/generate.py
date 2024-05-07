from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from unet import Unet
from ddpm import DDPM
from configs.trainconfig import tc, TrainConfig
import torch
from utils import save_image_from_batch
import math
import random


def generate(config: TrainConfig, progress: bool = True):
    # torch.manual_seed(random.randint(100, 500))
    torch.manual_seed(123)
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

    prog = []

    def save_progression_hook(i: int, x: torch.Tensor):
        # if config.use_ddim:
        #     if i % 10 == 0:
        #         prog.append(x.cpu())
        # else:
        #     if i % 100 == 0 or config.T - 1 == i:
        #         prog.append(x.cpu())
        if i % 100 == 0:
            prog.append(x.cpu())

    ddpm.eval()
    ddpm.unet.eval()
    with torch.no_grad():
        if config.use_ddpm:
            samples = ddpm.generate(
                config.generate_n_images,
                (config.data.image_channels,
                 config.data.image_size,
                 config.data.image_size),
                config.device,
                save_progression_hook if progress else None
            )
            save_image_from_batch(samples, "./outputs/generate/plt_ddpm.jpg",
                              math.floor(math.sqrt(config.generate_n_images)))
            
        samples = ddpm.generate_ddim(
            config.generate_n_images,
            (config.data.image_channels,
             config.data.image_size,
             config.data.image_size),
            config.device,
            config.ddim_sampling_steps,
            save_progression_hook if progress else None
        )
        save_image_from_batch(samples, "./outputs/generate/plt.jpg",
                              math.floor(math.sqrt(config.generate_n_images)))
        if progress:
            prog.append(samples.cpu())
            prog = torch.cat(prog)
            # print(prog.shape)
            grid = make_grid(prog,
                             nrow=config.generate_n_images, normalize=True,
                             value_range=(-1, 1))
            grid = grid.permute(1, 2, 0)
            plt.imshow(grid)
            plt.axis("off")
            plt.savefig("./outputs/generate/prog.jpg", bbox_inches="tight")


if __name__ == "__main__":
    generate(tc, True)
