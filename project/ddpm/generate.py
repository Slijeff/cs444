from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from unet import Unet
from ddpm import DDPM
from configs.trainconfig import tc, TrainConfig
import torch


def generate(config: TrainConfig, progress: bool = True):
    # torch.manual_seed(129)
    ddpm = DDPM(
        model=Unet(
            config.data.image_channels,
            config.data.image_channels,
            n_features=config.unet_features
        ),
        beta1=config.beta1,
        beta2=config.beta2,
        T=config.T,
        device=config.device
    ).to(config.device)

    ddpm.load_state_dict(torch.load(config.checkpoint_path))

    prog = []

    def save_progression_hook(i: int, x: torch.Tensor):
        if i % 100 == 0 or i == config.T - 1:
            prog.append(x.cpu())

    ddpm.eval()
    with torch.no_grad():
        samples = ddpm.generate(
            config.generate_n_images,
            (config.data.image_channels,
             config.data.image_size,
             config.data.image_size),
            config.device,
            save_progression_hook if progress else None
        )
        grid = make_grid(samples, nrow=4, normalize=True, value_range=(0, 1))
        # print(samples.shape)
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis("off")
        plt.savefig("./outputs/generate/plt.jpg", bbox_inches="tight")

        if progress:
            prog.append(samples.cpu())
            prog = torch.cat(prog)
            # print(prog.shape)
            grid = make_grid(prog,
                             nrow=config.generate_n_images, normalize=True,
                             value_range=(0, 1))
            grid = grid.permute(1, 2, 0)
            plt.imshow(grid)
            plt.axis("off")
            plt.savefig("./outputs/generate/prog.jpg", bbox_inches="tight")


if __name__ == "__main__":
    generate(tc, True)
