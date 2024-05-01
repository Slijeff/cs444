from torchmetrics.image.fid import FrechetInceptionDistance
from ddpm import DDPM
from unet import Unet
from .configs.trainconfig import tc, TrainConfig
from torch.utils.data import DataLoader
import torch


def computeFID(
    test_model_path: str,
    n_images: int,
    fid_feat: int,
    config: TrainConfig,
    real_dataloader: DataLoader
):
    '''
    Computes the FID score for a given model and the real data
    '''
    fid = FrechetInceptionDistance(feature=fid_feat)
    ddpm = DDPM(
        model=Unet(
            config.data.image_channels,
            config.data.image_channels,
            n_features=config.unet_features
        ),
        beta_schedule=config.beta_schedule,
        beta1=config.beta1,
        beta2=config.beta2,
        T=config.T,
        device=config.device
    ).to(config.device)

    ddpm.load_state_dict(torch.load(
        test_model_path, map_location=torch.device(config.device)))

    ddpm.eval()
    with torch.no_grad():
        total_fake = 0
        while total_fake < n_images:
            samples = ddpm.generate(
                config.generate_n_images,
                (config.data.image_channels,
                 config.data.image_size,
                 config.data.image_size),
                config.device,
                None
            )

            fid.update(samples, real=False)

            total_fake += config.generate_n_images

    for idx, (x, _) in enumerate(real_dataloader):
        x = x.to(config.device)
        fid.update(x, real=True)
        if (idx + 1) * x.shape[0] > n_images:
            break

    print(fid.compute())


if __name__ == "__main__":
    computeFID(
        "./checkpoints/ddpm_anime.pth",
        16,
        2048,
        tc,
        DataLoader(
            tc.data.dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=2
        )
    )
