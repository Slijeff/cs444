from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
# from ema_pytorch import EMA


from unet import Unet
from ddpm import DDPM
from utils import save_image_from_batch, model_summary

from configs.trainconfig import TrainConfig, tc
import math

import os
import numpy as np

# torch.backends.cudnn.enabled = False
import gc
torch.cuda.empty_cache()
gc.collect()
def train(config: TrainConfig):
    ddpm = DDPM(
        model=Unet(config.data.image_channels,
                   config.data.image_channels, 
                   n_features=config.unet_features,
                   attn_head=config.attention_head, 
                   attn_dim=config.attention_dim),
        beta_schedule=config.beta_schedule,
        beta1=config.beta1, beta2=config.beta2, T=config.T,
        device=config.device,
        criterion=config.criterion
    ).to(config.device)

    model_summary(ddpm.unet)

    if config.checkpoint_path:
        try:
            ddpm.load_state_dict(torch.load(config.checkpoint_path))
            print(f"loaded checkpoint from {config.checkpoint_path}")
        except:
            print(f"training from scratch to {config.checkpoint_path}")

    dataloader = DataLoader(
        config.data.dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    optim = config.optimizer(
        ddpm.parameters(), lr=config.lr)
    for epoch in (t := trange(config.num_epoch)):
        epoch_loss = []
        ddpm.train()
        for idx, (x, _) in enumerate(tqdm(dataloader, leave=False)):
            x = x.to(config.device)
            loss = ddpm(x)
            if config.gradient_accumulation:
                loss /= config.gradient_accumulation
            loss.backward()
            epoch_loss.append(loss.item())
            t.set_description(f"Last 10it Loss: {np.mean(epoch_loss[-10:])}")
            if config.gradient_accumulation:
                if (idx + 1) % config.gradient_accumulation == 0 \
                        or idx + 1 == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 5.0)
                    optim.step()
                    optim.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 5.0)
                optim.step()
                optim.zero_grad()

        if config.generate_every and (epoch % config.generate_every == 0 or
                                      epoch == config.num_epoch - 1):
            ddpm.eval()
            if config.seed:
                torch.manual_seed(config.seed)
            with torch.no_grad():
                if config.use_ddim:
                    samples = ddpm.generate_ddim(
                        config.generate_n_images,
                        (config.data.image_channels,
                         config.data.image_size,
                         config.data.image_size),
                        config.device,
                        config.ddim_sampling_steps
                    )
                else:
                    samples = ddpm.generate(
                        config.generate_n_images,
                        (config.data.image_channels,
                         config.data.image_size,
                         config.data.image_size),
                        config.device
                    )
                samples = torch.cat((samples, x[:4]), dim=0)
                os.makedirs(config.generate_output_path, exist_ok=True)
                save_image_from_batch(
                    samples, os.path.join(
                        config.generate_output_path, f"train_sample_ep{epoch}.png"),
                    math.floor(math.sqrt(config.generate_n_images + 4)))

        if config.checkpoint_path:
            torch.save(ddpm.state_dict(), config.checkpoint_path)


if __name__ == "__main__":
    train(tc)
