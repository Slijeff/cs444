from dataclasses import dataclass, field
from torch import optim
from configs.mnistconfig import mnist_config
from configs.dataconfig import DataConfig


@dataclass
class TrainConfig:
    num_epoch = 50
    device = "mps"
    checkpoint_path = "./checkpoints/ddpm_mnist_normalized.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = None
    generate_every = None
    generate_n_images = 16
    gradient_accumulation = None
    batch_size = 16
    unet_features = 128
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    seed = 123

    data: DataConfig = field(default_factory=lambda: mnist_config)

    optimizer: optim.Optimizer = optim.AdamW
    lr = 5e-5


tc = TrainConfig()
