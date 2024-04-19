from dataclasses import dataclass, field
from torch import optim
# from configs.mnistconfig import mnist_config
from configs.cifarconfig import cifar_config
from configs.dataconfig import DataConfig


@dataclass
class TrainConfig:
    num_epoch = 10
    device = "mps"
    checkpoint_path = "./checkpoints/cifar.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = None
    generate_every = 10
    generate_n_images = 8
    generate_output_path = "./outputs/cifar_progress/190-200/"
    gradient_accumulation = 2
    batch_size = 32
    unet_features = 128
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    seed = 123

    data: DataConfig = field(default_factory=lambda: cifar_config)

    optimizer: optim.Optimizer = optim.AdamW
    lr = 1e-4


tc = TrainConfig()
