from dataclasses import dataclass, field
from torch import optim
# from configs.mnistconfig import mnist_config
from configs.cifarconfig import cifar_config
from configs.dataconfig import DataConfig


@dataclass
class TrainConfig:
    num_epoch = 300
    device = "cuda"
    checkpoint_path = "./checkpoints/cifar_v2.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = None
    generate_every = 50
    generate_n_images = 16
    generate_output_path = "./outputs/cifar_progress/v2/500-800/"
    gradient_accumulation = None
    batch_size = 64
    unet_features = 256
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    seed = None

    data: DataConfig = field(default_factory=lambda: cifar_config)

    optimizer: optim.Optimizer = optim.AdamW
    lr = 5e-5


tc = TrainConfig()
