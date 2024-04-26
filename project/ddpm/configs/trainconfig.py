from dataclasses import dataclass, field
from torch import optim
from configs.mnistconfig import mnist_config
from configs.cifarconfig import cifar_config
from configs.dataconfig import DataConfig


@dataclass
class TrainConfig:
    num_epoch = 400
    device = "cuda"
    checkpoint_path = "./checkpoints/mnist.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = None
    generate_every = 20
    generate_n_images = 16
    generate_output_path = "./outputs/mnist_progress/500-900"
    gradient_accumulation = None
    batch_size = 64
    unet_features = 256
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    seed = 444

    data: DataConfig = field(default_factory=lambda: mnist_config)

    optimizer: optim.Optimizer = optim.AdamW
    lr = 5e-5


tc = TrainConfig()
