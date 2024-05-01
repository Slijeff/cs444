from dataclasses import dataclass, field
from torch import optim, nn
from configs.mnistconfig import mnist_config
from configs.cifarconfig import cifar_config
from configs.animeconfig import anime_config
from configs.dataconfig import DataConfig


'''
CIFAR and MNIST:
unet_features = 256
attention head = 2
attention dim = 8

ANIME:
unet_features = 256
attention head = 4
attention dim = 16
'''

@dataclass
class TrainConfig:
    num_epoch = 50
    device = "cpu"
    checkpoint_path = "./checkpoints/anime.pth"
    # checkpoint_path = "./checkpoints/cifar_v2.pth"
    # checkpoint_path = "./checkpoints/mnist.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = None
    generate_every = 5
    generate_n_images = 6
    generate_output_path = "./outputs/anime_progress/400-450"
    gradient_accumulation = None
    batch_size = 16
    unet_features = 256
    criterion = nn.MSELoss()
    
    beta_schedule = "linear"
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    
    seed = 444

    data: DataConfig = field(default_factory=lambda: anime_config)

    optimizer: optim.Optimizer = optim.AdamW
    lr = 1e-4


tc = TrainConfig()
