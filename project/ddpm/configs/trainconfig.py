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

ANIME scale:
anime_scale.pth
unet_features = 512
attention head = 8
attention dim = 32
'''

@dataclass
class TrainConfig:
    # HYPERPARAMS
    num_epoch = 50
    device = "cuda"
    # checkpoint_path = "./checkpoints/anime_scale.pth"
    checkpoint_path = "./checkpoints/cifar_v2.pth"
    # checkpoint_path = "./checkpoints/mnist.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = None
    generate_every = 5
    generate_n_images = 8
    generate_output_path = "./outputs/cifar_progress/v2/1450-1500"
    gradient_accumulation = None
    batch_size = 128

    # NETWORK RELATED
    unet_features = 256
    criterion = nn.MSELoss()
    attention_head = 2
    attention_dim = 8

    # SCHEDULING
    beta_schedule = "linear"
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    
    seed = 444

    data: DataConfig = field(default_factory=lambda: cifar_config)

    optimizer: optim.Optimizer = optim.Adam
    lr = 2e-4


tc = TrainConfig()
