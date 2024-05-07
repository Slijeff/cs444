from dataclasses import dataclass, field
from torch import optim, nn
from configs.mnistconfig import mnist_config
from configs.cifarconfig import cifar_config
from configs.animeconfig import anime_config
from configs.catconfig import cat_config
from configs.dataconfig import DataConfig
from lamb import Lamb

'''
CIFAR v2 and MNIST:
unet_features = 256
attention head = 2
attention dim = 8

MNIST scale
unet_features = 368
attention head = 2
attention dim = 8

MNIST small
unet_features = 128

MNIST v5
unet_features = 240
batch_size = 32

Cat
unet_features = 240
batch_size = 32

Anime v4
unet_features = 128

Anime v5
unet_features = 128

CIFAR v3:
unet_features = 256

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
    '''
    To train:
        1. set the checkpoint_path to a new path to save new weights
        2. adjust hyperparameters
        3. If use_ddpm = True, it would sample using ddpm during training, otherwise DDIM
        4. change the data field with corresponding data config (...lambda: your_config)
        5. run `python train.py`
    
    To inference:
        1. set the checkpoint_path to a previously saved weights
        2. change the data field with corresponding data config (...lambda: your_config), because
            the source data is used for determining image shape info
        3. If use_ddpm = True, it would sample using both DDIM and DDPM, otherwise DDIM only
        4. run `python generate.py`, the image save path is ./outputs/generate/
    '''
    
    # HYPERPARAMS
    num_epoch = 100
    device = "cuda"
    # checkpoint_path = "./checkpoints/anime_scale.pth"
    # checkpoint_path = "./checkpoints/cifar_v4.pth"
    # checkpoint_path = "./checkpoints/cifar_v3.pth"
    # checkpoint_path = "./checkpoints/cifar_v2.pth"
    # checkpoint_path = "./checkpoints/mnist.pth"
    # checkpoint_path = "./checkpoints/mnist_convUp.pth" # uses convTranspose to upsample
    # checkpoint_path = "./checkpoints/mnist_scale.pth"
    # checkpoint_path = "./checkpoints/mnist_small.pth"
    # checkpoint_path = "./checkpoints/mnist_v4.pth"
    # checkpoint_path = "./checkpoints/mnist_v5.pth"
    # checkpoint_path = "./checkpoints/ddpm_anime.pth"
    # checkpoint_path = "./checkpoints/anime_v5.pth"
    # checkpoint_path = "./checkpoints/cat_v5.pth"
    checkpoint_path = "./checkpoints/cat.pth"
    # checkpoint_path = None
    generate_every = 2
    generate_n_images = 16
    generate_output_path = "./outputs/anime_v5_progress/88-188"
    gradient_accumulation = None
    batch_size = 32
    ddim_sampling_steps = 45
    use_ddpm = True

    # NETWORK RELATED
    unet_features = 240
    criterion = nn.MSELoss()

    # SCHEDULING
    beta_schedule = "linear"
    beta1 = 1e-4
    beta2 = 0.02
    T = 1000
    
    seed = 444

    data: DataConfig = field(default_factory=lambda: cat_config)

    optimizer: optim.Optimizer = optim.Adam
    lr = 1e-5


tc = TrainConfig()
