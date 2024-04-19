from configs.cifarconfig import cifar_config
from utils import save_image_from_batch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dl = DataLoader(cifar_config.dataset, batch_size=16)
    for x, _ in dl:
        save_image_from_batch(x, "./data_visualization.png", 4)
        break
