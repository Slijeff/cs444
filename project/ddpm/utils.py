import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def save_image_from_batch(x: torch.Tensor, path: str, nrow: int):
    grid = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")


def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [
        layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = i.bias is not None
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel(
            ) + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")
