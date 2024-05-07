import torch
from unet_v2 import Unet
from torchview import draw_graph

if __name__ == '__main__':
    model = Unet(dim=64, channels=3, dim_mults=(1,2,4,8,16), time_emb_dim=32).to("cuda")
    # torch.onnx.export(model, (torch.randn((3, 3, 32, 32), device="cuda"), torch.tensor(3 / 1000).unsqueeze(0).to("cuda")), 'unet.onnx', input_names=['noise_image'], output_names=['predicted_noise'])
    model_graph = draw_graph(model, input_size=[(3, 3, 32, 32), (1,)], device='cuda', save_graph=True)