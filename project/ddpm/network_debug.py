from unet import Conv3, UnetDown, UnetUp, Attention, TimeStepEmbedding, Unet
from ddpm import DDPM
import torch

if __name__ == "__main__":
    c3 = Conv3(3, 16, True)
    x = torch.randn((4, 3, 32, 32))
    x = c3(x)
    assert x.shape == (4, 16, 32, 32)

    down = UnetDown(3, 16)
    x = torch.randn((4, 3, 32, 32))
    x = down(x)
    assert x.shape == (4, 16, 16, 16)

    up = UnetUp(2 * 16, 8)
    x = torch.randn((4, 16, 16, 16))
    x = up(x, x)
    assert x.shape == (4, 8, 32, 32)

    attn = Attention(3, 8, 2)
    x = torch.randn((4, 3, 32, 32))
    x = attn(x)
    assert x.shape == (4, 3, 32, 32)

    emb = TimeStepEmbedding(10)
    t = torch.Tensor([300 / 1000])
    t = emb(t)
    assert t.shape == (1, 10)

    unet = Unet(3, 3, n_features=32)
    x = torch.randn((4, 3, 32, 32))
    t = torch.Tensor([300 / 1000])
    out = unet(x, t)
    assert out.shape == x.shape

    ddpm = DDPM(Unet(3, 3, n_features=32), 1e-4, 0.002, 1000)
    x = torch.rand((2, 3, 32, 32))
    loss = ddpm(x)
    print(loss)
