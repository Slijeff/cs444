import torch
from torch import nn
import math
import einops
from einops import rearrange

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_channel)
        if up:
            self.conv1 = nn.Conv2d(2*in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv2d(out_channel, out_channel, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_channel)
        self.gnorm2 = nn.GroupNorm(8, out_channel)
        self.act  = nn.SiLU()
        
    def forward(self, x, timestep):
        # First Conv
        h = self.gnorm1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(timestep))
        time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
        h = h + time_emb
        h = self.gnorm2(self.act(self.conv2(h)))
        return self.transform(h)

class Unet(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1,2,4,8,16), time_emb_dim=32, device="cuda") -> None:
        super().__init__()

        print("Unet initializing...")
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), # (B,) => (B, 32)
            nn.Linear(time_emb_dim, time_emb_dim), # (B, 32) => (B, 32)
            nn.SiLU()
        )

        self.init_proj = nn.Conv2d(channels, dim * dim_mults[0], kernel_size=3, stride=1, padding=1)

        down_features = [dim*mult for mult in dim_mults]
        up_features =  down_features[::-1]

        print(up_features)

        self.d1 = SimpleBlock(down_features[0], down_features[1], time_emb_dim, up=False)
        print(f"Down {0} initialized")
        self.d2 = SimpleBlock(down_features[1], down_features[2], time_emb_dim, up=False)
        print(f"Down {1} initialized")
        self.d3 = SimpleBlock(down_features[2], down_features[3], time_emb_dim, up=False)
        print(f"Down {2} initialized")
        self.d4 = SimpleBlock(down_features[3], down_features[4], time_emb_dim, up=False)
        print(f"Down {3} initialized")
        

        self.u1 = SimpleBlock(up_features[0], up_features[1], time_emb_dim, up=True)
        print(f"Up {0} initialized")
        self.u2 = SimpleBlock(up_features[1], up_features[2], time_emb_dim, up=True)
        print(f"Up {1} initialized")
        self.u3 = SimpleBlock(up_features[2], up_features[3], time_emb_dim, up=True)
        print(f"Up {2} initialized")
        self.u4 = SimpleBlock(up_features[3], up_features[4], time_emb_dim, up=True)
        print(f"Up {3} initialized")


        self.last = nn.Conv2d(up_features[-1], channels, 1).to(device)

        print("Unet initialized")

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.init_proj(x) 

        r1 = self.d1(x, t)
        r2 = self.d2(r1, t)
        r3 = self.d3(r2, t)
        r4 = self.d4(r3, t)

        x = self.u1(torch.cat((r4, r4), dim=1), t)
        x = self.u2(torch.cat((x, r3), dim=1), t)
        x = self.u3(torch.cat((x, r2), dim=1),t)
        x = self.u4(torch.cat((x, r1), dim=1),t)

        return self.last(x)


        
