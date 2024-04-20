import torch
import torch.nn as nn
from einops import rearrange


class Conv3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False
    ):
        super().__init__()
        # self.normalization = nn.BatchNorm2d(out_channels)
        self.normalization = nn.GroupNorm(8, out_channels)
        self.preprocess = nn.Sequential(
            # retains original W & H dimension
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            self.normalization,
            nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            self.normalization,
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            self.normalization,
            nn.GELU()
        )
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        if self.residual:
            x = x + self.conv(x)
            return x / 1.414
        return self.conv(x)


class UnetDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.downsample_method = nn.MaxPool2d(2)
        self.layers = nn.Sequential(
            Conv3(in_channels, out_channels, True),
            self.downsample_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UnetUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.upsample_method = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            Conv3(in_channels, out_channels, True)
        )
        # self.upsample_method = nn.ConvTranspose2d(
        #     in_channels,
        #     out_channels,
        #     2, 2
        # )
        self.layers = nn.Sequential(
            self.upsample_method,
            Conv3(out_channels, out_channels, True),
            Conv3(out_channels, out_channels, True)
        )

    def forward(self,
                x: torch.Tensor,
                skip_layer: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip_layer), dim=1)
        return self.upsample_method(x)


# class Attention(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         embed_dim: int = 32,
#         num_heads: int = 4
#     ):
#         super().__init__()
#         # input should be (batch, w * h, channel)
#         self.attn = nn.MultiheadAttention(
#             embed_dim, num_heads, batch_first=True)

#         self.mapping = nn.Conv2d(input_dim, embed_dim, 1)
#         self.out_mapping = nn.Conv2d(embed_dim, input_dim, 1)

#     def forward(self, x: torch.Tensor):
#         w, h = x.shape[2:]
#         x = self.mapping(x)
#         x = rearrange(x, 'b c w h -> b (w h) c')
#         attn, _ = self.attn(x, x, x, need_weights=False)
#         x = rearrange(attn, 'b (w h) c -> b c w h', w=w, h=h)
#         out = self.out_mapping(x)
#         return out


class Attention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class TimeStepEmbedding(nn.Module):
    '''
    Embeds a single time step faction to a n-dim vector.
    e.g. At step 300 with a total of 1000 steps, the timestep input
    will be 300/1000
    '''

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: int):
        embed = 4 / (self.dim // 2 - 1)
        embed = torch.exp(
            torch.arange(self.dim // 2, device=time.device) *
            -embed
        )
        embed = time[:, None] * embed[None, :]
        embed = torch.cat((embed.sin(), embed.cos()), dim=1)
        return embed


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_features: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features

        # convert image to starting features
        self.init = Conv3(in_channels, n_features, True)

        # three downsample layers followed by attention
        self.down1 = UnetDown(n_features, n_features)
        self.down2 = UnetDown(n_features, 2 * n_features)
        self.down3 = UnetDown(2 * n_features, 4 * n_features)
        self.attndown1 = Attention(n_features)
        self.attndown2 = Attention(2 * n_features)
        self.attndown3 = Attention(4 * n_features)

        # bottleneck
        self.bottleneck_in = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )
        self.bottleneck_out = nn.Sequential(
            nn.ConvTranspose2d(
                4 * n_features,
                4 * n_features,
                4, 4
            ),
            nn.GroupNorm(8, 4 * n_features),
            nn.GELU()
        )
        # time embedding
        self.time1 = TimeStepEmbedding(4 * n_features)
        self.time2 = TimeStepEmbedding(2 * n_features)
        self.time3 = TimeStepEmbedding(n_features)

        # the same goes for upsampling layers
        self.up1 = UnetUp(2 * 4 * n_features, 2 * n_features)
        self.up2 = UnetUp(2 * 2 * n_features, n_features)
        self.up3 = UnetUp(2 * 1 * n_features, n_features)
        self.attnup1 = Attention(2 * n_features)
        self.attnup2 = Attention(n_features)
        self.attnup3 = Attention(n_features)

        # final output
        self.out = nn.Conv2d(2 * n_features, self.out_channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        x = self.init(x)

        d1 = self.down1(x)
        d1 = self.attndown1(d1)

        d2 = self.down2(d1)
        d2 = self.attndown2(d2)

        d3 = self.down3(d2)
        d3 = self.attndown3(d3)

        bneck = self.bottleneck_in(d3)
        temb1 = self.time1(t).view(-1, self.n_features * 4, 1, 1)
        bneck = self.bottleneck_out(bneck + temb1)

        temb2 = self.time2(t).view(-1, self.n_features * 2, 1, 1)
        u1 = self.up1(bneck, d3) + temb2
        u1 = self.attnup1(u1)

        temb3 = self.time3(t).view(-1, self.n_features, 1, 1)
        u2 = self.up2(u1, d2) + temb3
        u2 = self.attnup2(u2)

        u3 = self.up3(u2, d1)
        u3 = self.attnup3(u3)

        out = self.out(torch.cat((u3, x), dim=1))

        return out
