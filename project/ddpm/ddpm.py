import torch
from torch import nn
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta_schedule: str,
        beta1: float,
        beta2: float,
        T: int,
        device: str,
        criterion: nn.Module = nn.MSELoss(),
    ):
        super().__init__()

        self.unet = model
        self.T = T
        self.criterion = criterion
        self.precomp = {k: v.to(device) for k, v in
                        self.ddpm_schedules(beta1, beta2, T, beta_schedule).items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ts = torch.randint(1, self.T, (x.shape[0], )).to(x.device)
        epsilon = torch.randn_like(x)

        x_t = self.precomp['alphabar_sqrt'][ts, None, None, None] * x + \
            self.precomp['sqrtmab'][ts, None, None, None] * epsilon

        res = self.unet(
            x_t, ts / self.T
        )
        return self.criterion(epsilon, res)

    def generate(self, n_sample: int, size, device, hook=None) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)

        pbar = tqdm(range(self.T, 0, -1), desc="Sampling...", leave=False)

        for i in pbar:
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.unet(
                x_i,
                torch.tensor(i / self.T).to(device).repeat(n_sample, 1)
            )
            x_i = self.precomp['inverse_sqrt_alpha'][i] * \
                (x_i - eps * self.precomp['inv_sqrtmab'][i]) + \
                self.precomp['beta_t_sqrt'][i] * z

            if hook:
                hook(i, x_i)

        return x_i

    def ddpm_schedules(self, beta1: float, beta2: float, T: int, type):

        if type == "linear":
            beta_t = torch.linspace(beta1, beta2, T + 1)
        elif type == "cosine":
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
            s = 0.008 # hyperparameter to tune
            steps = T + 2
            x = torch.linspace(0, T + 1, steps)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_t = torch.clip(betas, 0.0001, 0.9999)

        print(beta_t.shape)
        assert beta_t.shape == (T + 1,)
        beta_t_sqrt = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        alpha_t_log = torch.log(alpha_t)

        alphabar = torch.cumsum(alpha_t_log, dim=0).exp()

        alphabar_sqrt = torch.sqrt(alphabar)

        inverse_sqrt_alpha = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar)
        inv_sqrtmab = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,
            "inverse_sqrt_alpha": inverse_sqrt_alpha,
            "beta_t_sqrt": beta_t_sqrt,
            "alphabar": alphabar,
            "alphabar_sqrt": alphabar_sqrt,
            "sqrtmab": sqrtmab,
            "inv_sqrtmab": inv_sqrtmab
        }
