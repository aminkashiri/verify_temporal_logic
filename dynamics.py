import torch
from torch import nn


class PendulumDynamics(nn.Module):
    def __init__(self, dt, gravity, length, mass):
        super().__init__()
        self.dt = dt
        self.gravity = gravity
        self.length = length
        self.mass = mass

    def forward(self, xt, ut):
        # got this from pg 15 of: https://arxiv.org/pdf/2108.01220.pdf
        # updated to values from page 21
        dt = self.dt
        gravity = self.gravity
        length = self.length
        mass = self.mass

        xt_0 = torch.matmul(xt, torch.Tensor([[1], [0]]).to(xt.device))
        xt_1 = torch.matmul(xt, torch.Tensor([[0], [1]]).to(xt.device))

        xt1_0 = torch.matmul(xt, torch.Tensor([[1.0], [dt]]).to(xt.device))
        xt1_1 = xt_1 + dt * (
            (gravity / length) * torch.sin(xt_0) + ut / (mass * length**2)
        )

        xt1 = torch.cat([xt1_0, xt1_1], 1)

        return xt1
