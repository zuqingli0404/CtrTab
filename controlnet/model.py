import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
import math

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class ControlRNet(nn.Module):
    def __init__(self, d_in, cond_in, d_layers, dropout, dim_t = 1024):
        super().__init__()
        self.dim_t = dim_t
        self.cond_in = cond_in

        self.start = zero_module(nn.Conv1d(1, 1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
        )
        self.zeros = nn.ModuleList()
        for b in range(2):
            self.zeros.append(zero_module(nn.Conv1d(1, 1, 1)))
        self.initzero = zero_module(nn.Conv1d(1, 1, 1))
        self.trans = nn.Linear(self.cond_in, d_in)
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
    
    def forward(self, x, timesteps, ct, cond=None):
        x = x.to()
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) 
        emb = self.time_embed(emb)
        cf = self.start(ct.unsqueeze(1)).squeeze(1)

        concatx = x + cf
        initx = self.initzero(concatx.unsqueeze(1)).squeeze(1)
        x = self.proj(concatx) + emb
        down = [x]
        x = self.zeros[0](self.mlp[1](self.mlp[0](x)).unsqueeze(1)).squeeze(1)
        down.insert(0, x)
        x = self.zeros[1](self.mlp[3](self.mlp[2](x)).unsqueeze(1)).squeeze(1)
        mid = x
        return [mid, down, initx]
