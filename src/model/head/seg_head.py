import math

import torch
from torch import nn

from src.third_party.sam3.model.model_misc import MLP


class CoordSegHead(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 256, num_layers: int = 3, num_freqs: int = 10) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs) * math.pi, persistent=False)
        self.h_query = MLP(dim, hidden_dim, dim, num_layers)
        self.h_coord = MLP(2 + 4 * num_freqs, hidden_dim, dim, num_layers)

    def encode(self, out_size: tuple[int, int]) -> torch.Tensor:
        h, w = out_size
        device = self.freqs.device
        ys = (torch.arange(h, device=device) + 0.5) / h
        xs = (torch.arange(w, device=device) + 0.5) / w
        u = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)  # [H, W, 2] as (x, y)
        angles = (u[..., None] * self.freqs).flatten(-2)
        return torch.cat([u, angles.sin(), angles.cos()], dim=-1)

    def forward(self, queries: torch.Tensor, feats: dict, out_size: tuple[int, int]) -> torch.Tensor:
        coord = self.h_coord(self.encode(out_size))
        return torch.einsum("bqc,hwc->bqhw", self.h_query(queries), coord)
