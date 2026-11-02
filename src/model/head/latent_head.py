import torch
from torch import nn

from src.third_party.sam3.model.model_misc import MLP


class LatentHead(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 256, num_layers: int = 3, latent_dim: int = 20) -> None:
        super().__init__()
        self.mlp = MLP(dim, hidden_dim, latent_dim, num_layers)

    def forward(self, queries: torch.Tensor, feats: dict, out_size: tuple[int, int]) -> torch.Tensor:
        """queries [B, Q, dim] -> latents [B, Q, latent_dim]."""
        return self.mlp(queries)
