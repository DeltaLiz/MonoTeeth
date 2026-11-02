import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

CKPT_PREFIX = "detector.transformer.decoder."


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, tokens: torch.Tensor, tokens_pos: torch.Tensor) -> torch.Tensor:
        q = self.norm2(q + self.dropout(self.self_attn(q, q, q, need_weights=False)[0]))
        q = self.norm1(q + self.dropout(self.cross_attn(q, tokens + tokens_pos, tokens, need_weights=False)[0]))
        ffn = self.linear2(self.dropout(F.relu(self.linear1(q))))
        return self.norm3(q + self.dropout(ffn))


class DetrDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int = 30,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        init_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.act_ckpt = False  # activation checkpointing per layer during training; set from cfg.train.act_ckpt
        self.query_embed = nn.Embedding(num_queries, dim)
        self.layers = nn.ModuleList(DecoderLayer(dim, num_heads, ffn_dim, dropout) for _ in range(num_layers))
        self.norm = nn.LayerNorm(dim)
        if init_ckpt is not None:
            state = torch.load(init_ckpt, map_location="cpu", weights_only=True, mmap=True)
            state = state.get("model", state)
            for name in ("layers", "norm"):
                module = getattr(self, name)
                module.load_state_dict({key: state[f"{CKPT_PREFIX}{name}.{key}"] for key in module.state_dict()})

    def forward(self, tokens: torch.Tensor, tokens_pos: torch.Tensor) -> torch.Tensor:
        """tokens, tokens_pos: [B, N, dim] -> queries [B, Q, dim]."""
        q = self.query_embed.weight.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            if self.act_ckpt and self.training:
                q = checkpoint(layer, q, tokens, tokens_pos, use_reentrant=False)
            else:
                q = layer(q, tokens, tokens_pos)
        return self.norm(q)
