"""MonoTeeth network: encoder -> query decoder -> heads, plus loss / metric / checkpoint / preview helpers."""

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..util.loss import bce_loss, dice_loss, seg_metrics
from ..util.vis import colorize, to_uint8


class MonoTeethNet(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, heads: dict[str, nn.Module]) -> None:
        """heads: name -> head; names are "latent" and optionally "seg"."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            rgb: [B, 3, H, W] in [0, 1].

        Returns:
            queries [B, Q, C], plus one output per head: latent [B, Q, 20], seg logits [B, Q, H, W].
        """
        feats = self.encoder(rgb)
        queries = self.decoder(feats["tokens"], feats["tokens_pos"])
        out = {"queries": queries}
        for name, head in self.heads.items():
            out[name] = head(queries, feats, tuple(rgb.shape[-2:]))
        return out

    def compute_loss(self, pred: dict, batch: dict, weights: dict[str, float]) -> dict[str, torch.Tensor]:
        terms = {}
        if "latent" in pred:
            terms["latent_l1"] = F.l1_loss(pred["latent"].float(), batch["latents"])
        if "seg" in pred:
            logits, targets = pred["seg"].float(), batch["semantic_maps"]
            terms["seg_bce"] = bce_loss(logits, targets)
            terms["seg_dice_loss"] = dice_loss(logits, targets)

        total = 0.0
        for key, value in terms.items():
            if key not in weights:
                raise KeyError(f"No weight for loss term {key!r}; set loss.{key}.")
            total = total + weights[key] * value
        return {"loss": total, **{key: value.detach() for key, value in terms.items()}}

    @torch.no_grad()
    def compute_metrics(self, pred: dict, batch: dict) -> dict[str, torch.Tensor]:
        metrics = {}
        if "latent" in pred:
            metrics["latent_l1"] = F.l1_loss(pred["latent"].float(), batch["latents"])
        if "seg" in pred:
            metrics |= seg_metrics(pred["seg"].float(), batch["semantic_maps"])
        return metrics

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("encoder.")]
        if missing or unexpected:
            raise KeyError(f"Checkpoint mismatch: missing={missing} unexpected={unexpected}")

    def visualize(self, pred: dict, batch: dict, i: int) -> np.ndarray:
        panels = [to_uint8(batch["rgb"][i]), colorize(batch["semantic_maps"][i])]
        if "seg" in pred:
            panels.append(colorize(pred["seg"][i].float().sigmoid()))
        return np.concatenate(panels, axis=1)


def build_network(model_cfg: DictConfig | dict) -> MonoTeethNet:
    """Build MonoTeethNet from cfg.model; only its encoder / decoder / heads entries are instantiated."""
    cfg = OmegaConf.to_container(model_cfg, resolve=True) if isinstance(model_cfg, DictConfig) else model_cfg
    return MonoTeethNet(
        encoder=instantiate(cfg["encoder"]),
        decoder=instantiate(cfg["decoder"]),
        heads={name: instantiate(head) for name, head in cfg["heads"].items()},
    )
