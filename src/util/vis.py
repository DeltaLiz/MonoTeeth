"""Image helpers for previews and exports."""

import numpy as np
import torch

from .constants import PALETTE


def to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """RGB [3, H, W] in [0, 1] -> uint8 [H, W, 3]."""
    return (rgb.detach().float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def colorize(probs: torch.Tensor) -> np.ndarray:
    """Per-part probabilities [Q, H, W] -> uint8 [H, W, 3]; pixels whose best part is below 0.5 stay black."""
    scores, labels = probs.detach().float().max(dim=0)
    scores, labels = scores.cpu().numpy(), labels.cpu().numpy()
    rgb = np.zeros((*scores.shape, 3), dtype=np.uint8)
    keep = scores >= 0.5
    rgb[keep] = PALETTE[labels[keep]]
    return rgb
