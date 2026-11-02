import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

log = logging.getLogger("monoteeth")


def setup_logging(log_file: Path, rank: int = 0) -> None:
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, delay=True)]
    logging.basicConfig(level=logging.WARNING, handlers=handlers, force=True,
                        format=f"[%(asctime)s][rank{rank}][%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.captureWarnings(True)
    log.setLevel(logging.INFO)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


class Averager:
    """Sample-weighted means of scalar tensors."""

    def __init__(self) -> None:
        self.sums, self.count = {}, 0

    def add(self, values: dict[str, torch.Tensor], n: int) -> None:
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + v.detach().double() * n
        self.count += n

    def result(self, reduce: bool = False) -> dict[str, float]:
        """reduce: sum over all DDP ranks first."""
        keys = sorted(self.sums)
        sums = torch.stack([self.sums[k] for k in keys])
        packed = torch.cat([sums, sums.new_tensor([self.count])])
        if reduce:
            dist.all_reduce(packed)
        return dict(zip(keys, (packed[:-1] / packed[-1]).tolist()))
