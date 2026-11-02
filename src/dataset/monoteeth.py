import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..util.constants import PARTS


def crop_resize(array: np.ndarray, crop: dict, interpolation: int) -> np.ndarray:
    """Crop array [H, W, ...] to the rounded box and resize to (crop["height"], crop["width"])."""
    height, width = array.shape[:2]
    left, top = max(0, round(crop["x0"])), max(0, round(crop["y0"]))
    right, bottom = min(width, round(crop["x1"])), min(height, round(crop["y1"]))
    return cv2.resize(array[top:bottom, left:right], (crop["width"], crop["height"]), interpolation=interpolation)


def read_crops(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


class MonoTeethDataset(Dataset):
    def __init__(self, root: str, split: str, augment: str, crop_index: int | None) -> None:
        """
        Args:
            root: Dataset root.
            split: "train" or "test".
            augment: "random_10" or "center_1".
            crop_index: Indicate the crop. "None" uses all.
        """
        self.root = Path(root)
        self.augment_root = self.root / "augment" / augment
        subject_ids = json.loads((self.root / "split.json").read_text())[split]
        self.samples = []  # (subject_id, crop_index, crop)
        for sid in subject_ids:
            crops = list(enumerate(read_crops(self.augment_root / "crop" / f"{sid}.jsonl")))
            if crop_index is not None:
                crops = [crops[crop_index]]
            self.samples += [(sid, k, crop) for k, crop in crops]

        self.latents = {sid: self._load_latents(sid) for sid in subject_ids}

    def _load_latents(self, sid: str) -> torch.Tensor:
        """Latents [30, 20] in PARTS order."""
        states = {jaw: torch.load(self.root / "data" / sid / f"{jaw}.pt", map_location="cpu", weights_only=True)
                  for jaw in ("upper", "lower")}
        return torch.stack([states[jaw][f"latent_vecs.{i}.weight"].reshape(-1).float() for jaw, i in PARTS])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """
        Returns:
            rgb: [3, h, w] float32 in [0, 1].
            semantic_maps: [30, h, w] float32 one-hot masks in PARTS order (all zero on background).
            latents: [30, 20] float32 in PARTS order.
            subject_id: str; crop_index: int.
        """
        sid, k, crop = self.samples[index]
        subject_dir = self.root / "data" / sid
        with Image.open(subject_dir / "rgb.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        rgb = crop_resize(rgb, crop, cv2.INTER_LINEAR)

        with Image.open(subject_dir / "segmentation.png") as image:
            labels = crop_resize(np.asarray(image), crop, cv2.INTER_NEAREST_EXACT)
        semantic = labels[None] == np.arange(1, len(PARTS) + 1, dtype=labels.dtype)[:, None, None]
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255),
            "semantic_maps": torch.from_numpy(semantic).float(),
            "latents": self.latents[sid].clone(),
            "subject_id": sid,
            "crop_index": k,
        }


def build_loader(cfg, split: str, batch_size: int, num_workers: int, distributed: bool = False) -> DataLoader:
    """Training loaders shuffle (through DistributedSampler when distributed); test loaders keep file order."""
    train = split == "train"
    crop_index = cfg.train_crop_index if train else cfg.test_crop_index
    dataset = MonoTeethDataset(cfg.root, split, cfg.augment, crop_index)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=train and sampler is None,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
