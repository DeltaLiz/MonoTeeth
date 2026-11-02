import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss averaged over samples and parts."""
    probs = logits.sigmoid()
    inter = (probs * targets).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def seg_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Dice and mIoU of masks thresholded at 0.5; mIoU skips parts absent from both masks."""
    pred = logits.sigmoid() > 0.5
    gt = targets > 0.5
    inter = (pred & gt).sum(dim=(-2, -1)).float()
    pred_area, gt_area = pred.sum(dim=(-2, -1)).float(), gt.sum(dim=(-2, -1)).float()
    union = pred_area + gt_area - inter
    valid = union > 0
    return {
        "seg_dice": ((2 * inter + 1) / (pred_area + gt_area + 1)).mean(),
        "seg_miou": (inter[valid] / union[valid]).mean() if valid.any() else union.new_ones(()),
    }
