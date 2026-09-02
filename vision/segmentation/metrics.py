"""Binary segmentation losses and metrics."""

import torch


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute soft Dice loss from logits and binary targets."""

    probs = torch.sigmoid(logits)

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)

    dice = (2 * intersection + eps) / (
        probs.sum(dim=1) + targets.sum(dim=1)
    )

    return 1 - dice.mean()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute thresholded Dice score."""

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(dim=1)

    dice = (2 * intersection + eps) / (
        preds.sum(dim=1) + targets.sum(dim=1) + eps
    )

    return dice.mean().item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute thresholded intersection-over-union."""

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.mean().item()
