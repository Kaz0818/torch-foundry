import math

from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config


class MulticlassCEDiceLoss(nn.Module):
    """Combine multiclass cross entropy with soft Dice loss.

    ``logits`` must have shape ``[B, C, H, W]`` and ``target`` must contain
    integer class IDs with shape ``[B, H, W]``.  Class ID 0 is treated as the
    background class when ``include_background`` is false.
    """

    def __init__(
        self,
        *,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        _validate_weight("ce_weight", ce_weight)
        _validate_weight("dice_weight", dice_weight)
        if ce_weight + dice_weight == 0:
            raise ValueError("ce_weight and dice_weight must not both be zero")
        if not isinstance(include_background, bool):
            raise TypeError("include_background must be a boolean")
        if not isinstance(smooth, (int, float)) or isinstance(smooth, bool):
            raise TypeError("smooth must be a positive finite number")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be a positive finite number")

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [B, C, H, W]")
        if target.shape != (logits.size(0), logits.size(2), logits.size(3)):
            raise ValueError("target must have shape [B, H, W] matching logits")
        if not self.include_background and logits.size(1) < 2:
            raise ValueError(
                "Dice loss excluding background requires at least two classes"
            )

        cross_entropy = F.cross_entropy(logits, target)
        probabilities = logits.softmax(dim=1)
        one_hot_target = F.one_hot(target, num_classes=logits.size(1))
        one_hot_target = one_hot_target.permute(0, 3, 1, 2).to(probabilities.dtype)

        reduction_dims = (0, 2, 3)
        intersection = (probabilities * one_hot_target).sum(dim=reduction_dims)
        denominator = probabilities.sum(dim=reduction_dims) + one_hot_target.sum(
            dim=reduction_dims
        )
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        dice_loss = 1 - dice.mean()

        return self.ce_weight * cross_entropy + self.dice_weight * dice_loss


def _validate_weight(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative finite number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def build_loss(config: Config) -> nn.Module:
    """Create the loss function selected by a segmentation ``Config``."""
    if config.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if config.loss_name == "ce_dice":
        return MulticlassCEDiceLoss(
            ce_weight=config.ce_weight,
            dice_weight=config.dice_weight,
            include_background=config.dice_include_background,
        )
    raise ValueError(f"unsupported loss_name: {config.loss_name}")
