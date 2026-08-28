import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

type LossName = Literal["cross_entropy", "ce_dice"]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Config:
    data_root: str = "./data"
    image_size: tuple[int, int] = (64, 64)
    batch_size: int = 4
    val_ratio: float = 0.2
    seed: int = 42
    num_epochs: int = 1
    learning_rate: float = 1e-3
    num_classes: int = 2
    wandb_enabled: bool = True
    loss_name: LossName = "cross_entropy"
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_include_background: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.data_root, str) or not self.data_root:
            raise ValueError("data_root must be a non-empty string")

        if (
            not isinstance(self.image_size, (list, tuple))
            or len(self.image_size) != 2
            or not all(_is_int(value) and value > 0 for value in self.image_size)
        ):
            raise ValueError("image_size must contain two positive integers")
        if any(value % 16 != 0 for value in self.image_size):
            raise ValueError("image_size values must be divisible by 16 for UNet")
        object.__setattr__(self, "image_size", tuple(self.image_size))

        for name in ("batch_size", "num_epochs", "num_classes"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")

        if not _is_int(self.seed) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

        if (
            isinstance(self.val_ratio, bool)
            or not isinstance(self.val_ratio, (int, float))
            or not math.isfinite(self.val_ratio)
            or not 0 < self.val_ratio < 1
        ):
            raise ValueError("val_ratio must satisfy 0 < val_ratio < 1")

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number")

        if not isinstance(self.wandb_enabled, bool):
            raise TypeError("wandb_enabled must be a boolean")

        if self.loss_name not in ("cross_entropy", "ce_dice"):
            raise ValueError("loss_name must be 'cross_entropy' or 'ce_dice'")

        for name in ("ce_weight", "dice_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.ce_weight + self.dice_weight == 0:
            raise ValueError("ce_weight and dice_weight must not both be zero")

        if not isinstance(self.dice_include_background, bool):
            raise TypeError("dice_include_background must be a boolean")
        if (
            self.loss_name == "ce_dice"
            and not self.dice_include_background
            and self.num_classes < 2
        ):
            raise ValueError(
                "ce_dice excluding background requires num_classes >= 2"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Config":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown config key(s): {names}")
        return cls(**dict(values))

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON config: {config_path}") from error
        if not isinstance(values, dict):
            raise TypeError("config JSON must contain an object")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["image_size"] = list(self.image_size)
        return values
