# pyright: basic

from pathlib import Path
from typing import cast

import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2


class PetSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "./data",
        split: str = "trainval",
        image_size: tuple[int, int] = (256, 256),
        train: bool = True,
    ) -> None:
        self.dataset = OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=True
        )

        if train:
            self.transforms = v2.Compose(
                [
                    v2.Resize(image_size),
                    v2.RandomHorizontalFlip(p=0.9),
                ]
            )

        else:
            self.transforms = v2.Compose(
                [
                    v2.Resize(image_size),
                ]
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, mask = self.dataset[index]

        # PIL -> v2用Tensor
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        # 同じResizeを両方に適用
        image, mask = self.transforms(image, mask)

        # image: uint8 [0,255] -> float32 [0,1]
        image = v2.ToDtype(torch.float32, scale=True)(image)

        # mask: class IDをそのまま整数として扱う
        # # 1,2,3 -> 0,1,2
        mask = mask.squeeze(0).long() - 1

        return image, mask


def load_dataloaders(
    test_size: float | None = None,
    *,
    root: str | Path = "./data",
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 32,
    seed: int = 42,
    val_ratio: float | None = None,
):
    if test_size is not None and val_ratio is not None:
        raise ValueError("specify either test_size or val_ratio, not both")
    split_ratio = (
        0.2
        if test_size is None and val_ratio is None
        else (val_ratio if val_ratio is not None else test_size)
    )
    if split_ratio is None or not 0 < split_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 < val_ratio < 1")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be an integer >= 1")

    train_dataset = PetSegmentationDataset(
        root=root, split="trainval", image_size=image_size, train=True
    )
    val_dataset = PetSegmentationDataset(
        root=root, split="trainval", image_size=image_size, train=False
    )

    indices = list(range(len(train_dataset)))

    train_idx, val_idx = train_test_split(
        indices, test_size=split_ratio, random_state=seed, shuffle=True
    )

    train_idx = cast(list[int], train_idx)
    val_idx = cast(list[int], val_idx)

    train_ds = Subset(train_dataset, train_idx)

    val_ds = Subset(val_dataset, val_idx)

    test_ds = PetSegmentationDataset(
        root=root, split="test", image_size=image_size, train=False
    )

    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=train_generator
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def load_smoke_dataloader(
    train_ds,
    val_ds,
    test_ds,
    train_ratio=0.1,
    val_ratio=0.1,
    test_ratio=0.05,
    seed=42,
):
    for name, dataset, ratio in (
        ("train", train_ds, train_ratio),
        ("val", val_ds, val_ratio),
        ("test", test_ds, test_ratio),
    ):
        if len(dataset) == 0:
            raise ValueError(f"{name}_ds must not be empty")
        if not 0 < ratio <= 1:
            raise ValueError(f"{name}_ratio must satisfy 0 < ratio <= 1")

    g = torch.Generator().manual_seed(seed)

    train_idx = torch.randperm(len(train_ds), generator=g)[
        : max(1, int(len(train_ds) * train_ratio))
    ].tolist()

    val_idx = torch.randperm(len(val_ds), generator=g)[
        : max(1, int(len(val_ds) * val_ratio))
    ].tolist()

    test_idx = torch.randperm(len(test_ds), generator=g)[
        : max(1, int(len(test_ds) * test_ratio))
    ].tolist()

    smoke_train_ds = Subset(train_ds, train_idx)
    smoke_val_ds = Subset(val_ds, val_idx)
    smoke_test_ds = Subset(test_ds, test_idx)

    smoke_train_generator = torch.Generator().manual_seed(seed)
    smoke_train_loader = DataLoader(
        smoke_train_ds,
        batch_size=5,
        shuffle=True,
        generator=smoke_train_generator,
    )

    smoke_val_loader = DataLoader(smoke_val_ds, batch_size=5, shuffle=False)

    smoke_test_loader = DataLoader(smoke_test_ds, batch_size=5, shuffle=False)

    return smoke_train_loader, smoke_val_loader, smoke_test_loader
