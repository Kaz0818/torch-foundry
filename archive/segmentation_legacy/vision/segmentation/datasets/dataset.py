from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import v2


class SegmentDataset(Dataset):
    """
    kaggleでのDataset作成方法,最後のmaskだけmaskのunique確認して0~始まっていれば
    -1はいらない。現状のままでok
    images: Pathでのimages配列を想定
    masks: Pathでのmask配列を想定
    """

    def __init__(
        self,
        images: list[Path],
        masks: list[Path],
        image_size: tuple[int, int] = (256, 256),
        augment: bool = True,
    ) -> None:
        if not images:
            raise ValueError("images must not be empty")
        if len(images) != len(masks):
            raise ValueError(
                f"images and masks must have the same length: "
                f"{len(images)} != {len(masks)}"
            )

        self.images = images
        self.masks = masks

        if augment:
            self.transforms = v2.Compose(
                [
                    v2.Resize(image_size),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.transforms = v2.Compose(
                [
                    v2.Resize(image_size),
                ]
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        image, mask = self.transforms(image, mask)

        image = v2.ToDtype(
            torch.float32,
            scale=True,
        )(image)

        mask = mask.squeeze(0).long()

        return image, mask


def get_dataloader(
    images: list[Path],
    masks: list[Path],
    batch_size: int,
    image_size: tuple[int, int] = (256, 256),
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    images: imageの配列.Pathで.globで取得したimage配列
    masks: Path.globで取得したmasksの配列
    batch_size: batchを指定
    """

    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if not isinstance(val_ratio, (int, float)) or isinstance(val_ratio, bool):
        raise TypeError("val_ratio must satisfy 0 < val_ratio < 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 < val_ratio < 1")

    dataset = SegmentDataset(
        images=images,
        masks=masks,
        image_size=image_size,
        augment=True,
    )
    val_dataset = SegmentDataset(
        images=images,
        masks=masks,
        image_size=image_size,
        augment=False,
    )

    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        shuffle=True,
        random_state=seed,
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(val_dataset, val_idx)

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
