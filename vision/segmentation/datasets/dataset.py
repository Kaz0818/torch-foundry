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


def get_dataloader(images, masks, batch_size):
    """
    images: imageの配列.Pathで.globで取得したimage配列
    masks: Path.globで取得したmasksの配列
    batch_size: batchを指定
    """

    indices = list(range(len(images)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    train_full = SegmentDataset(
        images=images,
        masks=masks,
        image_size=(256, 256),
        augment=True,
    )

    val_full = SegmentDataset(
        images=images,
        masks=masks,
        image_size=(256, 256),
        augment=False,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
