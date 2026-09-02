"""Flood Area Segmentation dataset adapter.

The adapter keeps the dataset-specific CSV and directory conventions in one
place. Other binary segmentation datasets can be added as separate adapters
without changing the training loop.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load the dataset metadata CSV from ``data_dir``."""

    return pd.read_csv(Path(data_dir) / "metadata.csv")


def prepare_dataframe(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add image/mask paths and existence flags to metadata."""

    data_dir = Path(data_dir)
    image_dir = data_dir / "Image"
    mask_dir = data_dir / "Mask"

    prepared = df.copy()
    prepared["image_path"] = prepared["Image"].apply(
        lambda name: image_dir / name
    )
    prepared["mask_path"] = prepared["Mask"].apply(
        lambda name: mask_dir / name
    )
    prepared["image_exists"] = prepared["image_path"].apply(Path.exists)
    prepared["mask_exists"] = prepared["mask_path"].apply(Path.exists)

    return prepared


def get_image_mask_size_info(row: pd.Series) -> pd.Series:
    """Return image/mask dimensions and whether the pair has matching sizes."""

    with Image.open(row["image_path"]) as image, Image.open(
        row["mask_path"]
    ) as mask:
        return pd.Series(
            {
                "image_height": image.height,
                "image_width": image.width,
                "mask_height": mask.height,
                "mask_width": mask.width,
                "size_match": image.size == mask.size,
            }
        )


def create_dataloaders(
    df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders with a 70:15:15 split."""

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        shuffle=True,
        random_state=seed,
    )

    train_dataset = FloodDataset(train_df, augment=True)
    val_dataset = FloodDataset(val_df, augment=False)
    test_dataset = FloodDataset(test_df, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def prepare_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Load, validate, and split the flood dataset."""

    data_dir = Path(data_dir)
    df = prepare_dataframe(load_metadata(data_dir), data_dir)

    existing_df = df[df["image_exists"] & df["mask_exists"]].copy()
    size_info = existing_df.apply(get_image_mask_size_info, axis=1)
    existing_df = pd.concat([existing_df, size_info], axis=1)

    clean_df = existing_df[existing_df["size_match"]].copy()
    train_loader, val_loader, test_loader = create_dataloaders(
        clean_df,
        batch_size=batch_size,
    )
    sample_counts = {
        "total": len(df),
        "existing_pairs": len(existing_df),
        "usable": len(clean_df),
    }

    return train_loader, val_loader, test_loader, sample_counts


class FloodDataset(Dataset):
    """Dataset for RGB flood images and binary grayscale masks."""

    def __init__(self, df: pd.DataFrame, augment: bool = True) -> None:
        self.df = df.reset_index(drop=True)

        if augment:
            self.transform = v2.Compose(
                [
                    v2.Resize((256, 256)),
                    v2.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.transform = v2.Compose([v2.Resize((256, 256))])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]

        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        # convert("L") keeps the mask as 8-bit grayscale.
        mask = np.array(Image.open(row["mask_path"]).convert("L"))

        # HWC -> CHW.
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask)

        image_tv = tv_tensors.Image(image_tensor)
        mask_tv = tv_tensors.Mask(mask_tensor)
        image_tv, mask_tv = self.transform(image_tv, mask_tv)

        # image: uint8 0~255 -> float32 0~1.
        image_tv = v2.ToDtype(torch.float32, scale=True)(image_tv)

        # mask: background / flood -> 0 / 1.
        mask_tv = (mask_tv > 127).float()

        return image_tv, mask_tv
