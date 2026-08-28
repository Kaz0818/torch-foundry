import argparse
import json
import random
from collections.abc import Sized
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import nn, optim

from vision.segmentation.config import Config
from vision.segmentation.datasets.dataset import (
    SegmentDataset,
    get_dataloader,
)
from vision.segmentation.datasets.oxford_iseg import prepare_oxford_iseg
from vision.segmentation.metrics.segmentation import (
    segmentation_counts,
    segmentation_metrics,
)
from vision.segmentation.models.unet import UNet
from vision.segmentation.training.evaluation import test_evaluation
from vision.segmentation.training.train import train
from vision.segmentation.utils.visualization import get_device, plot_overlay

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "vision" / "segmentation" / "config.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Oxford iSeg example and optionally log it to W&B."
    )
    parser.add_argument(
        "--wandb-project",
        help="W&B project name. Required when wandb_enabled is true.",
    )
    parser.add_argument(
        "--wandb-run-name",
        help="Optional W&B run name. W&B generates one when omitted.",
    )
    return parser.parse_args()


def main(
    *,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    print("Hello from torch-foundry!")
    config = Config.from_json(CONFIG_PATH)
    if config.wandb_enabled and not wandb_project:
        raise ValueError(
            "--wandb-project is required when wandb_enabled is true"
        )
    set_seed(config.seed)

    data_root = Path(config.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    # Oxford iSeg is small enough to use as a complete smoke-test dataset.
    images, masks = prepare_oxford_iseg(data_root)
    images = images[:20]
    masks = masks[:20]

    dataset = SegmentDataset(
        images=images,
        masks=masks,
        image_size=config.image_size,
        augment=False,
    )
    sample_image, sample_mask = dataset[0]
    print(
        "dataset sample:",
        "length=",
        len(dataset),
        "image_shape=",
        tuple(sample_image.shape),
        "image_dtype=",
        sample_image.dtype,
        "image_range=",
        (float(sample_image.min()), float(sample_image.max())),
        "mask_shape=",
        tuple(sample_mask.shape),
        "mask_dtype=",
        sample_mask.dtype,
        "mask_classes=",
        torch.unique(sample_mask).tolist(),
    )

    # 1: create train/validation loaders from the first 20 pairs.
    train_loader, val_loader = get_dataloader(
        images=images,
        masks=masks,
        batch_size=config.batch_size,
        image_size=config.image_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    print(
        "loaders:",
        "train_batches=",
        len(train_loader),
        "val_batches=",
        len(val_loader),
    )

    batch_images, batch_masks = next(iter(train_loader))
    print(
        "train batch:",
        "images=",
        tuple(batch_images.shape),
        batch_images.dtype,
        "masks=",
        tuple(batch_masks.shape),
        batch_masks.dtype,
    )

    # 2: get one_batch for check out compute model output shape
    device = get_device()
    print("device:", device)
    model = UNet(3, config.num_classes)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)

    # 3: criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    batch_images = batch_images.to(device)
    batch_masks = batch_masks.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(batch_images)
        expected_shape = (
            batch_images.size(0),
            config.num_classes,
            *config.image_size,
        )
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected U-Net output shape: {tuple(logits.shape)} "
                f"!= {expected_shape}"
            )
        check_loss = criterion(logits, batch_masks)
        predicted_masks = logits.argmax(dim=1)
        counts = segmentation_counts(
            predicted_masks,
            batch_masks,
            config.num_classes,
        )
        class_ious, mean_iou, class_dices, mean_dice = segmentation_metrics(
            *counts
        )
    print(
        "model check:",
        "logits=",
        tuple(logits.shape),
        "loss=",
        float(check_loss),
        "mIoU=",
        mean_iou,
        "mDice=",
        mean_dice,
        "class IoU=",
        class_ious,
        "class Dice=",
        class_dices,
    )

    output_dir = (
        PROJECT_ROOT
        / "vision"
        / "segmentation"
        / "runs"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print("output directory:", output_dir)
    effective_config = config.to_dict()
    effective_config["data_root"] = str(data_root)
    (output_dir / "config.json").write_text(
        json.dumps(effective_config, indent=2) + "\n", encoding="utf-8"
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    if not isinstance(train_dataset, Sized) or not isinstance(val_dataset, Sized):
        raise TypeError("train and validation datasets must define their size")

    wandb_config = {
        **effective_config,
        "dataset_name": "Oxford iSeg",
        "model_name": "UNet",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_parameters": total_params,
        "device": str(device),
    }
    wandb_mode = None if config.wandb_enabled else "disabled"

    with wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=wandb_config,
        dir=str(output_dir),
        mode=wandb_mode,
    ) as run:
        if config.wandb_enabled:
            run.define_metric("epoch")
            run.define_metric("train/*", step_metric="epoch")
            run.define_metric("val/*", step_metric="epoch")

        history = train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config.num_epochs,
            num_classes=config.num_classes,
            output_dir=output_dir,
            metric_logger=run.log if config.wandb_enabled else None,
        )

        if config.wandb_enabled:
            val_images, val_masks = next(iter(val_loader))
            model.eval()
            with torch.inference_mode():
                val_logits = model(val_images.to(device))
                val_predictions = val_logits.argmax(dim=1)
            comparison = plot_overlay(
                val_images,
                val_masks,
                val_predictions,
                max_images=5,
            )
            try:
                run.log({"results/predictions": wandb.Image(comparison)})
            finally:
                plt.close(comparison)
            print("W&B run:", run.url)

        # Use validation once more to exercise the public test-evaluation function.
        test_evaluation(
            val_loader,
            model,
            criterion,
            device,
            num_classes=config.num_classes,
        )
        print("saved history epochs:", len(history["train_loss"]))

    for artifact_name in ("model.pt", "history.json", "config.json"):
        artifact_path = output_dir / artifact_name
        if not artifact_path.is_file():
            raise RuntimeError(f"missing training artifact: {artifact_path}")


if __name__ == "__main__":
    arguments = parse_args()
    main(
        wandb_project=arguments.wandb_project,
        wandb_run_name=arguments.wandb_run_name,
    )
