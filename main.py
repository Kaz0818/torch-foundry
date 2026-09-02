"""Command-line entry point for flood-area binary segmentation."""

import argparse
from pathlib import Path

import torch
import wandb
from torch import nn, optim

from vision.segmentation.datasets.flood import prepare_dataloaders
from vision.segmentation.models import load_model
from vision.segmentation.train import train_one_epoch, validate
from vision.segmentation.utils import select_device


def print_data_summary(
    train_loader,
    val_loader,
    test_loader,
    sample_counts: dict[str, int],
) -> None:
    print(
        "Data samples: "
        f"total={sample_counts['total']}, "
        f"existing_pairs={sample_counts['existing_pairs']}, "
        f"usable={sample_counts['usable']}"
    )
    print(
        "Split samples: "
        f"train={len(train_loader.dataset)}, "
        f"validation={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )


def setup_training():
    model = load_model()
    total_params = sum(p.numel() for p in model.parameters())
    device = select_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model parameters: {total_params:,}")
    print(f"Device: {device}")

    return model, criterion, optimizer, device


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs: int,
    batch_size: int,
):
    run = wandb.init(
        project="flood-segmentation",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": 1e-3,
            "image_size": 256,
            "loss": "BCE + Dice",
        },
    )

    best_dice = 0.0
    best_model_path = "best_model.pt"

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        val_loss = val_metrics["loss"]
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]

        run.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated: Dice = {best_dice:.4f}")

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f}"
        )

    run.finish()
    return model


def main(
    data_dir: str | Path = "data",
    epochs: int = 10,
    batch_size: int = 8,
):
    """Prepare data, train the model, and return objects for notebook use."""

    data_dir = Path(data_dir)
    train_loader, val_loader, test_loader, sample_counts = prepare_dataloaders(
        data_dir,
        batch_size,
    )
    print_data_summary(train_loader, val_loader, test_loader, sample_counts)
    model, criterion, optimizer, device = setup_training()
    model = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs,
        batch_size,
    )
    return model, val_loader, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the flood area segmentation model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing metadata.csv, Image/, and Mask/.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
