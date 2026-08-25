from collections.abc import Callable, Iterable
from typing import Protocol, TypedDict

import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from ..metrics.segmentation import segmentation_counts, segmentation_metrics

type Batch = tuple[Tensor, Tensor]
type LossFunction = Callable[[Tensor, Tensor], Tensor]


class SegmentationModel(Protocol):
    def train(self, mode: bool = True) -> object: ...

    def eval(self) -> object: ...

    def __call__(self, images: Tensor) -> Tensor: ...


class TrainingHistory(TypedDict):
    train_loss: list[float]
    val_loss: list[float]
    val_iou: list[float]
    class_ious: list[list[float]]
    val_dice: list[float]
    class_dices: list[list[float]]


def train(
    *,
    train_loader: Iterable[Batch],
    val_loader: Iterable[Batch],
    model: SegmentationModel,
    criterion: LossFunction,
    optimizer: Optimizer,
    device: torch.device | str,
    num_epochs: int = 5,
    num_classes: int = 3,
) -> TrainingHistory:
    history: TrainingHistory = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "class_ious": [],
        "class_dices": [],
    }

    for epoch in range(num_epochs):
        # =================================================
        # Train
        # =================================================
        _ = model.train()

        running_loss = 0.0
        total_samples = 0

        for images, masks in tqdm(
            train_loader,
            desc="train",
            leave=False,
        ):
            images = images.to(device)
            masks = masks.to(device)

            batch_size = images.size(0)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
            optimizer.step()

            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        avg_train_loss = running_loss / total_samples

        # =================================================
        # Validation
        # =================================================
        _ = model.eval()

        running_val_loss = 0.0
        total_val_samples = 0

        # validation全体でclassごとの値を蓄積
        val_intersections = [0] * num_classes
        val_unions = [0] * num_classes
        val_pred_counts = [0] * num_classes
        val_target_counts = [0] * num_classes

        with torch.inference_mode():
            for images, masks in tqdm(
                val_loader,
                desc="val",
                leave=False,
            ):
                images = images.to(device)
                masks = masks.to(device)

                batch_size = images.size(0)

                logits = model(images)
                loss = criterion(logits, masks)

                running_val_loss += float(loss.item()) * batch_size
                total_val_samples += batch_size

                # [B, C, H, W] -> [B, H, W]
                pred_masks = logits.argmax(dim=1)

                # このbatchのclassごとの
                # intersection / union
                (
                    batch_intersections,
                    batch_unions,
                    batch_pred_counts,
                    batch_target_counts,
                ) = segmentation_counts(pred_masks, masks, num_classes)

                # validation全体に加算
                for class_id in range(num_classes):
                    val_intersections[class_id] += batch_intersections[class_id]
                    val_unions[class_id] += batch_unions[class_id]
                    val_pred_counts[class_id] += batch_pred_counts[class_id]
                    val_target_counts[class_id] += batch_target_counts[class_id]

        avg_val_loss = running_val_loss / total_val_samples

        # =================================================
        # IoU
        # =================================================
        class_ious, val_miou, class_dices, val_mdice = segmentation_metrics(
            val_intersections,
            val_unions,
            val_pred_counts,
            val_target_counts,
        )

        # =================================================
        # History
        # =================================================
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        history["val_iou"].append(val_miou)
        history["class_ious"].append(class_ious)

        history["val_dice"].append(val_mdice)
        history["class_dices"].append(class_dices)

        print(
            " | ".join(
                (
                    f"epoch: {epoch + 1}",
                    f"train loss: {avg_train_loss:.4f}",
                    f"val loss: {avg_val_loss:.4f}",
                    f"mIoU: {val_miou:.4f}",
                    f"mDice: {val_mdice:.4f}",
                    f"class IoU: {[round(x, 4) for x in class_ious]}",
                    f"class Dice: {[round(x, 4) for x in class_dices]}",
                )
            )
        )

    return history
