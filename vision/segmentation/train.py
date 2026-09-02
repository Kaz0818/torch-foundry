"""Training and validation loops for binary segmentation."""

import torch
from tqdm import tqdm

from .metrics import dice_loss, dice_score, iou_score


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)

        bce = criterion(logits, masks)
        dice = dice_loss(logits, masks)
        loss = bce + dice

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def validate(
    model,
    loader,
    criterion,
    device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            logits = model(images)

            bce = criterion(logits, masks)
            dice = dice_loss(logits, masks)
            loss = bce + dice

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_dice += dice_score(logits, masks) * batch_size
            total_iou += iou_score(logits, masks) * batch_size

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }
