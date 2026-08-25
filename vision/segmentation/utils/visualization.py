# pyright: basic

import matplotlib.pyplot as plt
from torch import Tensor

from ..training.train import TrainingHistory


def show_image_mask_batch(
    images: Tensor,
    masks: Tensor,
    max_images: int = 5,
) -> None:
    """
    Plot images and corresponding segmentation masks from a batch.

    images: [B, C, H, W] next(iter(DataLoader))で取得
    masks:   [B, H, W]   上と同じ
    """
    n = min(max_images, len(images))

    _, axes = plt.subplots(
        n,
        2,
        figsize=(10, n * 3),
        squeeze=False)

    for i in range(n):
        image = images[i].cpu()
        mask = masks[i].cpu()

        axes[i, 0].imshow(image.permute(1, 2, 0))
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_overlay(images: Tensor, masks: Tensor, pred_masks: Tensor) -> None:
    """
    model学習後の比較:
    images: datasetからの画像
    masks: ground truth mask
    pred_mask: logitsに各pixel毎にargmaxをした予測mask
    """

    n = min(5, len(images))
    _, axes = plt.subplots(n, 4, figsize=(16, 4*n), squeeze=False)

    for i in range(n):
        image = images[i].permute(1, 2, 0).cpu()
        true_mask = masks[i].cpu()
        pred_mask = pred_masks[i].cpu()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image', fontsize=20)
        axes[i, 0].axis('off')


        axes[i, 1].imshow(true_mask)
        axes[i, 1].set_title('Ground Truth', fontsize=20)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask)
        axes[i, 2].set_title('Prediction', fontsize=20)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(image)
        axes[i, 3].imshow(pred_mask, alpha=0.4)
        axes[i, 3].set_title('Prediction Overlay', fontsize=20)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


def plot_train_val_loss_val_iou(history: TrainingHistory) -> None:

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train / Val Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'])
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Val IoU')

    plt.tight_layout()
    plt.show()
