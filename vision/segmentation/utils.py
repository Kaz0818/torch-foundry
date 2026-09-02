"""Small runtime and prediction-visualization helpers."""

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = select_device()


def collect_predictions(
    model,
    loader,
    device: str | torch.device,
    num_samples: int = 3,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect image, ground-truth mask, and prediction samples."""

    samples = []
    with torch.inference_mode():
        model.eval()
        for images, masks in tqdm(
            loader,
            total=len(loader),
            desc="eval",
            leave=False,
        ):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            predicts = torch.sigmoid(logits) >= 0.5

            for image, mask, pred in zip(images, masks, predicts):
                samples.append((image.cpu(), mask.cpu(), pred.cpu()))
                if len(samples) == num_samples:
                    break

            if len(samples) == num_samples:
                break

    return samples


def plot_predictions(samples):
    """Plot image, ground truth, prediction, and overlay for each sample."""

    if not samples:
        raise ValueError("samples must contain at least one prediction")

    fig, axes = plt.subplots(
        len(samples),
        4,
        figsize=(12, 4 * len(samples)),
        squeeze=False,
    )
    for row, (image, mask, pred) in enumerate(samples):
        ax_img, ax_gt, ax_pred, ax_overlay = axes[row]

        ax_img.imshow(image.permute(1, 2, 0))
        ax_img.set_title("Image")

        ax_gt.imshow(mask, cmap="gray")
        ax_gt.set_title("GT")

        ax_pred.imshow(pred.squeeze(0), cmap="gray")
        ax_pred.set_title("Predict")

        ax_overlay.imshow(image.permute(1, 2, 0))
        ax_overlay.imshow(pred.squeeze(0), alpha=0.5, cmap="Blues")
        ax_overlay.set_title("Overlay")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    return fig, axes


__all__ = ["collect_predictions", "device", "plot_predictions", "select_device"]
