"""Segmentation model factory."""

import segmentation_models_pytorch as smp


def load_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    out_channels: int = 1,
):
    """Build the U-Net used by the flood segmentation example."""

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )
