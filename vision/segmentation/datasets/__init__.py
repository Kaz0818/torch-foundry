"""Dataset adapters used by the segmentation examples."""

from .flood import FloodDataset, prepare_dataloaders

__all__ = ["FloodDataset", "prepare_dataloaders"]
