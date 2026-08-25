# pyright: basic

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2


class PetSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        split: str = "trainval",
        image_size: tuple[int, int] = (256, 256),
        train: bool = True,
    ) -> None:
        self.dataset = OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=True
        )

        if train:
            self.transforms = v2.Compose(
                [
                    v2.Resize(image_size),
                    v2.RandomHorizontalFlip(p=0.9),
                ]
            )

        else:
            self.transforms = v2.Compose(
                [
                    v2.Resize(image_size),
                ]
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, mask = self.dataset[index]

        # PIL -> v2用Tensor
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        # 同じResizeを両方に適用
        image, mask = self.transforms(image, mask)

        # image: uint8 [0,255] -> float32 [0,1]
        image = v2.ToDtype(torch.float32, scale=True)(image)

        # mask: class IDをそのまま整数として扱う
        # # 1,2,3 -> 0,1,2
        mask = mask.squeeze(0).long() - 1

        return image, mask
