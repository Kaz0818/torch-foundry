import torch

from vision.segmentation.models.unet import UNet


def main():
    print("Hello from torch-foundry!")
    x = torch.randn(3, 3, 256, 256)
    model = UNet(3, 3)
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
