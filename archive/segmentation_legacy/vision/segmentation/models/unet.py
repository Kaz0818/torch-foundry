# pyright: basic

import torch  # pyright: ignore[reportMissingImports]
from torch import Tensor, nn  # pyright: ignore[reportMissingImports]


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        transposed: bool = True,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoder1 = self._double_conv(in_ch, 64)
        self.encoder2 = self._double_conv(64, 128)
        self.encoder3 = self._double_conv(128, 256)
        self.encoder4 = self._double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)

        # Decoder
        self.up4 = self._up_conv(1024, 512, transposed)
        self.decoder4 = self._double_conv(1024, 512)

        self.up3 = self._up_conv(512, 256, transposed)
        self.decoder3 = self._double_conv(512, 256)

        self.up2 = self._up_conv(256, 128, transposed)
        self.decoder2 = self._double_conv(256, 128)

        self.up1 = self._up_conv(128, 64, transposed)
        self.decoder1 = self._double_conv(128, 64)

        # Pixel-wise classification
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _double_conv(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def _up_conv(
        self,
        in_ch: int,
        out_ch: int,
        transposed: bool = True,
    ) -> nn.Module:
        if transposed:
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:

        # Encoder
        enc1 = self.encoder1(x)  # [5, 3, 256, 256] -> [5, 64, 256, 256]
        x = self.maxpool(enc1)  # [5, 64, 256, 256] -> [5, 64, 128, 128]

        enc2 = self.encoder2(x)  # [5, 64, 128, 128] -> [5, 128, 128, 128]
        x = self.maxpool(enc2)  # [5, 128, 128, 128] -> [5, 128, 64, 64]

        enc3 = self.encoder3(x)  # [5, 128, 64, 64] -> [5, 256, 64, 64]
        x = self.maxpool(enc3)  # [5, 256, 64, 64] -> [5, 256, 32, 32]

        enc4 = self.encoder4(x)  # [5, 256, 32, 32] -> [5, 512, 32, 32]
        x = self.maxpool(enc4)  # [5, 512, 32, 32] -> [5, 512, 16, 16]

        # Bottleneck
        x = self.bottleneck(x)  # [5, 512, 16, 16] -> [5, 1024, 16, 16]

        # Decoder
        x = self.up4(x)  # [5, 1024, 16, 16] -> [5, 512, 32, 32]
        x = torch.cat(
            [x, enc4], dim=1
        )  # enc4[5, 512, 32, 32]+[5, 512, 32, 32] ->[5, 1024, 32, 32]
        x = self.decoder4(x)  # [5, 1024, 32, 32] -> [5, 512, 32, 32]

        x = self.up3(x)  # [5, 512, 32, 32] -> [5, 256, 64, 64]
        x = torch.cat(
            [x, enc3], dim=1
        )  # enc3[5, 256, 64, 64]+[5, 256, 64, 64] ->[5, 512, 64, 64]
        x = self.decoder3(x)  # [5, 512, 64, 64] -> [5, 256, 64, 64]

        x = self.up2(x)  # [5, 256, 64, 64] -> [5, 128, 128, 128]
        x = torch.cat(
            [x, enc2], dim=1
        )  # enc2[5, 128, 128, 128]+[5, 128, 128, 128] ->[5, 256, 128, 128]
        x = self.decoder2(x)  # [5, 256, 128, 128] -> [5, 128, 128, 128]

        x = self.up1(x)  # [5, 128, 128, 128] -> [5, 64, 256, 256]
        x = torch.cat(
            [x, enc1], dim=1
        )  # enc1[5, 64, 256, 256]+[5, 64, 256, 256] -> [5, 128, 256, 256]
        x = self.decoder1(x)  # [5, 128, 256, 256] -> [5, 64, 256, 256]

        # [B, 64, H, W] -> [B, num_classes, H, W]
        logits = self.classifier(x)  # [5, 64, 256, 256] -> [5, 3, 256, 256]

        return logits
