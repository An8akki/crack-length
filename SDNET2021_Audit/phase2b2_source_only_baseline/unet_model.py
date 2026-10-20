"""
U-Net Semantic Segmentation Architecture for SDNET2021 Delamination Benchmark
Phase 2B.2: AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Classic 4-stage U-Net architecture for single-channel infrared thermography inputs
and 3-class delamination outputs (Sound, Class 2 Above Rebar, Class 3 Below Rebar).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard U-Net Double Convolution block:
    [Conv2d (3x3) -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Conventional 4-level U-Net for 3-class segmentation.
    Input: (B, 1, H, W) single-channel grayscale thermogram.
    Output: (B, 3, H, W) raw unnormalized logits for:
      Class 0: Sound concrete
      Class 1: Class 2 delamination (above top rebar mat)
      Class 2: Class 3 delamination (below top rebar mat)
    """
    def __init__(self, in_channels=1, num_classes=3, features=[16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        in_ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for f in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(f * 2, f))

        # Final 1x1 classification head
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsample
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsample with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_conn = skip_connections[idx // 2]

            # In case input dimensions are not perfectly divisible
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


def get_unet_model(in_channels=1, num_classes=3, features=[16, 32, 64, 128]):
    """Instantiates the standard U-Net model."""
    return UNet(in_channels=in_channels, num_classes=num_classes, features=features)
