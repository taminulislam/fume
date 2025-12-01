"""
Task-specific heads for FUME
- Segmentation Head (for gas mask prediction)
- Classification Head (for pH class prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Segmentation head with decoder for pixel-level prediction
    Uses simple decoder with skip connections
    """

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 3,
        decoder_channels: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # Final prediction
        self.final_conv = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, target_size: tuple = (480, 640)) -> torch.Tensor:
        """
        Args:
            features: (B, C, H', W') encoded features
            target_size: (H, W) output size

        Returns:
            seg_logits: (B, num_classes, H, W)
        """
        x = self.decoder(features)

        # Upsample to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # Final prediction
        seg_logits = self.final_conv(x)

        return seg_logits


class DeepLabV3PlusHead(nn.Module):
    """
    DeepLabV3+ style segmentation head with ASPP
    More sophisticated decoder for better boundary precision
    """

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 3,
        low_level_channels: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes

        # ASPP Module
        self.aspp = ASPP(in_channels, out_channels=256)

        # Decoder
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(
        self,
        high_level_features: torch.Tensor,
        low_level_features: torch.Tensor,
        target_size: tuple = (480, 640)
    ) -> torch.Tensor:
        """
        Args:
            high_level_features: (B, C_high, H/32, W/32) from encoder layer4
            low_level_features: (B, C_low, H/4, W/4) from encoder layer1
            target_size: (H, W) output size

        Returns:
            seg_logits: (B, num_classes, H, W)
        """
        # ASPP on high-level features
        x = self.aspp(high_level_features)

        # Upsample to match low-level features size
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)

        # Process low-level features
        low_level = self.decoder_conv1(low_level_features)

        # Concatenate
        x = torch.cat([x, low_level], dim=1)

        # Decoder
        x = self.decoder_conv2(x)

        # Upsample to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # Final prediction
        seg_logits = self.final_conv(x)

        return seg_logits


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()

        # 1x1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 conv with different dilation rates
        self.conv2 = self._make_aspp_layer(in_channels, out_channels, dilation=6)
        self.conv3 = self._make_aspp_layer(in_channels, out_channels, dilation=12)
        self.conv4 = self._make_aspp_layer(in_channels, out_channels, dilation=18)

        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_aspp_layer(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.fusion(x)

        return x


class ClassificationHead(nn.Module):
    """
    Classification head for pH class prediction
    Uses global pooling + fully connected layers
    """

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) encoded features

        Returns:
            cls_logits: (B, num_classes)
        """
        # Global pooling
        x = self.global_pool(features)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # Classification
        cls_logits = self.classifier(x)

        return cls_logits
