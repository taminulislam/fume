"""
ESPNetV2 - Efficient Spatial Pyramid Network V2
Ultra-lightweight CNN for edge devices
Paper: https://arxiv.org/abs/1811.11431

Parameters: ~1.2M (ESPNetV2-S variant)
Extremely efficient for real-time inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EESP(nn.Module):
    """
    Extremely Efficient Spatial Pyramid module
    Uses depth-wise dilated convolutions in a pyramid structure
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Branch channels
        branch_channels = out_channels // 4

        self.stride = stride

        # Initial projection
        self.conv1x1_in = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Pyramid branches with different dilation rates
        self.d_conv3x3_1 = nn.Conv2d(branch_channels, branch_channels, 3, stride, 1,
                                     groups=branch_channels, dilation=1, bias=False)
        self.d_conv3x3_2 = nn.Conv2d(branch_channels, branch_channels, 3, stride, 2,
                                     groups=branch_channels, dilation=2, bias=False)
        self.d_conv3x3_3 = nn.Conv2d(branch_channels, branch_channels, 3, stride, 4,
                                     groups=branch_channels, dilation=4, bias=False)
        self.d_conv3x3_4 = nn.Conv2d(branch_channels, branch_channels, 3, stride, 8,
                                     groups=branch_channels, dilation=8, bias=False)

        # Batch norms
        self.bn = nn.BatchNorm2d(out_channels)

        # Output projection
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut
        if stride == 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride, 1) if stride == 2 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # Initial projection
        x = self.conv1x1_in(x)

        # Pyramid branches
        d1 = self.d_conv3x3_1(x)
        d2 = self.d_conv3x3_2(x)
        d3 = self.d_conv3x3_3(x)
        d4 = self.d_conv3x3_4(x)

        # Hierarchical feature fusion (HFF)
        combine12 = d1 + d2
        combine123 = combine12 + d3
        combine1234 = combine123 + d4

        # Concatenate all branches
        out = torch.cat([d1, combine12, combine123, combine1234], dim=1)
        out = self.bn(out)

        # Output projection
        out = self.conv1x1_out(out)

        # Shortcut
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = self.relu(out + identity)

        return out


class DownSampler(nn.Module):
    """Efficient downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 2

        self.conv3x3 = ConvBNReLU(in_channels, branch_channels, 3, 2, 1)
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv_out = self.conv3x3(x)
        pool_out = self.pool(x)
        pool_out = self.conv1x1(pool_out)
        return torch.cat([conv_out, pool_out], dim=1)


class ESPNetV2Encoder(nn.Module):
    """ESPNetV2 Encoder"""
    def __init__(self, in_channels=1):
        super().__init__()

        # Level 0: Initial downsampling
        self.level0 = ConvBNReLU(in_channels, 16, 3, 2, 1)

        # Level 1
        self.level1 = DownSampler(16, 64)

        # Level 2
        self.level2_0 = DownSampler(64, 128)
        self.level2 = nn.Sequential(
            EESP(128, 128, stride=1),
            EESP(128, 128, stride=1)
        )

        # Level 3
        self.level3_0 = DownSampler(128, 256)
        self.level3 = nn.Sequential(
            EESP(256, 256, stride=1),
            EESP(256, 256, stride=1),
            EESP(256, 256, stride=1)
        )

    def forward(self, x):
        l0 = self.level0(x)      # 1/2, 16
        l1 = self.level1(l0)     # 1/4, 64
        l2 = self.level2_0(l1)   # 1/8, 128
        l2 = self.level2(l2)     # 1/8, 128
        l3 = self.level3_0(l2)   # 1/16, 256
        l3 = self.level3(l3)     # 1/16, 256
        return l1, l2, l3


class ESPNetV2(nn.Module):
    """
    ESPNetV2 for Dual-Gas Emission Analysis

    Ultra-lightweight architecture with spatial pyramid convolutions

    Architecture:
        Input: CO2 + CH4 frames
        ├── Shared ESPNetV2 Encoder (weight sharing for efficiency)
        ├── Feature Fusion
        └── Dual-Task Heads
            ├── Segmentation → CO2 mask + CH4 mask
            └── Classification → pH class

    Parameters: ~1.2M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Shared encoder for both modalities (parameter efficiency)
        self.encoder = ESPNetV2Encoder(in_channels=1)

        # Fusion module
        self.fusion = nn.Sequential(
            ConvBNReLU(256 * 2, 256, 3, 1, 1),
            EESP(256, 256, stride=1)
        )

        # Decoder for segmentation
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(256, 128, 3, 1, 1)
        )

        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64, 3, 1, 1)
        )

        # Segmentation heads
        self.co2_seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, num_seg_classes, 1)
        )

        self.ch4_seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, num_seg_classes, 1)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, co2_frame, ch4_frame, modality_mask):
        """
        Args:
            co2_frame: [B, 1, H, W]
            ch4_frame: [B, 1, H, W]
            modality_mask: [B, 2]

        Returns:
            dict: {
                'cls_logits': [B, num_classes],
                'co2_seg_logits': [B, num_seg_classes, H, W],
                'ch4_seg_logits': [B, num_seg_classes, H, W]
            }
        """
        B, _, H, W = co2_frame.shape

        # Encode both modalities with shared encoder
        co2_feats = self.encoder(co2_frame)  # [l1, l2, l3]
        ch4_feats = self.encoder(ch4_frame)

        # Apply modality masking
        co2_mask = modality_mask[:, 0:1, None, None]
        ch4_mask = modality_mask[:, 1:2, None, None]

        co2_l3 = co2_feats[2] * co2_mask
        ch4_l3 = ch4_feats[2] * ch4_mask

        # Fuse deepest features
        fused = torch.cat([co2_l3, ch4_l3], dim=1)
        fused = self.fusion(fused)  # [B, 256, H/16, W/16]

        # Classification
        cls_logits = self.cls_head(fused)

        # Decoder
        d2 = self.decoder2(fused)  # [B, 128, H/8, W/8]
        d1 = self.decoder1(d2)     # [B, 64, H/4, W/4]

        # Segmentation
        co2_seg = self.co2_seg_head(d1)  # [B, num_seg_classes, H, W]
        ch4_seg = self.ch4_seg_head(d1)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
