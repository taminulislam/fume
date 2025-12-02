"""
DDRNet-23-Slim - Deep Dual-Resolution Network
SOTA lightweight segmentation model
Paper: https://arxiv.org/abs/2101.06085

Parameters: ~5.7M (Slim version)
Optimized for real-time segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class BasicBlock(nn.Module):
    """Basic residual block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module"""
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        )

        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        )

        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(9),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        )

        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(17),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        )

        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(33),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        )

        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(inter_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels * 4, inter_channels, 3, 1, 1, bias=False)
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(inter_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels * 2, out_channels, 1, bias=False)
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]

        scale0 = self.scale0(x)

        scale1 = self.scale1(x)
        scale1 = F.interpolate(scale1, size=(height, width), mode='bilinear', align_corners=False)

        scale2 = self.scale2(x)
        scale2 = F.interpolate(scale2, size=(height, width), mode='bilinear', align_corners=False)

        scale3 = self.scale3(x)
        scale3 = F.interpolate(scale3, size=(height, width), mode='bilinear', align_corners=False)

        scale4 = self.scale4(x)
        scale4 = F.interpolate(scale4, size=(height, width), mode='bilinear', align_corners=False)

        scale_out = self.scale_process(torch.cat([scale1, scale2, scale3, scale4], 1))

        out = self.compression(torch.cat([scale0, scale_out], 1)) + self.shortcut(x)

        return out


class DDRNetSlim(nn.Module):
    """
    DDRNet-23-Slim for Dual-Gas Emission Analysis

    Dual-resolution architecture:
    - High-res branch: preserves spatial details
    - Low-res branch: captures semantic context
    - Bilateral fusion between branches

    Parameters: ~5.7M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # High-res branch (1/4 resolution)
        self.high_layer1 = self._make_layer(32, 32, 2)
        self.high_layer2 = self._make_layer(32, 64, 2, stride=2)

        # Low-res branch (1/8 -> 1/32 resolution)
        self.low_layer1 = self._make_layer(32, 64, 2, stride=2)
        self.low_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.low_layer3 = self._make_layer(128, 256, 2, stride=2)

        # Compression layers for fusion
        self.compress_high = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.compress_low = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # DAPPM for context aggregation
        self.dappm = DAPPM(256, 64, 128)

        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(32 + 128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Segmentation heads
        self.co2_seg_head = nn.Conv2d(64, num_seg_classes, 1)
        self.ch4_seg_head = nn.Conv2d(64, num_seg_classes, 1)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

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

        # Simple fusion of input modalities
        x = (co2_frame + ch4_frame) / 2  # Early fusion

        # Stem
        x = self.stem(x)  # [B, 32, H/2, W/2]

        # Dual-resolution branches
        # High-res branch (spatial details)
        high = self.high_layer1(x)  # [B, 32, H/2, W/2]
        high = self.high_layer2(high)  # [B, 64, H/4, W/4]

        # Low-res branch (semantic context)
        low = self.low_layer1(x)  # [B, 64, H/4, W/4]
        low = self.low_layer2(low)  # [B, 128, H/8, W/8]
        low = self.low_layer3(low)  # [B, 256, H/16, W/16]

        # Context aggregation
        low = self.dappm(low)  # [B, 128, H/16, W/16]

        # Classification
        cls_logits = self.cls_head(low)

        # Upsample low-res to match high-res
        low_up = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)

        # Compress and fuse
        high_comp = self.compress_high(high)  # [B, 32, H/4, W/4]
        low_comp = self.compress_low(low_up)  # [B, 32, H/4, W/4]

        # Fused features
        fused = torch.cat([high_comp, low_up], dim=1)  # [B, 32+128, H/4, W/4]
        fused = self.seg_decoder(fused)  # [B, 64, H/4, W/4]

        # Segmentation
        co2_seg = self.co2_seg_head(fused)
        co2_seg = F.interpolate(co2_seg, (H, W), mode='bilinear', align_corners=False)

        ch4_seg = self.ch4_seg_head(fused)
        ch4_seg = F.interpolate(ch4_seg, (H, W), mode='bilinear', align_corners=False)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
