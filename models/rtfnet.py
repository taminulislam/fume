"""
RTFNet - RGB-Thermal Fusion Network
Multi-modal fusion for semantic segmentation
Paper: https://arxiv.org/abs/1909.03849

Adapted for CO2-CH4 dual-gas analysis
Parameters: ~4.2M (ResNet18 backbone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """ResNet Basic Block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

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


class ResNetEncoder(nn.Module):
    """Lightweight ResNet18-style encoder"""
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4

        x1 = self.layer1(x)    # 1/4, 64
        x2 = self.layer2(x1)   # 1/8, 128
        x3 = self.layer3(x2)   # 1/16, 256
        x4 = self.layer4(x3)   # 1/32, 512

        return x1, x2, x3, x4


class FusionModule(nn.Module):
    """Multi-level feature fusion"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels * 2, in_channels, 3, 1, 1),
            ConvBNReLU(in_channels, in_channels, 3, 1, 1)
        )

    def forward(self, co2_feat, ch4_feat):
        """Fuse CO2 and CH4 features"""
        fused = torch.cat([co2_feat, ch4_feat], dim=1)
        fused = self.conv(fused)
        return fused


class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels, 3, 1, 1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, skip):
        # Upsample x to match skip
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RTFNet(nn.Module):
    """
    RTFNet - RGB-Thermal Fusion Network for Dual-Gas Analysis

    Architecture:
        Input: CO2 + CH4 frames
        ├── Dual ResNet18 Encoders (CO2 + CH4)
        ├── Multi-level Feature Fusion
        ├── Decoder with Skip Connections
        └── Dual-Task Heads
            ├── Segmentation → CO2 mask + CH4 mask
            └── Classification → pH class

    Parameters: ~4.2M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Encoders
        self.co2_encoder = ResNetEncoder(in_channels=1)
        self.ch4_encoder = ResNetEncoder(in_channels=1)

        # Fusion modules at each level
        self.fusion1 = FusionModule(64)
        self.fusion2 = FusionModule(128)
        self.fusion3 = FusionModule(256)
        self.fusion4 = FusionModule(512)

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)

        # Final segmentation layers
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
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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

        # Encode both modalities
        co2_feats = self.co2_encoder(co2_frame)  # [x1, x2, x3, x4]
        ch4_feats = self.ch4_encoder(ch4_frame)

        # Apply modality masking to each level
        co2_mask = modality_mask[:, 0:1, None, None]
        ch4_mask = modality_mask[:, 1:2, None, None]

        co2_feats = [f * co2_mask for f in co2_feats]
        ch4_feats = [f * ch4_mask for f in ch4_feats]

        # Multi-level fusion
        fused1 = self.fusion1(co2_feats[0], ch4_feats[0])  # 64 channels
        fused2 = self.fusion2(co2_feats[1], ch4_feats[1])  # 128 channels
        fused3 = self.fusion3(co2_feats[2], ch4_feats[2])  # 256 channels
        fused4 = self.fusion4(co2_feats[3], ch4_feats[3])  # 512 channels

        # Classification from deepest features
        cls_logits = self.cls_head(fused4)

        # Decoder with skip connections
        d4 = self.decoder4(fused4, fused3)  # 256
        d3 = self.decoder3(d4, fused2)       # 128
        d2 = self.decoder2(d3, fused1)       # 64

        # Segmentation
        co2_seg = self.co2_seg_head(d2)
        ch4_seg = self.ch4_seg_head(d2)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
