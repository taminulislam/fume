"""
BiSeNetV2 - Bilateral Segmentation Network V2
Lightweight semantic segmentation with dual-path architecture
Paper: https://arxiv.org/abs/2004.02147

Parameters: ~3.4M (Detail Path + Semantic Path)
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


class DetailBranch(nn.Module):
    """Detail Branch - capture low-level spatial details"""
    def __init__(self, in_channels=1):
        super().__init__()
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )

    def forward(self, x):
        x = self.stage1(x)  # 1/2
        x = self.stage2(x)  # 1/4
        x = self.stage3(x)  # 1/8
        return x


class StemBlock(nn.Module):
    """Stem block for Semantic Branch"""
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, 2, 1)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, out_channels//2, 1, 1, 0),
            ConvBNReLU(out_channels//2, out_channels, 3, 2, 1)
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_channels*2, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        x = self.fuse(x)
        return x


class GatherExpansion(nn.Module):
    """Gather-and-Expansion Layer"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        mid_channels = in_channels * expansion
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3, 1, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        return self.relu(x + identity)


class SemanticBranch(nn.Module):
    """Semantic Branch - capture high-level context"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = StemBlock(in_channels, 16)

        # Stage 3
        self.stage3 = nn.Sequential(
            GatherExpansion(16, 32, 2),
            GatherExpansion(32, 32, 1)
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            GatherExpansion(32, 64, 2),
            GatherExpansion(64, 64, 1)
        )

        # Stage 5
        self.stage5 = nn.Sequential(
            GatherExpansion(64, 128, 2),
            GatherExpansion(128, 128, 1),
            GatherExpansion(128, 128, 1),
            GatherExpansion(128, 128, 1)
        )

        # Context Embedding
        self.ce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)    # 1/4
        x = self.stage3(x)  # 1/8
        x = self.stage4(x)  # 1/16
        x = self.stage5(x)  # 1/32

        # Context embedding
        ce = self.ce(x)
        ce = F.interpolate(ce, x.shape[2:], mode='bilinear', align_corners=False)
        x = x + ce

        return x


class AggregationLayer(nn.Module):
    """Bilateral Guided Aggregation Layer"""
    def __init__(self, detail_channels=128, semantic_channels=128, out_channels=128):
        super().__init__()

        # Detail branch
        self.detail_dwconv = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, 1, 1, groups=detail_channels, bias=False),
            nn.BatchNorm2d(detail_channels)
        )
        self.detail_conv = nn.Conv2d(detail_channels, out_channels, 1, bias=False)

        # Semantic branch
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.semantic_dwconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv = ConvBNReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, detail, semantic):
        # Upsample semantic to match detail
        semantic = F.interpolate(semantic, detail.shape[2:], mode='bilinear', align_corners=False)

        # Detail branch
        detail_dwconv = self.detail_dwconv(detail)
        detail_conv = self.detail_conv(detail)
        detail_out = detail_conv + detail_dwconv

        # Semantic branch
        semantic_conv = self.semantic_conv(semantic)
        semantic_dwconv = self.semantic_dwconv(semantic_conv)
        semantic_out = semantic_conv + semantic_dwconv

        # Fusion
        out = detail_out + semantic_out
        out = self.conv(out)

        return out


class BiSeNetV2(nn.Module):
    """
    BiSeNetV2 for Dual-Gas Emission Analysis

    Architecture:
        Input: CO2 + CH4 frames (dual-stream)
        ├── Detail Branch (low-level features)
        ├── Semantic Branch (high-level context)
        └── Aggregation Layer (fusion)
            ↓
        Dual-Task Heads:
        ├── Segmentation → CO2 mask + CH4 mask
        └── Classification → pH class

    Parameters: ~3.4M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # CO2 stream
        self.co2_detail = DetailBranch(in_channels=1)
        self.co2_semantic = SemanticBranch(in_channels=1)

        # CH4 stream (shared weights to save parameters)
        self.ch4_detail = self.co2_detail
        self.ch4_semantic = self.co2_semantic

        # Aggregation
        self.co2_agg = AggregationLayer(128, 128, 128)
        self.ch4_agg = AggregationLayer(128, 128, 128)

        # Cross-modal fusion
        self.fusion = nn.Sequential(
            ConvBNReLU(256, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )

        # Segmentation heads
        self.co2_seg_head = nn.Sequential(
            ConvBNReLU(128, 64, 3, 1, 1),
            nn.Conv2d(64, num_seg_classes, 1)
        )

        self.ch4_seg_head = nn.Sequential(
            ConvBNReLU(128, 64, 3, 1, 1),
            nn.Conv2d(64, num_seg_classes, 1)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, co2_frame, ch4_frame, modality_mask):
        """
        Args:
            co2_frame: [B, 1, H, W]
            ch4_frame: [B, 1, H, W]
            modality_mask: [B, 2] - availability mask

        Returns:
            dict: {
                'cls_logits': [B, num_classes],
                'co2_seg_logits': [B, num_seg_classes, H, W],
                'ch4_seg_logits': [B, num_seg_classes, H, W]
            }
        """
        B, _, H, W = co2_frame.shape

        # CO2 stream
        co2_detail = self.co2_detail(co2_frame)
        co2_semantic = self.co2_semantic(co2_frame)
        co2_fused = self.co2_agg(co2_detail, co2_semantic)

        # CH4 stream
        ch4_detail = self.ch4_detail(ch4_frame)
        ch4_semantic = self.ch4_semantic(ch4_frame)
        ch4_fused = self.ch4_agg(ch4_detail, ch4_semantic)

        # Cross-modal fusion
        fused = torch.cat([co2_fused, ch4_fused], dim=1)
        fused = self.fusion(fused)

        # Apply modality masking
        co2_mask = modality_mask[:, 0:1, None, None]
        ch4_mask = modality_mask[:, 1:2, None, None]
        co2_fused = co2_fused * co2_mask
        ch4_fused = ch4_fused * ch4_mask

        # Segmentation
        co2_seg = self.co2_seg_head(co2_fused)
        co2_seg = F.interpolate(co2_seg, (H, W), mode='bilinear', align_corners=False)

        ch4_seg = self.ch4_seg_head(ch4_fused)
        ch4_seg = F.interpolate(ch4_seg, (H, W), mode='bilinear', align_corners=False)

        # Classification
        cls_logits = self.cls_head(fused)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
