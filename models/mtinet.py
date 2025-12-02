"""
MTI-Net - Multi-Task Interaction Network
Multi-task learning with task interaction
Paper: Multi-Task Learning approach with task interactions

Parameters: ~3.5M
Designed for joint segmentation and classification
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


class TaskInteractionModule(nn.Module):
    """
    Task Interaction Module
    Enables information flow between segmentation and classification tasks
    """
    def __init__(self, channels):
        super().__init__()

        # Seg to Cls branch
        self.seg_to_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # Cls to Seg branch
        self.cls_to_seg = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.fusion = ConvBNReLU(channels * 2, channels, 1, 1, 0)

    def forward(self, seg_feat, cls_feat):
        """
        Args:
            seg_feat: Segmentation features [B, C, H, W]
            cls_feat: Classification features [B, C, H, W]

        Returns:
            Enhanced segmentation and classification features
        """
        # Seg -> Cls attention
        seg_attention = self.seg_to_cls(seg_feat)
        cls_enhanced = cls_feat * seg_attention

        # Cls -> Seg attention
        cls_attention = self.cls_to_seg(cls_feat)
        seg_enhanced = seg_feat * cls_attention

        # Fuse
        seg_out = self.fusion(torch.cat([seg_feat, seg_enhanced], dim=1))
        cls_out = self.fusion(torch.cat([cls_feat, cls_enhanced], dim=1))

        return seg_out, cls_out


class MobileNetBlock(nn.Module):
    """MobileNetV2-style inverted residual block"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)

        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1, 1, 0))

        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Pointwise projection
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SharedEncoder(nn.Module):
    """Shared encoder for both modalities"""
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 2, 1),
            MobileNetBlock(32, 16, 1, expand_ratio=1)
        )

        # Stages
        self.stage1 = nn.Sequential(
            MobileNetBlock(16, 24, 2, expand_ratio=6),
            MobileNetBlock(24, 24, 1, expand_ratio=6)
        )

        self.stage2 = nn.Sequential(
            MobileNetBlock(24, 32, 2, expand_ratio=6),
            MobileNetBlock(32, 32, 1, expand_ratio=6),
            MobileNetBlock(32, 32, 1, expand_ratio=6)
        )

        self.stage3 = nn.Sequential(
            MobileNetBlock(32, 64, 2, expand_ratio=6),
            MobileNetBlock(64, 64, 1, expand_ratio=6),
            MobileNetBlock(64, 64, 1, expand_ratio=6),
            MobileNetBlock(64, 64, 1, expand_ratio=6)
        )

        self.stage4 = nn.Sequential(
            MobileNetBlock(64, 128, 2, expand_ratio=6),
            MobileNetBlock(128, 128, 1, expand_ratio=6),
            MobileNetBlock(128, 128, 1, expand_ratio=6)
        )

    def forward(self, x):
        x = self.stem(x)      # 1/2, 16
        x1 = self.stage1(x)   # 1/4, 24
        x2 = self.stage2(x1)  # 1/8, 32
        x3 = self.stage3(x2)  # 1/16, 64
        x4 = self.stage4(x3)  # 1/32, 128
        return x1, x2, x3, x4


class MTINet(nn.Module):
    """
    MTI-Net - Multi-Task Interaction Network for Dual-Gas Analysis

    Architecture:
        Input: CO2 + CH4 frames
        ├── Shared Encoder (MobileNetV2-style)
        ├── Task-Specific Branches
        │   ├── Segmentation Branch
        │   └── Classification Branch
        ├── Task Interaction Modules (bidirectional information flow)
        └── Dual-Task Heads
            ├── Segmentation → CO2 mask + CH4 mask
            └── Classification → pH class

    Parameters: ~3.5M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Shared encoder for both modalities
        self.encoder = SharedEncoder(in_channels=1)

        # Cross-modal fusion
        self.fusion = ConvBNReLU(128 * 2, 128, 3, 1, 1)

        # Task-specific feature extraction
        self.seg_branch = nn.Sequential(
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )

        self.cls_branch = nn.Sequential(
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )

        # Task interaction module
        self.task_interaction = TaskInteractionModule(128)

        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(32, 32, 3, 1, 1)
        )

        # Segmentation heads
        self.co2_seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_seg_classes, 1)
        )

        self.ch4_seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_seg_classes, 1)
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
        co2_feats = self.encoder(co2_frame)  # [x1, x2, x3, x4]
        ch4_feats = self.encoder(ch4_frame)

        # Apply modality masking to deepest features
        co2_mask = modality_mask[:, 0:1, None, None]
        ch4_mask = modality_mask[:, 1:2, None, None]

        co2_deep = co2_feats[3] * co2_mask  # [B, 128, H/32, W/32]
        ch4_deep = ch4_feats[3] * ch4_mask

        # Cross-modal fusion
        fused = torch.cat([co2_deep, ch4_deep], dim=1)
        fused = self.fusion(fused)  # [B, 128, H/32, W/32]

        # Task-specific branches
        seg_feat = self.seg_branch(fused)
        cls_feat = self.cls_branch(fused)

        # Task interaction (key innovation!)
        seg_feat, cls_feat = self.task_interaction(seg_feat, cls_feat)

        # Classification
        cls_logits = self.cls_head(cls_feat)

        # Segmentation
        seg_out = self.seg_decoder(seg_feat)  # [B, 32, H/4, W/4]

        co2_seg = self.co2_seg_head(seg_out)  # [B, num_seg_classes, H, W]
        ch4_seg = self.ch4_seg_head(seg_out)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
