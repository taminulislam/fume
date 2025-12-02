"""
CMX - Cross-Modal Fusion Transformer
Multi-modal fusion SOTA for RGB-Thermal semantic segmentation
Paper: https://arxiv.org/abs/2203.04838

Adapted for CO2-CH4 dual-gas analysis
Parameters: ~3.8M (lightweight version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Convolution"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    def forward(self, x):
        return self.dwconv(x)


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for fusing CO2 and CH4 features
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, C, H, W] - CO2 features
            x2: [B, C, H, W] - CH4 features
        Returns:
            fused features
        """
        B, C, H, W = x1.shape

        # Q from x1, K,V from x2 (cross-modal)
        q = self.q(x1).reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        k = self.k(x2).reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = self.v(x2).reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        # Attention
        attn = (q @ k) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class CMXBlock(nn.Module):
    """CMX Transformer Block with Cross-Modal Fusion"""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = CrossModalAttention(dim, num_heads)

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x1, x2):
        """
        Args:
            x1: CO2 features
            x2: CH4 features
        """
        # Cross-modal attention
        x1 = x1 + self.attn(self.norm1(x1), self.norm1(x2))

        # Feed-forward
        x1 = x1 + self.mlp(self.norm2(x1))

        return x1


class LightweightEncoder(nn.Module):
    """Lightweight CNN encoder"""
    def __init__(self, in_channels=1):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 2, 1),
            ConvBNReLU(32, 32, 3, 1, 1)
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNReLU(32, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 2, 1),
            ConvBNReLU(256, 256, 3, 1, 1)
        )

    def forward(self, x):
        x = self.stem(x)    # 1/2, 32
        x = self.stage1(x)  # 1/4, 64
        x = self.stage2(x)  # 1/8, 128
        x = self.stage3(x)  # 1/16, 256
        return x


class CMX(nn.Module):
    """
    CMX - Cross-Modal Fusion Transformer for Dual-Gas Analysis

    Architecture:
        Input: CO2 + CH4 frames
        ├── Lightweight CNN Encoders (separate)
        ├── Cross-Modal Transformer Blocks (fusion)
        └── Dual-Task Heads
            ├── Segmentation → CO2 mask + CH4 mask
            └── Classification → pH class

    Parameters: ~3.8M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Encoders (separate for each modality)
        self.co2_encoder = LightweightEncoder(in_channels=1)
        self.ch4_encoder = LightweightEncoder(in_channels=1)

        # Cross-modal fusion blocks
        self.cmx_blocks = nn.ModuleList([
            CMXBlock(dim=256, num_heads=4),
            CMXBlock(dim=256, num_heads=4)
        ])

        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Segmentation heads
        self.co2_seg_head = nn.Sequential(
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, num_seg_classes, 1)
        )

        self.ch4_seg_head = nn.Sequential(
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, num_seg_classes, 1)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
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
        co2_feat = self.co2_encoder(co2_frame)  # [B, 256, H/16, W/16]
        ch4_feat = self.ch4_encoder(ch4_frame)  # [B, 256, H/16, W/16]

        # Apply modality masking
        co2_mask = modality_mask[:, 0:1, None, None]
        ch4_mask = modality_mask[:, 1:2, None, None]
        co2_feat = co2_feat * co2_mask
        ch4_feat = ch4_feat * ch4_mask

        # Cross-modal fusion
        fused_co2 = co2_feat
        fused_ch4 = ch4_feat

        for block in self.cmx_blocks:
            fused_co2 = block(fused_co2, fused_ch4)
            fused_ch4 = block(fused_ch4, fused_co2)

        # Average fusion
        fused = (fused_co2 + fused_ch4) / 2

        # Classification
        cls_logits = self.cls_head(fused)

        # Segmentation
        seg_feat = self.decoder(fused)  # [B, 64, H/4, W/4]

        co2_seg = self.co2_seg_head(seg_feat)
        co2_seg = F.interpolate(co2_seg, (H, W), mode='bilinear', align_corners=False)

        ch4_seg = self.ch4_seg_head(seg_feat)
        ch4_seg = F.interpolate(ch4_seg, (H, W), mode='bilinear', align_corners=False)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
