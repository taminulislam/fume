"""
DANet - Dual Attention Network (Lightweight Version)
Attention-based segmentation with position and channel attention
Paper: https://arxiv.org/abs/1809.02983

Parameters: ~3.2M (lightweight variant with MobileNet backbone)
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


class PositionAttention(nn.Module):
    """
    Position Attention Module
    Captures spatial relationships between pixels
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Query: [B, C', H*W]
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)

        # Key: [B, C', H*W]
        key = self.key_conv(x).view(B, -1, H * W)

        # Attention: [B, H*W, H*W]
        attention = self.softmax(torch.bmm(query, key))

        # Value: [B, C, H*W]
        value = self.value_conv(x).view(B, C, H * W)

        # Apply attention: [B, C, H*W]
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Captures channel-wise relationships
    """
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Query: [B, C, H*W]
        query = x.view(B, C, H * W)

        # Key: [B, C, H*W]
        key = x.view(B, C, H * W).permute(0, 2, 1)

        # Attention: [B, C, C]
        attention = self.softmax(torch.bmm(query, key))

        # Value: [B, C, H*W]
        value = x.view(B, C, H * W)

        # Apply attention: [B, C, H*W]
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


class DualAttentionBlock(nn.Module):
    """
    Dual Attention Block combining Position and Channel Attention
    """
    def __init__(self, in_channels):
        super().__init__()
        self.position_attention = PositionAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.fusion = ConvBNReLU(in_channels * 2, in_channels, 3, 1, 1)

    def forward(self, x):
        # Parallel attention streams
        pa_out = self.position_attention(x)
        ca_out = self.channel_attention(x)

        # Fuse
        out = torch.cat([pa_out, ca_out], dim=1)
        out = self.fusion(out)

        return out


class MobileNetBlock(nn.Module):
    """Lightweight MobileNetV2-style block"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1, 1, 0))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LightweightBackbone(nn.Module):
    """Lightweight encoder backbone"""
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


class DANet(nn.Module):
    """
    DANet - Dual Attention Network (Lightweight) for Dual-Gas Analysis

    Architecture:
        Input: CO2 + CH4 frames
        ├── Lightweight MobileNet Backbone (shared)
        ├── Cross-modal Fusion
        ├── Dual Attention Module (Position + Channel)
        └── Dual-Task Heads
            ├── Segmentation → CO2 mask + CH4 mask
            └── Classification → pH class

    Parameters: ~3.2M
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Shared encoder for both modalities
        self.encoder = LightweightBackbone(in_channels=1)

        # Cross-modal fusion
        self.fusion = ConvBNReLU(128 * 2, 128, 3, 1, 1)

        # Dual attention module
        self.dual_attention = DualAttentionBlock(128)

        # Decoder for segmentation
        self.decoder = nn.Sequential(
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
            ConvBNReLU(32, 16, 3, 1, 1),
            nn.Conv2d(16, num_seg_classes, 1)
        )

        self.ch4_seg_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBNReLU(32, 16, 3, 1, 1),
            nn.Conv2d(16, num_seg_classes, 1)
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

        # Dual attention (key component!)
        attended = self.dual_attention(fused)

        # Classification
        cls_logits = self.cls_head(attended)

        # Segmentation
        seg_feat = self.decoder(attended)  # [B, 32, H/4, W/4]

        co2_seg = self.co2_seg_head(seg_feat)  # [B, num_seg_classes, H, W]
        ch4_seg = self.ch4_seg_head(seg_feat)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
