"""
ENet - Efficient Neural Network
Ultra-lightweight segmentation model for real-time applications
Paper: https://arxiv.org/abs/1606.02147

Parameters: ~0.4M (extremely lightweight!)
Designed for edge devices and real-time inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class InitialBlock(nn.Module):
    """ENet initial block - parallel conv and maxpool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, 2, 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.maxpool(x)
        out = torch.cat([conv_out, pool_out], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class BottleneckBlock(nn.Module):
    """ENet bottleneck module"""
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3,
                 padding=1, dilation=1, asymmetric=False, downsample=False, dropout_prob=0.1):
        super().__init__()

        internal_channels = in_channels // internal_ratio
        self.downsample = downsample

        # Main branch
        # 1x1 projection
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, internal_channels, 2, 2, bias=False)
            self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        else:
            self.conv1 = nn.Conv2d(in_channels, internal_channels, 1, 1, bias=False)
            self.pool = None

        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        # Main convolution
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1),
                         padding=(padding, 0), dilation=dilation, bias=False),
                nn.Conv2d(internal_channels, internal_channels, (1, kernel_size),
                         padding=(0, padding), dilation=dilation, bias=False)
            )
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size,
                                  padding=padding, dilation=dilation, bias=False)

        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()

        # 1x1 expansion
        self.conv3 = nn.Conv2d(internal_channels, out_channels, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        self.prelu_out = nn.PReLU()

        # Skip connection
        if downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 2, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
        elif in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # Skip connection
        if self.shortcut is not None:
            identity = self.shortcut(identity)
            identity = self.bn_shortcut(identity)
        elif self.downsample:
            # Use max pooling for identity when no explicit shortcut
            identity = F.max_pool2d(identity, 2, 2)
            # Pad channels if needed
            pad_size = out.size(1) - identity.size(1)
            if pad_size > 0:
                identity = F.pad(identity, (0, 0, 0, 0, 0, pad_size))

        out = out + identity
        out = self.prelu_out(out)

        return out


class UpsamplingBlock(nn.Module):
    """ENet upsampling bottleneck"""
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        # Main branch
        self.conv1 = nn.Conv2d(in_channels, internal_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        self.deconv = nn.ConvTranspose2d(internal_channels, internal_channels, 3, 2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(internal_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None

        # Skip connection
        self.unpool = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False)
        self.bn_unpool = nn.BatchNorm2d(out_channels)

        self.prelu_out = nn.PReLU()

    def forward(self, x):
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.deconv(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # Skip connection
        identity = self.unpool(x)
        identity = self.bn_unpool(identity)

        out = out + identity
        out = self.prelu_out(out)

        return out


class ENet(nn.Module):
    """
    ENet for Dual-Gas Emission Analysis

    Ultra-lightweight architecture with asymmetric encoder-decoder

    Architecture:
        Input: CO2 + CH4 frames (early fusion)
        ├── Encoder (bottleneck blocks with downsampling)
        ├── Decoder (upsampling blocks)
        └── Dual-Task Heads
            ├── Segmentation → CO2 mask + CH4 mask
            └── Classification → pH class

    Parameters: ~0.4M (smallest model!)
    """

    def __init__(self, num_classes=3, num_seg_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Initial block
        self.initial = InitialBlock(1, 16)  # Changed to 1 channel input

        # Stage 1 - Encoder
        self.bottleneck1_0 = BottleneckBlock(16, 64, downsample=True, dropout_prob=0.01)
        self.bottleneck1_1 = BottleneckBlock(64, 64, dropout_prob=0.01)
        self.bottleneck1_2 = BottleneckBlock(64, 64, dropout_prob=0.01)
        self.bottleneck1_3 = BottleneckBlock(64, 64, dropout_prob=0.01)
        self.bottleneck1_4 = BottleneckBlock(64, 64, dropout_prob=0.01)

        # Stage 2 - Encoder
        self.bottleneck2_0 = BottleneckBlock(64, 128, downsample=True, dropout_prob=0.1)
        self.bottleneck2_1 = BottleneckBlock(128, 128, dropout_prob=0.1)
        self.bottleneck2_2 = BottleneckBlock(128, 128, dilation=2, padding=2, dropout_prob=0.1)
        self.bottleneck2_3 = BottleneckBlock(128, 128, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_4 = BottleneckBlock(128, 128, dilation=4, padding=4, dropout_prob=0.1)
        self.bottleneck2_5 = BottleneckBlock(128, 128, dropout_prob=0.1)
        self.bottleneck2_6 = BottleneckBlock(128, 128, dilation=8, padding=8, dropout_prob=0.1)
        self.bottleneck2_7 = BottleneckBlock(128, 128, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_8 = BottleneckBlock(128, 128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 3 - Decoder
        self.upsample3_0 = UpsamplingBlock(128, 64, dropout_prob=0.1)
        self.bottleneck3_1 = BottleneckBlock(64, 64, dropout_prob=0.1)
        self.bottleneck3_2 = BottleneckBlock(64, 64, dropout_prob=0.1)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBlock(64, 16, dropout_prob=0.1)
        self.bottleneck4_1 = BottleneckBlock(16, 16, dropout_prob=0.1)

        # Final upsampling
        self.final_upsample = nn.ConvTranspose2d(16, 16, 2, 2)

        # Segmentation heads
        self.co2_seg_head = nn.Conv2d(16, num_seg_classes, 1)
        self.ch4_seg_head = nn.Conv2d(16, num_seg_classes, 1)

        # Classification head (from encoder features)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
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

        # Early fusion of modalities
        x = (co2_frame + ch4_frame) / 2

        # Encoder
        x = self.initial(x)  # 1/2

        # Stage 1
        x = self.bottleneck1_0(x)  # 1/4, 64
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # Stage 2
        x = self.bottleneck2_0(x)  # 1/8, 128
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        encoder_out = self.bottleneck2_8(x)

        # Classification from encoder
        cls_logits = self.cls_head(encoder_out)

        # Decoder
        # Stage 3
        x = self.upsample3_0(encoder_out)  # 1/4, 64
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)

        # Stage 4
        x = self.upsample4_0(x)  # 1/2, 16
        x = self.bottleneck4_1(x)

        # Final upsampling
        x = self.final_upsample(x)  # Full resolution, 16

        # Segmentation
        co2_seg = self.co2_seg_head(x)
        ch4_seg = self.ch4_seg_head(x)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg,
            'ch4_seg_logits': ch4_seg
        }

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
