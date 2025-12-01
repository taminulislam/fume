"""
Fast-SCNN: Fast Semantic Segmentation Network
Paper: https://arxiv.org/abs/1902.04502 (BMVC 2019)

Adapted for FUME dual-gas emission analysis
Total Parameters: ~1.1M per encoder (2.8M shared)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Standard convolution block with BN and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise Separable Convolution"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class LinearBottleneck(nn.Module):
    """Linear bottleneck block (inverted residual)"""

    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        layers = []
        # Expansion
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise (linear)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LearningToDownsample(nn.Module):
    """
    Learning to Downsample module
    Fast downsampling to 1/8 resolution
    """

    def __init__(self, in_channels=1, dw_channels=(32, 48), out_channels=64):
        super().__init__()
        self.conv = ConvBlock(in_channels, dw_channels[0], stride=2)  # 1/2
        self.dsconv1 = DSConv(dw_channels[0], dw_channels[1], stride=2)  # 1/4
        self.dsconv2 = DSConv(dw_channels[1], out_channels, stride=2)  # 1/8

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """
    Global Feature Extractor using bottleneck residual blocks
    Efficient context extraction
    """

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, expansion=6, num_blocks=(3, 3, 3)):
        super().__init__()

        self.bottleneck1 = self._make_layer(
            LinearBottleneck, in_channels, block_channels[0],
            num_blocks[0], stride=2, expansion=expansion
        )  # 1/16

        self.bottleneck2 = self._make_layer(
            LinearBottleneck, block_channels[0], block_channels[1],
            num_blocks[1], stride=2, expansion=expansion
        )  # 1/32

        self.bottleneck3 = self._make_layer(
            LinearBottleneck, block_channels[1], block_channels[2],
            num_blocks[2], stride=1, expansion=expansion
        )  # 1/32

        self.ppm = PyramidPoolingModule(block_channels[2], out_channels)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride, expansion):
        layers = [block(in_channels, out_channels, stride, expansion)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, 1, expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class PPMConvBlock(nn.Module):
    """Conv block for PPM without BatchNorm (to handle 1x1 spatial inputs)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for multi-scale context
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Different pooling scales
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = PPMConvBlock(in_channels, in_channels // 4)
        self.conv2 = PPMConvBlock(in_channels, in_channels // 4)
        self.conv3 = PPMConvBlock(in_channels, in_channels // 4)
        self.conv4 = PPMConvBlock(in_channels, in_channels // 4)

        self.out_conv = ConvBlock(in_channels * 2, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]

        feat1 = F.interpolate(self.conv1(self.pool1(x)), size, mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), size, mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), size, mode='bilinear', align_corners=False)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), size, mode='bilinear', align_corners=False)

        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out_conv(x)

        return x


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module
    Fuses low-level (detail) and high-level (context) features
    """

    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.dwconv = DSConv(low_channels, out_channels, stride=1)

        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        # Upsample high-level features
        high_feat = F.interpolate(high_feat, size=low_feat.size()[2:],
                                   mode='bilinear', align_corners=False)
        high_feat = self.conv_high(high_feat)

        # Process low-level features
        low_feat = self.dwconv(low_feat)

        # Fuse
        fused = low_feat + high_feat
        fused = self.relu(fused)

        # Attention refinement
        att = self.attention(fused)
        fused = fused * att

        return fused


class FastSCNNEncoder(nn.Module):
    """
    Fast-SCNN Encoder for FUME

    Returns:
        - low_level_features: 1/8 resolution (for detail)
        - high_level_features: 1/32 resolution (for context)
    """

    def __init__(self, in_channels=1):
        super().__init__()

        # Learning to Downsample (1/8)
        self.learning_to_downsample = LearningToDownsample(
            in_channels=in_channels,
            dw_channels=(32, 48),
            out_channels=64
        )

        # Global Feature Extractor (1/32)
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=(64, 96, 128),
            out_channels=128,
            expansion=6,
            num_blocks=(3, 3, 3)
        )

        # Output channels
        self.low_channels = 64   # From learning_to_downsample
        self.high_channels = 128  # From global_feature_extractor

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input image

        Returns:
            low_level_features: (B, 64, H/8, W/8)
            high_level_features: (B, 128, H/32, W/32)
        """
        low_level_features = self.learning_to_downsample(x)
        high_level_features = self.global_feature_extractor(low_level_features)

        return low_level_features, high_level_features

    def get_output_channels(self):
        """Return output channel dimensions"""
        return {
            'low_channels': self.low_channels,
            'high_channels': self.high_channels
        }


class FastSCNNSegmentationHead(nn.Module):
    """
    Segmentation head for Fast-SCNN
    """

    def __init__(self, low_channels=64, high_channels=128, num_classes=3):
        super().__init__()

        # Feature fusion
        self.feature_fusion = FeatureFusionModule(
            low_channels=low_channels,
            high_channels=high_channels,
            out_channels=128
        )

        # Classifier
        self.classifier = nn.Sequential(
            DSConv(128, 128),
            DSConv(128, 128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, low_feat, high_feat, target_size=(480, 640)):
        """
        Args:
            low_feat: (B, 64, H/8, W/8) low-level features
            high_feat: (B, 128, H/32, W/32) high-level features
            target_size: (H, W) output size

        Returns:
            seg_logits: (B, num_classes, H, W)
        """
        # Fuse features
        fused = self.feature_fusion(low_feat, high_feat)  # (B, 128, H/8, W/8)

        # Classify
        out = self.classifier(fused)  # (B, num_classes, H/8, W/8)

        # Upsample to target size
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

        return out


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Fast-SCNN encoder
    print("="*70)
    print("Fast-SCNN Architecture Test")
    print("="*70)

    # Create encoder
    encoder = FastSCNNEncoder(in_channels=1)

    # Test input
    x = torch.randn(2, 1, 480, 640)

    # Forward pass
    low_feat, high_feat = encoder(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Low-level features: {low_feat.shape}")
    print(f"High-level features: {high_feat.shape}")

    # Count parameters
    params = count_parameters(encoder)
    print(f"\nEncoder parameters: {params:,} ({params/1e6:.2f}M)")

    # Test segmentation head
    seg_head = FastSCNNSegmentationHead(64, 128, num_classes=3)
    seg_out = seg_head(low_feat, high_feat, target_size=(480, 640))

    print(f"Segmentation output: {seg_out.shape}")
    print(f"Seg head parameters: {count_parameters(seg_head):,} ({count_parameters(seg_head)/1e6:.2f}M)")

    print(f"\nTotal parameters: {(params + count_parameters(seg_head)):,} "
          f"({(params + count_parameters(seg_head))/1e6:.2f}M)")

    print("\n" + "="*70)
    print("✅ Fast-SCNN implementation successful!")
    print("="*70)
