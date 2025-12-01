"""
FUME: Fused Unified Multi-gas Emission Network
Cross-Modal Fusion for Gas Emission Analysis

Main model architecture with dual-stream encoder and cross-modal attention
"""

import torch
import torch.nn as nn
from typing import Dict
from .backbones import ResNet50Encoder
from .attention import DualStreamFusion
from .heads import DeepLabV3PlusHead, ClassificationHead


class FUMEModel(nn.Module):
    """
    FUME: Multi-Modal Fusion Network for Acidosis Detection

    Architecture:
        Input: Paired CO2 and CH4 frames
        ├── CO2 Stream: ResNet-50 Encoder
        └── CH4 Stream: ResNet-50 Encoder
            ↓
        Cross-Modal Attention Fusion
            ↓
        Dual-Task Heads:
        ├── Segmentation Head → CO2 mask + CH4 mask
        └── Classification Head → pH class
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3,
        pretrained: bool = True,
        use_deeplabv3plus: bool = True
    ):
        """
        Args:
            num_classes: Number of pH classes (3: Healthy, Transitional, Acidotic)
            num_seg_classes: Number of segmentation classes (3: background, tube, gas)
            pretrained: Use ImageNet pretrained weights
            use_deeplabv3plus: Use DeepLabV3+ decoder (True) or simple decoder (False)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # Dual-stream encoders (shared weights or separate)
        self.co2_encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)
        self.ch4_encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)

        # Cross-modal fusion module
        self.fusion = DualStreamFusion(in_channels=2048)  # ResNet-50 output channels

        # Task heads
        if use_deeplabv3plus:
            self.co2_seg_head = DeepLabV3PlusHead(
                in_channels=2048,
                num_classes=num_seg_classes,
                low_level_channels=256
            )
            self.ch4_seg_head = DeepLabV3PlusHead(
                in_channels=2048,
                num_classes=num_seg_classes,
                low_level_channels=256
            )
        else:
            from .heads import SegmentationHead
            self.co2_seg_head = SegmentationHead(in_channels=2048, num_classes=num_seg_classes)
            self.ch4_seg_head = SegmentationHead(in_channels=2048, num_classes=num_seg_classes)

        self.cls_head = ClassificationHead(in_channels=2048, num_classes=num_classes)

        self.use_deeplabv3plus = use_deeplabv3plus

    def forward(
        self,
        co2_frame: torch.Tensor,
        ch4_frame: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            co2_frame: (B, 1, H, W) CO2 thermal frames
            ch4_frame: (B, 1, H, W) CH4 thermal frames
            modality_mask: (B, 2) binary mask [use_co2, use_ch4]

        Returns:
            Dictionary with keys:
                - cls_logits: (B, num_classes) classification logits
                - co2_seg_logits: (B, num_seg_classes, H, W) CO2 segmentation
                - ch4_seg_logits: (B, num_seg_classes, H, W) CH4 segmentation
        """
        target_size = (co2_frame.shape[2], co2_frame.shape[3])  # (H, W)

        # Encode both streams
        co2_features = self.co2_encoder(co2_frame)  # List of multi-scale features
        ch4_features = self.ch4_encoder(ch4_frame)

        # Extract features at different scales
        # co2_features = [C0, C1, C2, C3, C4]
        # C1: 256 ch (for DeepLabV3+ low-level features)
        # C4: 2048 ch (for high-level features and classification)

        co2_high = co2_features[-1]  # (B, 2048, H/32, W/32)
        ch4_high = ch4_features[-1]

        # Cross-modal fusion
        fused_features = self.fusion(co2_high, ch4_high, modality_mask)  # (B, 2048, H/32, W/32)

        # Classification head (uses fused features)
        cls_logits = self.cls_head(fused_features)

        # Segmentation heads (separate for each gas type)
        if self.use_deeplabv3plus:
            co2_low = co2_features[1]  # (B, 256, H/4, W/4)
            ch4_low = ch4_features[1]
            co2_seg_logits = self.co2_seg_head(co2_high, co2_low, target_size)
            ch4_seg_logits = self.ch4_seg_head(ch4_high, ch4_low, target_size)
        else:
            co2_seg_logits = self.co2_seg_head(co2_high, target_size)
            ch4_seg_logits = self.ch4_seg_head(ch4_high, target_size)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg_logits,
            'ch4_seg_logits': ch4_seg_logits
        }

    def get_num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Variant: Shared encoder weights
class FUMEModelSharedEncoder(FUMEModel):
    """
    FUME variant with shared encoder weights for both CO2 and CH4 streams
    Reduces parameters but may reduce modality-specific learning
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Use same encoder for both streams
        self.ch4_encoder = self.co2_encoder  # Weight sharing


# Variant: Without cross-attention (ablation)
class FUMEModelNoAttention(nn.Module):
    """
    FUME variant without cross-attention
    For ablation study: simple concatenation fusion
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3,
        pretrained: bool = True
    ):
        super().__init__()

        self.co2_encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)
        self.ch4_encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)

        # Simple fusion (concatenation + 1x1 conv)
        self.fusion = nn.Sequential(
            nn.Conv2d(2048 * 2, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.co2_seg_head = DeepLabV3PlusHead(in_channels=2048, num_classes=num_seg_classes, low_level_channels=256)
        self.ch4_seg_head = DeepLabV3PlusHead(in_channels=2048, num_classes=num_seg_classes, low_level_channels=256)
        self.cls_head = ClassificationHead(in_channels=2048, num_classes=num_classes)

    def forward(self, co2_frame, ch4_frame, modality_mask):
        target_size = (co2_frame.shape[2], co2_frame.shape[3])

        co2_features = self.co2_encoder(co2_frame)
        ch4_features = self.ch4_encoder(ch4_frame)

        co2_high = co2_features[-1]
        ch4_high = ch4_features[-1]

        # Simple concatenation fusion
        fused = torch.cat([co2_high, ch4_high], dim=1)
        fused_features = self.fusion(fused)

        cls_logits = self.cls_head(fused_features)

        co2_low = co2_features[1]
        ch4_low = ch4_features[1]
        co2_seg_logits = self.co2_seg_head(co2_high, co2_low, target_size)
        ch4_seg_logits = self.ch4_seg_head(ch4_high, ch4_low, target_size)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg_logits,
            'ch4_seg_logits': ch4_seg_logits
        }
