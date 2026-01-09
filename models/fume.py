

import torch
import torch.nn as nn
from typing import Dict
from .backbones import ResNet50Encoder
from .attention import DualStreamFusion
from .heads import DeepLabV3PlusHead, ClassificationHead


class FUMEModel(nn.Module):
   

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3,
        pretrained: bool = True,
        use_deeplabv3plus: bool = True
    ):
        
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


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Use same encoder for both streams
        self.ch4_encoder = self.co2_encoder  # Weight sharing


# Variant: Without cross-attention (ablation)
class FUMEModelNoAttention(nn.Module):


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
