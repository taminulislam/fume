"""
FUME with Fast-SCNN Backbone
Fast-SCNN + Cross-Modal Attention for Dual-Gas Emission Analysis

Total Parameters: ~2.8M (with shared encoder)
"""

import torch
import torch.nn as nn
from typing import Dict
from .fastscnn import FastSCNNEncoder, FastSCNNSegmentationHead
from .attention import DualStreamFusion


class FUMEFastSCNN(nn.Module):
    """
    FUME with Fast-SCNN backbone

    Architecture:
        Input: Paired CO2 and CH4 frames
        ├── CO2 Stream: Fast-SCNN Encoder
        └── CH4 Stream: Fast-SCNN Encoder (shared weights)
            ↓
        Cross-Modal Attention Fusion
            ↓
        Dual-Task Heads:
        ├── Segmentation → CO2 mask + CH4 mask
        └── Classification → pH class

    Parameters: ~2.8M (within 3M budget!)
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3,
        shared_encoder: bool = True
    ):
        """
        Args:
            num_classes: Number of pH classes (3: Healthy, Transitional, Acidotic)
            num_seg_classes: Number of segmentation classes (3: background, tube, gas)
            shared_encoder: Share encoder weights between CO2 and CH4 (saves parameters)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.shared_encoder = shared_encoder

        # Encoders
        self.co2_encoder = FastSCNNEncoder(in_channels=1)

        if shared_encoder:
            # Share weights to stay within budget
            self.ch4_encoder = self.co2_encoder
        else:
            # Separate encoder for CH4 (more parameters but better modality-specific learning)
            self.ch4_encoder = FastSCNNEncoder(in_channels=1)

        # Get encoder output channels
        encoder_info = self.co2_encoder.get_output_channels()
        self.low_channels = encoder_info['low_channels']  # 64
        self.high_channels = encoder_info['high_channels']  # 128

        # Cross-modal fusion on high-level features
        self.fusion = DualStreamFusion(in_channels=self.high_channels)

        # Segmentation heads (separate for each gas type)
        self.co2_seg_head = FastSCNNSegmentationHead(
            low_channels=self.low_channels,
            high_channels=self.high_channels,
            num_classes=num_seg_classes
        )

        self.ch4_seg_head = FastSCNNSegmentationHead(
            low_channels=self.low_channels,
            high_channels=self.high_channels,
            num_classes=num_seg_classes
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.high_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

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
        co2_low, co2_high = self.co2_encoder(co2_frame)
        # co2_low: (B, 64, H/8, W/8)
        # co2_high: (B, 128, H/32, W/32)

        ch4_low, ch4_high = self.ch4_encoder(ch4_frame)

        # Cross-modal fusion on high-level features
        fused_high = self.fusion(co2_high, ch4_high, modality_mask)  # (B, 128, H/32, W/32)

        # Classification head (uses fused high-level features)
        cls_logits = self.cls_head(fused_high)

        # Segmentation heads (use original stream-specific features)
        co2_seg_logits = self.co2_seg_head(co2_low, co2_high, target_size)
        ch4_seg_logits = self.ch4_seg_head(ch4_low, ch4_high, target_size)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg_logits,
            'ch4_seg_logits': ch4_seg_logits
        }

    def get_num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FUMEFastSCNNNoAttention(nn.Module):
    """
    FUME Fast-SCNN variant without cross-attention
    For ablation study: simple concatenation fusion
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3,
        shared_encoder: bool = True
    ):
        super().__init__()

        self.co2_encoder = FastSCNNEncoder(in_channels=1)

        if shared_encoder:
            self.ch4_encoder = self.co2_encoder
        else:
            self.ch4_encoder = FastSCNNEncoder(in_channels=1)

        encoder_info = self.co2_encoder.get_output_channels()
        self.high_channels = encoder_info['high_channels']

        # Simple fusion (concatenation + conv)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.high_channels * 2, self.high_channels, 1, bias=False),
            nn.BatchNorm2d(self.high_channels),
            nn.ReLU(inplace=True)
        )

        self.co2_seg_head = FastSCNNSegmentationHead(
            low_channels=encoder_info['low_channels'],
            high_channels=self.high_channels,
            num_classes=num_seg_classes
        )

        self.ch4_seg_head = FastSCNNSegmentationHead(
            low_channels=encoder_info['low_channels'],
            high_channels=self.high_channels,
            num_classes=num_seg_classes
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.high_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, co2_frame, ch4_frame, modality_mask):
        target_size = (co2_frame.shape[2], co2_frame.shape[3])

        co2_low, co2_high = self.co2_encoder(co2_frame)
        ch4_low, ch4_high = self.ch4_encoder(ch4_frame)

        # Simple concatenation fusion
        fused = torch.cat([co2_high, ch4_high], dim=1)
        fused_high = self.fusion(fused)

        cls_logits = self.cls_head(fused_high)
        co2_seg_logits = self.co2_seg_head(co2_low, co2_high, target_size)
        ch4_seg_logits = self.ch4_seg_head(ch4_low, ch4_high, target_size)

        return {
            'cls_logits': cls_logits,
            'co2_seg_logits': co2_seg_logits,
            'ch4_seg_logits': ch4_seg_logits
        }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*70)
    print("FUME-FastSCNN Model Size Verification")
    print("="*70)

    # Test with shared encoder (default for <3M budget)
    print("\n1. FUME-FastSCNN (Shared Encoder)")
    model = FUMEFastSCNN(num_classes=3, num_seg_classes=3, shared_encoder=True)

    # Dummy input
    co2 = torch.randn(2, 1, 480, 640)
    ch4 = torch.randn(2, 1, 480, 640)
    modality_mask = torch.ones(2, 2)

    # Forward pass
    outputs = model(co2, ch4, modality_mask)

    print(f"   Classification output: {outputs['cls_logits'].shape}")
    print(f"   CO2 segmentation: {outputs['co2_seg_logits'].shape}")
    print(f"   CH4 segmentation: {outputs['ch4_seg_logits'].shape}")

    params = count_parameters(model)
    print(f"\n   Total parameters: {params:,} ({params/1e6:.2f}M)")

    if params < 3e6:
        print(f"   ✅ WITHIN BUDGET (<3M)!")
    else:
        print(f"   ⚠️ OVER BUDGET (>{params/1e6:.1f}M)")

    # Test with separate encoders
    print("\n2. FUME-FastSCNN (Separate Encoders)")
    model_sep = FUMEFastSCNN(num_classes=3, num_seg_classes=3, shared_encoder=False)
    params_sep = count_parameters(model_sep)
    print(f"   Total parameters: {params_sep:,} ({params_sep/1e6:.2f}M)")

    if params_sep < 3e6:
        print(f"   ✅ WITHIN BUDGET (<3M)!")
    else:
        print(f"   ⚠️ OVER BUDGET (>{params_sep/1e6:.1f}M)")

    # Test ablation model
    print("\n3. FUME-FastSCNN (No Attention - Ablation)")
    model_no_att = FUMEFastSCNNNoAttention(num_classes=3, num_seg_classes=3, shared_encoder=True)
    params_no_att = count_parameters(model_no_att)
    print(f"   Total parameters: {params_no_att:,} ({params_no_att/1e6:.2f}M)")

    print("\n" + "="*70)
    print("✅ All models successfully created!")
    print("="*70)

    # Breakdown
    print("\nParameter Breakdown:")
    print(f"  Shared encoder saves: {(params_sep - params)/1e6:.2f}M parameters")
    print(f"  Cross-attention adds: {(params - params_no_att)/1e6:.2f}M parameters")
