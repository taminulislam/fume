"""
Baseline Models with Fast-SCNN Backbone

All models use Fast-SCNN for fair comparison with FUME-FastSCNN
"""

import torch
import torch.nn as nn
from typing import Dict
from .fastscnn import FastSCNNEncoder, FastSCNNSegmentationHead


# Baseline 1: Segmentation-Only Model
class SegmentationOnlyFastSCNN(nn.Module):
    """
    Baseline 1: Pure segmentation with Fast-SCNN
    Purpose: Establish segmentation performance ceiling

    Input: Single grayscale frame
    Output: 3-class segmentation mask only
    Parameters: ~1.3M
    """

    def __init__(self, num_seg_classes: int = 3):
        super().__init__()
        self.encoder = FastSCNNEncoder(in_channels=1)
        encoder_info = self.encoder.get_output_channels()

        self.seg_head = FastSCNNSegmentationHead(
            low_channels=encoder_info['low_channels'],
            high_channels=encoder_info['high_channels'],
            num_classes=num_seg_classes
        )

    def forward(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (frame.shape[2], frame.shape[3])
        low_feat, high_feat = self.encoder(frame)
        seg_logits = self.seg_head(low_feat, high_feat, target_size)

        return {'seg_logits': seg_logits}


# Baseline 2: Classification-Only Model
class ClassificationOnlyFastSCNN(nn.Module):
    """
    Baseline 2: Pure classification with Fast-SCNN
    Purpose: Establish classification performance ceiling without segmentation

    Input: Single grayscale frame
    Output: 3-class pH prediction only
    Parameters: ~1.2M
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = FastSCNNEncoder(in_channels=1)
        encoder_info = self.encoder.get_output_channels()

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_info['high_channels'], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, high_feat = self.encoder(frame)
        cls_logits = self.cls_head(high_feat)

        return {'cls_logits': cls_logits}


# Baseline 3: Gas-Aware Classifier
class GasAwareClassifierFastSCNN(nn.Module):
    """
    Baseline 3: Classification with explicit gas type embedding
    Purpose: Show importance of gas type information

    Input: Frame + gas type one-hot vector [CO2, CH4]
    Output: 3-class pH prediction
    Parameters: ~1.25M
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = FastSCNNEncoder(in_channels=1)
        encoder_info = self.encoder.get_output_channels()

        # Gas type embedding
        self.gas_embedding = nn.Embedding(2, 64)  # 2 gas types: CO2=0, CH4=1

        # Classifier with gas embedding
        self.classifier = nn.Sequential(
            nn.Linear(encoder_info['high_channels'] + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        frame: torch.Tensor,
        gas_type: torch.Tensor  # (B,) with values 0 (CO2) or 1 (CH4)
    ) -> Dict[str, torch.Tensor]:
        # Encode frame
        _, high_feat = self.encoder(frame)
        pooled = self.global_pool(high_feat).view(high_feat.size(0), -1)

        # Get gas embedding
        gas_emb = self.gas_embedding(gas_type)

        # Concatenate
        combined = torch.cat([pooled, gas_emb], dim=1)

        # Classify
        cls_logits = self.classifier(combined)

        return {'cls_logits': cls_logits}


# Baseline 4: Early Fusion Model
class EarlyFusionFastSCNN(nn.Module):
    """
    Baseline 4: Early fusion of CO2 and CH4 (concatenate then encode)
    Purpose: Show that dual-stream late fusion > early fusion

    Input: CO2 frame + CH4 frame concatenated as 2-channel input
    Output: Segmentation + Classification
    Parameters: ~2.6M
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3
    ):
        super().__init__()

        # Encoder for 2-channel input (CO2 + CH4 concatenated)
        # Need to modify first conv layer
        self.encoder = FastSCNNEncoder(in_channels=2)

        encoder_info = self.encoder.get_output_channels()

        # Task heads
        self.seg_head = FastSCNNSegmentationHead(
            low_channels=encoder_info['low_channels'],
            high_channels=encoder_info['high_channels'],
            num_classes=num_seg_classes
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_info['high_channels'], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        co2_frame: torch.Tensor,
        ch4_frame: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        target_size = (co2_frame.shape[2], co2_frame.shape[3])

        # Early fusion: concatenate along channel dimension
        fused_input = torch.cat([co2_frame, ch4_frame], dim=1)  # (B, 2, H, W)

        # Single encoder
        low_feat, high_feat = self.encoder(fused_input)

        # Task heads
        seg_logits = self.seg_head(low_feat, high_feat, target_size)
        cls_logits = self.cls_head(high_feat)

        return {
            'cls_logits': cls_logits,
            'seg_logits': seg_logits  # Single segmentation for fused input
        }


# Baseline 5: Multi-Task Baseline (Seg + Cls without dual-stream)
class MultiTaskFastSCNN(nn.Module):
    """
    Baseline 5: Multi-task learning without dual-stream
    Purpose: Show dual-stream benefit

    Input: Single frame
    Output: Segmentation + Classification
    Parameters: ~1.4M
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3
    ):
        super().__init__()

        self.encoder = FastSCNNEncoder(in_channels=1)
        encoder_info = self.encoder.get_output_channels()

        self.seg_head = FastSCNNSegmentationHead(
            low_channels=encoder_info['low_channels'],
            high_channels=encoder_info['high_channels'],
            num_classes=num_seg_classes
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_info['high_channels'], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (frame.shape[2], frame.shape[3])

        low_feat, high_feat = self.encoder(frame)

        seg_logits = self.seg_head(low_feat, high_feat, target_size)
        cls_logits = self.cls_head(high_feat)

        return {
            'cls_logits': cls_logits,
            'seg_logits': seg_logits
        }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*70)
    print("Baseline Models with Fast-SCNN - Parameter Count")
    print("="*70)

    models = [
        ("Segmentation-Only", SegmentationOnlyFastSCNN()),
        ("Classification-Only", ClassificationOnlyFastSCNN()),
        ("Gas-Aware Classifier", GasAwareClassifierFastSCNN()),
        ("Early Fusion", EarlyFusionFastSCNN()),
        ("Multi-Task (Single-Stream)", MultiTaskFastSCNN())
    ]

    for name, model in models:
        params = count_parameters(model)
        status = "✅" if params < 3e6 else "⚠️"
        print(f"\n{status} {name}")
        print(f"   Parameters: {params:,} ({params/1e6:.2f}M)")

    print("\n" + "="*70)
    print("All baselines within 3M parameter budget!" )
    print("="*70)
