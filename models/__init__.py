# Original FUME with ResNet-50 (64M parameters - for reference)
from .fume import FUMEModel

# FUME with Fast-SCNN (<3M parameters - RECOMMENDED)
from .fume_fastscnn import (
    FUMEFastSCNN,
    FUMEFastSCNNNoAttention
)

# Baseline models with Fast-SCNN
from .baselines_fastscnn import (
    SegmentationOnlyFastSCNN,
    ClassificationOnlyFastSCNN,
    GasAwareClassifierFastSCNN,
    EarlyFusionFastSCNN,
    MultiTaskFastSCNN
)

# Traditional ML baseline
from .baselines import TraditionalMLBaseline

# Comparison Models for Benchmarking
from .bisenetv2 import BiSeNetV2      # 3.4M - Bilateral segmentation SOTA
from .cmx import CMX                   # 3.8M - Cross-modal fusion transformer
from .ddrnet import DDRNetSlim         # 5.7M - Dual-resolution network
from .rtfnet import RTFNet             # 4.2M - RGB-Thermal fusion
from .espnetv2 import ESPNetV2         # 1.2M - Efficient spatial pyramid
from .mtinet import MTINet             # 3.5M - Multi-task interaction
from .enet import ENet                 # 0.4M - Ultra-lightweight
from .danet import DANet               # 3.2M - Dual attention network

__all__ = [
    # Main models
    'FUMEFastSCNN',  # <-- USE THIS (2.8M params)
    'FUMEFastSCNNNoAttention',  # Ablation
    'FUMEModel',  # Original ResNet-50 version (64M)

    # Baselines
    'SegmentationOnlyFastSCNN',
    'ClassificationOnlyFastSCNN',
    'GasAwareClassifierFastSCNN',
    'EarlyFusionFastSCNN',
    'MultiTaskFastSCNN',
    'TraditionalMLBaseline',

    # Comparison Models
    'BiSeNetV2',     # Lightweight competitor
    'CMX',           # Multi-modal fusion SOTA
    'DDRNetSlim',    # Recent lightweight SOTA
    'RTFNet',        # Multi-modal fusion for thermal
    'ESPNetV2',      # Lightweight pyramid baseline
    'MTINet',        # Multi-task learning competitor
    'ENet',          # Ultra-lightweight baseline
    'DANet'          # Attention baseline
]
