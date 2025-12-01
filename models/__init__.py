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
    'TraditionalMLBaseline'
]
