import torch
import sys
sys.path.insert(0, '.')

from models.fume import FUMEModel, FUMEModelSharedEncoder
from models.backbones import ResNet50Encoder

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_parameters_by_module(model):
    """Count parameters for each major module"""
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    return module_params

print("="*70)
print("FUME Model Size Analysis")
print("="*70)

# Check FUME (dual encoder, separate weights)
print("\n1. FUME Model (Dual-Stream, Separate Encoders)")
model = FUMEModel(num_classes=3, num_seg_classes=3, pretrained=False)
total, trainable = count_parameters(model)
print(f"   Total parameters: {total:,} ({total/1e6:.2f}M)")
print(f"   Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")

module_params = count_parameters_by_module(model)
print("\n   Breakdown by module:")
for name, params in module_params.items():
    print(f"     - {name}: {params:,} ({params/1e6:.2f}M)")

# Check FUME with shared encoder
print("\n2. FUME Model (Shared Encoder)")
model_shared = FUMEModelSharedEncoder(num_classes=3, num_seg_classes=3, pretrained=False)
total, trainable = count_parameters(model_shared)
print(f"   Total parameters: {total:,} ({total/1e6:.2f}M)")
print(f"   Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")

# Check single ResNet-50 encoder
print("\n3. Single ResNet-50 Encoder")
encoder = ResNet50Encoder(pretrained=False, in_channels=1)
total, trainable = count_parameters(encoder)
print(f"   Total parameters: {total:,} ({total/1e6:.2f}M)")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print(f"❌ FUME is TOO LARGE: ~{count_parameters(FUMEModel())[0]/1e6:.1f}M parameters")
print(f"   Target: <3M parameters")
print(f"   Current: ~94M parameters (31x over budget!)")
print("\n🔥 CRITICAL: Need to switch to lightweight backbone!")
print("="*70)

