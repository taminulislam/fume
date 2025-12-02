"""
Test script to verify all comparison models can be instantiated
and count their parameters
"""

import torch
from models import (
    FUMEFastSCNN,
    BiSeNetV2,
    CMX,
    DDRNetSlim,
    RTFNet,
    ESPNetV2,
    MTINet,
    ENet,
    DANet
)


def test_model(ModelClass, name):
    """Test a single model"""
    print(f"\nTesting {name}...")

    try:
        # Instantiate model
        model = ModelClass(num_classes=3, num_seg_classes=3)

        # Count parameters
        num_params = model.get_num_parameters()

        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 2
        co2_frame = torch.randn(batch_size, 1, 480, 640).to(device)
        ch4_frame = torch.randn(batch_size, 1, 480, 640).to(device)
        modality_mask = torch.ones(batch_size, 2).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(co2_frame, ch4_frame, modality_mask)

        # Check outputs
        assert 'cls_logits' in outputs, "Missing cls_logits"
        assert 'co2_seg_logits' in outputs, "Missing co2_seg_logits"
        assert 'ch4_seg_logits' in outputs, "Missing ch4_seg_logits"

        assert outputs['cls_logits'].shape == (batch_size, 3), f"Wrong cls shape: {outputs['cls_logits'].shape}"
        assert outputs['co2_seg_logits'].shape == (batch_size, 3, 480, 640), f"Wrong seg shape: {outputs['co2_seg_logits'].shape}"

        print(f"✅ {name}: {num_params:,} params ({num_params/1e6:.2f}M)")
        print(f"   Output shapes: cls={outputs['cls_logits'].shape}, seg={outputs['co2_seg_logits'].shape}")

        return True, num_params

    except Exception as e:
        print(f"❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    print("="*70)
    print("TESTING ALL COMPARISON MODELS")
    print("="*70)

    models = {
        'FUMEFastSCNN': FUMEFastSCNN,
        'BiSeNetV2': BiSeNetV2,
        'CMX': CMX,
        'DDRNetSlim': DDRNetSlim,
        'RTFNet': RTFNet,
        'ESPNetV2': ESPNetV2,
        'MTINet': MTINet,
        'ENet': ENet,
        'DANet': DANet
    }

    results = []

    for name, ModelClass in models.items():
        success, num_params = test_model(ModelClass, name)
        results.append({
            'name': name,
            'success': success,
            'params': num_params
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Sort by parameters
    results_sorted = sorted(results, key=lambda x: x['params'])

    for result in results_sorted:
        status = "✅" if result['success'] else "❌"
        name = result['name']
        params = result['params']
        print(f"{status} {name:<20} {params:>12,} params ({params/1e6:>5.2f}M)")

    # Count successes
    num_success = sum(1 for r in results if r['success'])
    print(f"\n{num_success}/{len(results)} models passed tests")

    print("="*70)


if __name__ == "__main__":
    main()
