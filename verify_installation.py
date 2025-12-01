#!/usr/bin/env python3
"""
FUME Installation Verification Script
Checks if all required packages are installed and working
"""

import sys

def check_package(package_name, import_name=None, version_attr='__version__'):
    """Check if a package is installed and print its version"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, 'unknown')
        print(f"✅ {package_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} NOT INSTALLED")
        return False

def main():
    print("="*70)
    print("FUME Installation Verification")
    print("="*70)
    print()

    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('scikit-image', 'skimage'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('tqdm', 'tqdm'),
        ('PyYAML', 'yaml'),
        ('wandb', 'wandb'),
        ('albumentations', 'albumentations'),
        ('timm', 'timm'),
    ]

    print("Package Versions:")
    print("-"*70)

    success_count = 0
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            success_count += 1

    print()
    print("-"*70)
    print(f"Installed: {success_count}/{len(packages)} packages")
    print()

    # Check CUDA
    print("="*70)
    print("CUDA Configuration:")
    print("="*70)
    try:
        import torch
        print(f"PyTorch version:     {torch.__version__}")
        print(f"CUDA available:      {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version:        {torch.version.cuda}")
            print(f"CUDA device count:   {torch.cuda.device_count()}")
            print(f"CUDA device name:    {torch.cuda.get_device_name(0)}")
            print(f"CUDA device memory:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA not available - training will be slow on CPU")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")

    print()
    print("="*70)

    if success_count == len(packages):
        print("✅ All packages installed successfully!")
        print("✅ Ready for training!")
        return 0
    else:
        print(f"⚠️  {len(packages) - success_count} packages missing")
        print("Please install missing packages before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())
