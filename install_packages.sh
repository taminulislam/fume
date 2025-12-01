#!/bin/bash

# FUME Package Installation Script
# Run this after creating the conda environment

echo "======================================================================"
echo "FUME Package Installation"
echo "======================================================================"

# Activate environment
echo "Activating fume_env..."
eval "$(conda shell.bash hook)"
conda activate fume_env

# Check activation
if [ "$CONDA_DEFAULT_ENV" != "fume_env" ]; then
    echo "❌ Failed to activate fume_env"
    echo "Please run: conda activate fume_env"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Install PyTorch with CUDA 12.1
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
echo ""
echo "Installing other packages..."
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install scikit-image==0.21.0
pip install opencv-python==4.8.0.76
pip install Pillow==10.0.0
pip install tqdm==4.65.0
pip install PyYAML==6.0
pip install wandb==0.15.12
pip install albumentations==1.3.1
pip install timm==0.9.7

echo ""
echo "======================================================================"
echo "Verifying Installation"
echo "======================================================================"

# Verify PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CUDA not available')"

# Verify other packages
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import albumentations; print(f'Albumentations version: {albumentations.__version__}')"
python -c "import wandb; print(f'W&B version: {wandb.__version__}')"

echo ""
echo "======================================================================"
echo "✅ Installation Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. cd data && python pairing.py"
echo "  2. python train.py --config configs/fume_fastscnn_config.yaml"
echo ""
