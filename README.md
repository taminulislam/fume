# FUME: Fused Unified Multi-gas Emission Network

**Cross-Modal Fusion for Gas Emission Analysis**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Research Question:** Can we predict rumen acidosis risk from OGI camera dual-gas emissions in controlled lab settings?

## 📋 Overview

FUME is a multi-task deep learning framework for automated rumen acidosis detection using thermal OGI (Optical Gas Imaging) camera data. The model performs:
- **Segmentation:** Pixel-level gas emission detection (CO2 and CH4)
- **Classification:** pH-based health state prediction (Healthy, Transitional, Acidotic)

### Key Features

✨ **Dual-Stream Architecture** - Separate encoders for CO2 and CH4 gas types
✨ **Cross-Modal Attention** - Learn interactions between gas emissions
✨ **Multi-Task Learning** - Joint segmentation and classification
✨ **Modality Dropout** - Robust to missing gas type data
✨ **Class Imbalance Handling** - Focal Loss + weighted sampling
✨ **Comprehensive Baselines** - 5 baseline models for rigorous evaluation

---

## 📊 Dataset

### Statistics
- **Total Samples:** 8,967 (21× augmentation from 428 originals)
- **Split:** 70% train (6,276) / 15% val (1,345) / 15% test (1,346)
- **Classes:** Healthy (45%), Transitional (2.6%), Acidotic (52.5%)
- **Gas Types:** CO2 (69%), CH4 (31%)
- **pH Levels:** 5.0, 5.3, 5.6, 5.9, 6.2, 6.5
- **Image Size:** 640×480 grayscale

### Class Distribution Challenge
⚠️ **Severe Imbalance:** Transitional class is only **2.6%** of dataset

**Solution:** Focal Loss with α=[1.0, 8.0, 1.2] + weighted random sampling

---

## 🏗️ Model Architecture

```
Input: Paired CO2 and CH4 Frames (640×480×1 each)
    ↓
┌─────────────────────────────────────────┐
│  Dual-Stream Encoder (ResNet-50)       │
│  ├── CO2 Stream → Features_CO2          │
│  └── CH4 Stream → Features_CH4          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Cross-Modal Attention Fusion           │
│  ├── Self-attention within each stream  │
│  └── Cross-attention between streams    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Dual-Task Heads                        │
│  ├── Segmentation → CO2 + CH4 masks     │
│  └── Classification → pH class          │
└─────────────────────────────────────────┘
```

### Model Variants
- **FUME (Main):** Dual-stream with cross-attention
- **FUME-Shared:** Shared encoder weights (ablation)
- **FUME-NoAttention:** Simple concatenation fusion (ablation)

---

## 🎯 Baseline Models

1. **Segmentation-Only:** Pure segmentation (establish ceiling)
2. **Classification-Only:** Pure classification (establish ceiling)
3. **Gas-Aware Classifier:** Classification with gas type embedding
4. **Early Fusion:** Concatenate CO2+CH4 before encoding
5. **Traditional ML:** Random Forest on hand-crafted features

---

## 📁 Project Structure

```
Acidosis/FUME/
├── data/
│   ├── pairing.py              # CO2-CH4 sample pairing
│   ├── dataset.py              # PyTorch dataset with modality dropout
│   └── transforms.py           # Albumentations augmentation
├── models/
│   ├── fume.py                 # Main FUME model
│   ├── baselines.py            # 5 baseline models
│   ├── backbones.py            # ResNet-50 encoder
│   ├── attention.py            # Cross-modal attention
│   └── heads.py                # Seg & classification heads
├── losses/
│   ├── focal_loss.py           # Focal + Dice losses
│   └── multi_task_loss.py      # Combined seg+cls loss
├── utils/
│   ├── metrics.py              # IoU, Dice, Balanced Acc, F1
│   ├── logger.py               # Weights & Biases integration
│   └── visualization.py        # Plotting utilities
├── configs/
│   ├── fume_config.yaml        # Main config
│   ├── baseline_configs/       # Baseline experiment configs
│   └── ablation_configs/       # Ablation study configs
├── notebooks/
│   ├── train_fume.ipynb        # Training notebook
│   └── test_fume.ipynb         # Evaluation notebook
├── train.py                     # Training script
├── test.py                      # Evaluation script
├── environment.yml              # Conda environment
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate fume_env

# OR use pip
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Create paired CO2-CH4 samples
cd data
python pairing.py

# This generates:
# - paired_train_annotations.csv
# - paired_val_annotations.csv
# - paired_test_annotations.csv
```

### 3. Training

**Option A: Using Jupyter Notebook (Recommended)**
```bash
jupyter notebook notebooks/train_fume.ipynb
```

**Option B: Using Python Script**
```bash
python train.py --config configs/fume_config.yaml
```

### 4. Evaluation

```bash
jupyter notebook notebooks/test_fume.ipynb
```

---

## 📈 Metrics

### Primary Metrics

**Classification:**
- ✅ Balanced Accuracy (handles class imbalance)
- ✅ Per-class F1-score (especially Transitional!)
- ✅ Confusion Matrix (3×3)
- ✅ Macro F1 (equal weight to all classes)
- ✅ Cohen's Kappa

**Segmentation:**
- ✅ Mean IoU (mIoU)
- ✅ Gas IoU (class 2 - most important!)
- ✅ Dice Score

**Multi-Task:**
- ✅ Joint: α×mIoU + β×Balanced_Acc

---

## 🧪 Ablation Studies

### 1. Input Modality Ablation
- CO2-only
- CH4-only
- CO2+CH4 early fusion
- **CO2+CH4 dual-stream (FUME)** ← Expected best

### 2. Multi-Task Learning Ablation
- Classification only
- Segmentation only
- **Both (multi-task)** ← Expected best

### 3. Fusion Strategy Ablation
- Concatenation
- Element-wise addition
- **Cross-attention (FUME)** ← Expected best

### 4. Backbone Ablation
- ResNet-18, ResNet-50, ResNet-101
- EfficientNet-B0

### 5. Class Imbalance Handling
- No weighting
- Class-weighted loss
- **Focal Loss (FUME)** ← Expected best
- Oversampling

### 6. pH Granularity
- 3-class (Healthy/Trans/Acidotic)
- 6-class (all pH levels)
- 2-class (Healthy vs Acidotic)
- Regression

---

## 🔧 Training Configuration

```yaml
model:
  name: FUME
  num_classes: 3
  num_seg_classes: 3
  pretrained: true

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  modality_dropout: 0.2

loss:
  seg_weight: 1.0
  cls_weight: 1.0
  focal_gamma: 2.0
  cls_alpha: [1.0, 8.0, 1.2]  # [Healthy, Trans, Acidotic]

augmentation:
  horizontal_flip: 0.5
  rotation: 15
  brightness_contrast: 0.2
  gaussian_noise: 0.3
```

---

## 📊 Expected Results

| Model | mIoU | Gas IoU | Balanced Acc | Macro F1 |
|-------|------|---------|--------------|----------|
| Seg-Only | 78% | 82% | - | - |
| Cls-Only | - | - | 65% | 0.58 |
| Gas-Aware | - | - | 68% | 0.62 |
| Early Fusion | 75% | 78% | 70% | 0.65 |
| **FUME (Ours)** | **82%** | **85%** | **75%** | **0.72** |

*(These are projected values - actual results depend on training)*

---

## 🎓 Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{fume2025,
  title={FUME: Cross-Modal Fusion for Gas Emission Analysis},
  author={Your Name},
  booktitle={CVPR},
  year={2025}
}
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- ResNet-50 backbone from torchvision
- Albumentations for data augmentation
- Weights & Biases for experiment tracking

---

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub Issues: [Submit an issue](../../issues)

---

## 🛠️ Development Status

✅ Data pairing module
✅ Dataset loader with modality dropout
✅ FUME model architecture
✅ 5 baseline models
✅ Loss functions (Focal + Multi-task)
⏳ Training framework (in progress)
⏳ Evaluation metrics
⏳ Visualization tools
⏳ Pretrained weights

---

**Built with ❤️ for advancing agriculture and animal health through AI**
