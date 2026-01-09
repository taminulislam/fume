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

---

## 📊 Dataset

### Statistics
- **Total Samples:** 8,967 (21× augmentation from 428 originals)
- **Split:** 70% train (6,276) / 15% val (1,345) / 15% test (1,346)
- **Classes:** Healthy (45%), Transitional (2.6%), Acidotic (52.5%)
- **Gas Types:** CO2 (69%), CH4 (31%)
- **pH Levels:** 5.0, 5.3, 5.6, 5.9, 6.2, 6.5
- **Image Size:** 640×480 grayscale


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
- **FUME (Main):** Dual-stream ResNet-50 with cross-attention
- **FUME-FastSCNN:** Lightweight Fast-SCNN backbone (~2.8M parameters)

---

## 📁 Project Structure

```
fume/
├── data/
│   ├── pairing.py              # CO2-CH4 sample pairing
│   ├── dataset.py              # PyTorch dataset with modality dropout
│   └── transforms.py           # Albumentations augmentation
├── models/
│   ├── fume.py                 # Main FUME model (ResNet-50)
│   ├── fume_fastscnn.py        # FUME with Fast-SCNN backbone
│   ├── fastscnn.py             # Fast-SCNN encoder
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
│   └── fume_config.yaml        # Model configuration
├── notebooks/
│   ├── train_fume.ipynb        # Training notebook
│   └── test_fume.ipynb         # Evaluation notebook
├── train.py                     # Training script
├── test_models.py               # Evaluation script
├── check_model_size.py          # Model parameter checker
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

### 1. Fusion Strategy
- Concatenation
- Element-wise addition
- **Cross-attention (FUME)** ← Expected best

### 2. Backbone Architecture
- **ResNet-50 (FUME)** - High accuracy
- **Fast-SCNN (FUME-FastSCNN)** - Lightweight (~2.8M params)

### 3. Multi-Task Learning
- Classification only
- Segmentation only
- **Both (multi-task)** ← Expected best

### 4. Encoder Sharing
- Separate CO2/CH4 encoders
- **Shared encoder weights** ← Parameter efficient

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
  num_epochs: 50
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
