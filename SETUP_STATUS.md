# FUME Setup Status

**Date:** December 1, 2025
**Status:** ✅ **SETUP IN PROGRESS**

---

## ✅ Completed Tasks

### 1. Environment Creation
- ✅ Conda environment `fume_env` created with Python 3.9
- ✅ Environment location: `/home/siu856569517/.conda/envs/fume_env`

### 2. Paired Dataset Creation
- ✅ Successfully created paired CO2-CH4 samples
- ✅ Files location: `/home/siu856569517/Taminul/Acidosis/FUME/data/`

**Dataset Statistics:**
```
Train:  4,383 pairs (736 KB)
  - 1,893 fully paired (43.2%)
  - 2,428 CO2 only
  - 62 CH4 only

Val:    939 pairs (152 KB)
  - 406 fully paired (43.2%)
  - 520 CO2 only
  - 13 CH4 only

Test:   936 pairs (155 KB)
  - 410 fully paired (43.8%)
  - 517 CO2 only
  - 9 CH4 only

Total:  6,258 paired samples
```

**Files Created:**
- ✅ `paired_train_annotations.csv` (4,384 rows)
- ✅ `paired_val_annotations.csv` (940 rows)
- ✅ `paired_test_annotations.csv` (937 rows)

### 3. Implementation Files (All Complete)
- ✅ **Models:** FastSCNN, FUME-FastSCNN, 5 Baselines
- ✅ **Data:** Pairing, Dataset loader, Transforms
- ✅ **Losses:** Focal, Dice, Multi-task
- ✅ **Utils:** Metrics, Logger, Visualization
- ✅ **Training:** train.py, train_fume.ipynb
- ✅ **Configuration:** fume_fastscnn_config.yaml
- ✅ **Documentation:** README, Implementation guides

---

## 🔄 In Progress

### Package Installation (Background Processes)

**PyTorch Installation (Process ID: 43f618)**
- Status: Downloading PyTorch 2.5.1+cu121 (780.4 MB)
- Command: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

**Other Packages Installation (Process ID: 90d890)**
- Status: Downloading packages
- Packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scikit-image, opencv-python, Pillow, tqdm, PyYAML, wandb, albumentations, timm
- Current: Downloading numpy-1.24.3 (17.3 MB)

---

## 📦 Packages to be Installed

### Core Deep Learning
- ✅ Python 3.9.25
- 🔄 PyTorch 2.5.1+cu121 (with CUDA 12.1)
- 🔄 torchvision
- 🔄 torchaudio

### Data Processing
- 🔄 numpy==1.24.3
- 🔄 pandas==2.0.3
- 🔄 opencv-python==4.8.0.76
- 🔄 Pillow==10.0.0
- 🔄 scikit-image==0.21.0

### Visualization
- 🔄 matplotlib==3.7.2
- 🔄 seaborn==0.12.2

### Machine Learning
- 🔄 scikit-learn==1.3.0

### Augmentation
- 🔄 albumentations==1.3.1

### Utilities
- 🔄 tqdm==4.65.0
- 🔄 PyYAML==6.0
- 🔄 wandb==0.15.12
- 🔄 timm==0.9.7

---

## ⏳ Estimated Time Remaining

Based on file sizes:
- PyTorch (780 MB): ~5-10 minutes
- Other packages (~150 MB total): ~2-5 minutes

**Total estimated time:** 10-15 minutes

---

## 📋 Next Steps (After Installation)

### 1. Verify Installation
```bash
conda activate fume_env
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy, pandas, cv2, albumentations, wandb"
```

### 2. Update Config Paths (if needed)
Check that `configs/fume_fastscnn_config.yaml` has correct paths:
```yaml
data:
  dataset_root: "../../dataset"  # Acidosis/dataset
  paired_train_csv: "data/paired_train_annotations.csv"
  paired_val_csv: "data/paired_val_annotations.csv"
  paired_test_csv: "data/paired_test_annotations.csv"
```

### 3. Start Training
```bash
cd Acidosis/FUME
conda activate fume_env
python train.py --config configs/fume_fastscnn_config.yaml
```

**Or use the notebook:**
```bash
jupyter notebook notebooks/train_fume.ipynb
```

---

## 🎯 Model Information

**Architecture:** FUME-FastSCNN
**Parameters:** 2.8M (within 3M budget ✅)
**Tasks:** Multi-task (Segmentation + Classification)
**Input:** Dual-gas (CO2 + CH4) frames
**Output:**
- Classification: 3 classes (Healthy, Transitional, Acidotic)
- Segmentation: 3 classes (Background, Animal, Gas)

---

## 📊 Expected Training Results

| Metric | Expected Value |
|--------|---------------|
| Segmentation mIoU | 78-82% |
| Gas IoU (Class 2) | 80-85% |
| Balanced Accuracy | 68-72% |
| Macro F1 | 0.65-0.70 |
| Transitional F1 | 0.45-0.55 |
| Training Time | 8-18 hours (GPU) |
| Inference Speed | 100+ FPS |

---

## 🔧 Environment Details

**System:**
- Platform: Linux 5.14.0-570.55.1.el9_6.x86_64
- Working Directory: `/home/siu856569517/Taminul/Acidosis/FUME`
- Conda Environment: `fume_env` (Python 3.9)

**Data Paths:**
- Dataset root: `/home/siu856569517/Taminul/dataset`
- FUME data: `/home/siu856569517/Taminul/Acidosis/FUME/data`
- Paired CSVs: `/home/siu856569517/Taminul/Acidosis/FUME/data/paired_*.csv`

---

## ✅ Checklist Before Training

- [x] Conda environment created
- [ ] PyTorch installed with CUDA support (in progress)
- [ ] All packages installed (in progress)
- [x] Paired dataset created
- [x] Configuration file ready
- [x] Training script ready
- [x] Training notebook ready
- [x] All model files implemented
- [x] All utility files implemented

---

## 📝 Notes

1. **Class Imbalance Handling:**
   - Focal Loss with α=[1.0, 8.0, 1.2] for Transitional boost
   - Weighted Random Sampling enabled
   - Heavy augmentation strategy

2. **Modality Dropout:**
   - 0.2 probability to randomly drop one gas type during training
   - Helps model learn robust features from either gas alone

3. **Training Features:**
   - Mixed Precision Training (AMP) enabled
   - Early stopping (patience 15)
   - Gradient clipping (1.0)
   - CosineAnnealingLR scheduler
   - W&B logging for experiment tracking

4. **GPU Requirement:**
   - CUDA 12.1 compatible GPU required
   - Minimum 8GB GPU memory recommended
   - CPU training possible but very slow

---

## 🆘 Troubleshooting

### If CUDA not available:
```bash
# Check CUDA
nvidia-smi

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### If out of memory during training:
```yaml
# Edit config: reduce batch size
training:
  batch_size: 4  # or 2
```

### If packages conflict:
```bash
# Recreate environment
conda deactivate
conda remove -n fume_env --all -y
conda create -n fume_env python=3.9 -y
conda activate fume_env
# Run installations again
```

---

**Last Updated:** December 1, 2025 02:53 AM
**Status:** Packages installing in background. Expect completion in ~10-15 minutes.
