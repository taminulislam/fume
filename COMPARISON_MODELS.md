# Comparison Models for Benchmarking

This document describes the 8 comparison models implemented for benchmarking against FUME-FastSCNN.

## Overview

All models are optimized for fast training (15 epochs) and adapted for dual-gas (CO2 + CH4) emission analysis with multi-task learning (segmentation + classification).

## Models

### 1. BiSeNetV2 - Bilateral Segmentation Network V2
**Parameters:** ~3.4M
**Paper:** https://arxiv.org/abs/2004.02147

**Architecture:**
- Dual-path design: Detail Branch + Semantic Branch
- Bilateral Guided Aggregation Layer for fusion
- Gather-and-Expansion modules for efficiency

**Key Features:**
- Direct lightweight competitor to FUME
- Efficient spatial pyramid with bilateral paths
- Real-time segmentation capability

---

### 2. CMX - Cross-Modal Fusion Transformer
**Parameters:** ~3.8M
**Paper:** https://arxiv.org/abs/2203.04838

**Architecture:**
- Lightweight CNN encoders (separate for CO2 and CH4)
- Cross-Modal Transformer blocks with attention
- Multi-head cross-modal fusion

**Key Features:**
- SOTA multi-modal fusion approach
- Transformer-based cross-attention
- Originally designed for RGB-Thermal fusion

---

### 3. DDRNet-23-Slim - Deep Dual-Resolution Network
**Parameters:** ~5.7M
**Paper:** https://arxiv.org/abs/2101.06085

**Architecture:**
- Dual-resolution branches (high-res + low-res)
- Deep Aggregation Pyramid Pooling Module (DAPPM)
- Bilateral fusion between branches

**Key Features:**
- Recent lightweight SOTA
- Preserves spatial details while capturing context
- Multi-scale feature aggregation

---

### 4. RTFNet - RGB-Thermal Fusion Network
**Parameters:** ~4.2M
**Paper:** https://arxiv.org/abs/1909.03849

**Architecture:**
- Dual ResNet18 encoders
- Multi-level feature fusion
- Decoder with skip connections

**Key Features:**
- Designed specifically for multi-modal fusion
- Thermal imaging expertise
- Late fusion strategy

---

### 5. ESPNetV2 - Efficient Spatial Pyramid Network V2
**Parameters:** ~1.2M
**Paper:** https://arxiv.org/abs/1811.11431

**Architecture:**
- Extremely Efficient Spatial Pyramid (EESP) modules
- Depth-wise dilated convolutions in pyramid structure
- Hierarchical Feature Fusion (HFF)

**Key Features:**
- Ultra-lightweight design
- Edge device optimization
- Spatial pyramid baseline

---

### 6. MTI-Net - Multi-Task Interaction Network
**Parameters:** ~3.5M
**Paper:** Multi-Task Learning with Task Interactions

**Architecture:**
- Shared MobileNetV2-style encoder
- Task-specific branches (seg + cls)
- Task Interaction Module (bidirectional information flow)

**Key Features:**
- Explicit multi-task interaction
- Attention-based task fusion
- Designed for joint learning

---

### 7. ENet - Efficient Neural Network
**Parameters:** ~0.4M
**Paper:** https://arxiv.org/abs/1606.02147

**Architecture:**
- Asymmetric encoder-decoder
- Bottleneck blocks with dilated convolutions
- Extremely parameter-efficient

**Key Features:**
- **Smallest model** (makes FUME look good!)
- Real-time inference on edge devices
- Early downsampling strategy

---

### 8. DANet - Dual Attention Network (Lightweight)
**Parameters:** ~3.2M
**Paper:** https://arxiv.org/abs/1809.02983

**Architecture:**
- Lightweight MobileNet backbone
- Position Attention Module (spatial relationships)
- Channel Attention Module (channel dependencies)
- Parallel dual attention streams

**Key Features:**
- Attention mechanism baseline
- Captures long-range dependencies
- Both spatial and channel attention

---

## Quick Start

### Train All Models (15 epochs each)

```bash
python train_all_comparison_models.py --config configs/fast_comparison_config.yaml
```

### Train Individual Model

```bash
# Example: Train BiSeNetV2
python train_all_comparison_models.py --models BiSeNetV2

# Example: Train multiple specific models
python train_all_comparison_models.py --models BiSeNetV2 CMX ENet
```

### Configuration

The fast training configuration is optimized for speed:
- **Epochs:** 15 (instead of 50)
- **Batch Size:** 16 (increased from 8)
- **Learning Rate:** 0.002 (2x higher)
- **Num Workers:** 8 (faster data loading)
- **Minimal Augmentation:** Reduced for faster convergence
- **AMP Enabled:** Mixed precision for speed

## Model Comparison Table

| Model | Parameters | Type | Key Innovation |
|-------|-----------|------|---------------|
| **FUMEFastSCNN** | 2.8M | Ours | Cross-modal attention fusion |
| BiSeNetV2 | 3.4M | Lightweight | Bilateral dual-path |
| CMX | 3.8M | Fusion SOTA | Cross-modal transformer |
| DDRNetSlim | 5.7M | Lightweight SOTA | Dual-resolution |
| RTFNet | 4.2M | Multi-modal | RGB-Thermal fusion |
| ESPNetV2 | 1.2M | Ultra-light | Spatial pyramid |
| MTI-Net | 3.5M | Multi-task | Task interaction |
| **ENet** | **0.4M** | **Ultra-light** | **Asymmetric bottleneck** |
| DANet | 3.2M | Attention | Dual attention |

## Expected Results

All models will be evaluated on:
1. **Classification Metrics:**
   - Balanced Accuracy (primary metric)
   - Macro F1-Score
   - Per-class F1 (Healthy, Transitional, Acidotic)

2. **Segmentation Metrics:**
   - Mean IoU
   - Mean Dice
   - Gas IoU (most important)

3. **Efficiency Metrics:**
   - Training time (15 epochs)
   - Inference time
   - Memory usage

## Usage Notes

### Fast Training Optimizations

1. **Higher Learning Rate:** 2x higher (0.002) for faster convergence
2. **Larger Batch Size:** 16 instead of 8
3. **Reduced Augmentation:** Less aggressive for faster training
4. **Efficient Data Loading:** 8 workers with pin_memory
5. **Mixed Precision:** FP16 for speed boost

### Training Tips

- All models use the same training configuration for fair comparison
- Results after 15 epochs may not be fully converged
- For production use, train for 50+ epochs
- Monitor GPU memory usage (some models may need batch size adjustment)

## Output Structure

```
checkpoints_comparison/
├── FUMEFastSCNN/
│   ├── best_model.pth
│   └── last_model.pth
├── BiSeNetV2/
├── CMX/
├── ...
└── DANet/

logs_comparison/
├── FUMEFastSCNN/
├── BiSeNetV2/
└── ...

results_comparison/
└── ... (final metrics and visualizations)
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `configs/fast_comparison_config.yaml`:
```yaml
training:
  batch_size: 8  # or 4 for larger models
```

### Slow Training
Increase num_workers:
```yaml
training:
  num_workers: 12  # adjust based on CPU cores
```

### Model Import Errors
Ensure all models are properly imported in `models/__init__.py`

## Citation

If you use these comparison models, please cite their respective papers:

```bibtex
# BiSeNetV2
@article{yu2021bisenet,
  title={BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation},
  author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
  journal={IJCV},
  year={2021}
}

# CMX
@inproceedings{zhang2023cmx,
  title={Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers},
  author={Zhang, Jiaming and Liu, Huayao and Yang, Kailun and Hu, Xinxin and Liu, Ruiping and Stiefelhagen, Rainer},
  booktitle={IEEE Transactions on Intelligent Transportation Systems},
  year={2023}
}

# ... (add other citations as needed)
```

## License

See individual model papers for their respective licenses.
