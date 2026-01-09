"""
FUME Evaluation Script
Evaluates trained FUME-FastSCNN model on test set
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

from models import FUMEFastSCNN
from data import FUMEDataset, get_val_transforms
from utils.metrics import SegmentationMetrics, ClassificationMetrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, input_size=(1, 1, 480, 640), device='cuda'):
    try:
        from thop import profile
        dummy_co2 = torch.randn(input_size).to(device)
        dummy_ch4 = torch.randn(input_size).to(device)
        dummy_mask = torch.ones(1, 2).to(device)
        flops, params = profile(model, inputs=(dummy_co2, dummy_ch4, dummy_mask), verbose=False)
        return flops
    except ImportError:
        params = count_parameters(model)
        flops_estimate = params * 2 * input_size[2] * input_size[3] / 32
        return flops_estimate


def measure_fps(model, device='cuda', num_iterations=100):
    model.eval()
    dummy_co2 = torch.randn(1, 1, 480, 640).to(device)
    dummy_ch4 = torch.randn(1, 1, 480, 640).to(device)
    dummy_mask = torch.ones(1, 2).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_co2, dummy_ch4, dummy_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_co2, dummy_ch4, dummy_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    fps = num_iterations / elapsed_time
    return fps


def create_colored_mask(mask, colors):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        colored[mask == cls_id] = color
    return colored


def create_overlay(image, mask, colors, alpha=0.5):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colored_mask = create_colored_mask(mask, colors)
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay


def save_predictions(test_dataset, model, device, class_names, save_dir='results/test_predictions', num_samples=100):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gt_colors = {0: (0, 0, 0), 1: (255, 255, 200), 2: (160, 160, 160)}
    co2_pred_colors = {0: (0, 0, 0), 1: (255, 255, 200), 2: (135, 206, 250)}
    ch4_pred_colors = {0: (0, 0, 0), 1: (255, 255, 200), 2: (255, 182, 255)}

    ph_ranges = {
        'Healthy': 'pH 6.2-7.0',
        'Transitional': 'pH 5.5-6.2',
        'Acidotic': 'pH < 5.5'
    }

    sample_indices = np.linspace(0, len(test_dataset)-1, num_samples, dtype=int)
    model.eval()

    for idx in tqdm(sample_indices, desc="Saving predictions"):
        batch = test_dataset[idx]

        co2_frame = batch['co2_frame'].unsqueeze(0).to(device)
        ch4_frame = batch['ch4_frame'].unsqueeze(0).to(device)
        modality_mask = batch['modality_mask'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(co2_frame, ch4_frame, modality_mask)

        co2_img = (batch['co2_frame'][0].numpy() * 255).astype(np.uint8)
        ch4_img = (batch['ch4_frame'][0].numpy() * 255).astype(np.uint8)
        co2_gt_mask = batch['co2_mask'].numpy()
        ch4_gt_mask = batch['ch4_mask'].numpy()
        co2_pred_mask = outputs['co2_seg_logits'].argmax(dim=1)[0].cpu().numpy()
        ch4_pred_mask = outputs['ch4_seg_logits'].argmax(dim=1)[0].cpu().numpy()

        gt_class = batch['class_label'].item()
        pred_class = outputs['cls_logits'].argmax(dim=1).item()
        ph_label = ph_ranges[class_names[pred_class]]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        axes[0, 0].imshow(co2_img, cmap='gray')
        axes[0, 0].set_title('CO2 Input')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(create_colored_mask(co2_gt_mask, gt_colors))
        axes[0, 1].set_title('CO2 GT')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(create_colored_mask(co2_pred_mask, co2_pred_colors))
        axes[0, 2].set_title('CO2 Pred')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(create_overlay(co2_img, co2_pred_mask, co2_pred_colors))
        axes[0, 3].set_title('CO2 Overlay')
        axes[0, 3].axis('off')

        axes[1, 0].imshow(ch4_img, cmap='gray')
        axes[1, 0].set_title('CH4 Input')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(create_colored_mask(ch4_gt_mask, gt_colors))
        axes[1, 1].set_title('CH4 GT')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(create_colored_mask(ch4_pred_mask, ch4_pred_colors))
        axes[1, 2].set_title('CH4 Pred')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(create_overlay(ch4_img, ch4_pred_mask, ch4_pred_colors))
        axes[1, 3].set_title('CH4 Overlay')
        axes[1, 3].axis('off')

        status = 'Correct' if pred_class == gt_class else 'Wrong'
        plt.suptitle(f'Sample {idx} | GT: {class_names[gt_class]} | Pred: {class_names[pred_class]} ({status})')
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nSaved {len(sample_indices)} predictions to {save_dir}")


def main():
    # Load configuration
    config_path = 'configs/fume_fastscnn_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = FUMEFastSCNN(
        num_classes=config['model']['num_classes'],
        num_seg_classes=config['model']['num_seg_classes'],
        shared_encoder=config['model']['shared_encoder']
    ).to(device)

    checkpoint_path = 'checkpoints/best_model.pth'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best metric: {checkpoint['best_metric']:.4f}")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    model.eval()

    # Model efficiency metrics
    num_params = count_parameters(model)
    flops = estimate_flops(model, device=device)
    fps = measure_fps(model, device=device)

    print("\nModel Efficiency:")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"  FPS: {fps:.2f} frames/second")

    # Load test dataset
    data_config = config['data']
    test_transform = get_val_transforms(tuple(data_config['image_size']))

    test_dataset = FUMEDataset(
        paired_csv=data_config['paired_test_csv'],
        dataset_root=data_config['dataset_root'],
        transform=test_transform,
        modality_dropout=0.0,
        is_training=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    print(f"\nTest samples: {len(test_dataset)}")

    # Evaluate
    seg_metrics = SegmentationMetrics(num_classes=config['model']['num_seg_classes'])
    cls_metrics = ClassificationMetrics(
        num_classes=config['model']['num_classes'],
        class_names=config['data']['class_names']
    )

    all_preds = []
    all_labels = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            co2_frame = batch['co2_frame'].to(device)
            ch4_frame = batch['ch4_frame'].to(device)
            co2_mask = batch['co2_mask'].to(device)
            class_label = batch['class_label'].to(device)
            modality_mask = batch['modality_mask'].to(device)

            outputs = model(co2_frame, ch4_frame, modality_mask)

            pred_cls = outputs['cls_logits'].argmax(dim=1)
            pred_seg = outputs['co2_seg_logits'].argmax(dim=1)

            cls_metrics.update(pred_cls, class_label)
            seg_metrics.update(pred_seg, co2_mask)

            all_preds.extend(pred_cls.cpu().numpy())
            all_labels.extend(class_label.cpu().numpy())

    seg_results = seg_metrics.compute()
    cls_results = cls_metrics.compute()

    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)

    print("\nSegmentation Metrics:")
    for key in ['mean_iou', 'mean_dice', 'pixel_accuracy']:
        if key in seg_results:
            print(f"  {key}: {seg_results[key]:.4f}")

    print("\nClassification Metrics:")
    for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'cohens_kappa']:
        if key in cls_results:
            print(f"  {key}: {cls_results[key]:.4f}")

    print("\nPer-class F1 Scores:")
    class_names = config['data']['class_names']
    for name in class_names:
        key = f'{name}_f1'
        if key in cls_results:
            print(f"  {name}: {cls_results[key]:.4f}")

    # Save confusion matrix
    Path('results').mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to results/confusion_matrix.png")

    # Save predictions
    save_predictions(test_dataset, model, device, class_names, num_samples=100)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': [
            'Parameters (M)', 'FLOPs (G)', 'FPS',
            'mIoU', 'Mean Dice', 'Pixel Accuracy',
            'Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', "Cohen's Kappa",
            'Healthy F1', 'Transitional F1', 'Acidotic F1'
        ],
        'Value': [
            f"{num_params/1e6:.2f}", f"{flops/1e9:.2f}", f"{fps:.2f}",
            f"{seg_results.get('mean_iou', 0):.4f}",
            f"{seg_results.get('mean_dice', 0):.4f}",
            f"{seg_results.get('pixel_accuracy', 0):.4f}",
            f"{cls_results.get('accuracy', 0):.4f}",
            f"{cls_results.get('balanced_accuracy', 0):.4f}",
            f"{cls_results.get('macro_f1', 0):.4f}",
            f"{cls_results.get('weighted_f1', 0):.4f}",
            f"{cls_results.get('cohens_kappa', 0):.4f}",
            f"{cls_results.get('Healthy_f1', 0):.4f}",
            f"{cls_results.get('Transitional_f1', 0):.4f}",
            f"{cls_results.get('Acidotic_f1', 0):.4f}"
        ]
    })

    results_df.to_csv('results/test_results.csv', index=False)
    print("Results saved to results/test_results.csv")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
