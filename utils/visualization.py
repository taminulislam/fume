"""
Visualization utilities for FUME
Plotting training curves, segmentation results, confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from sklearn.metrics import confusion_matrix
import cv2


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dict of metric name -> list of values
        val_metrics: Dict of metric name -> list of values
        save_path: Path to save figure
    """
    num_metrics = len(train_metrics) + 1  # +1 for loss
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    # Plot loss
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metrics
    for idx, (metric_name, train_vals) in enumerate(train_metrics.items(), 1):
        val_vals = val_metrics.get(metric_name, [])

        axes[idx].plot(train_vals, label=f'Train {metric_name}', linewidth=2)
        if val_vals:
            axes[idx].plot(val_vals, label=f'Val {metric_name}', linewidth=2)

        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(f'{metric_name} Over Epochs')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Normalize by row (True) or show counts (False)
        save_path: Path to save figure
        title: Plot title
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title += " (Normalized)"
    else:
        fmt = 'd'

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")

    plt.show()


def visualize_segmentation(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    class_colors: Optional[Dict[int, tuple]] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.5
):
    """
    Visualize segmentation results with overlay

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        pred_mask: Predicted mask (H, W) with class indices
        true_mask: Ground truth mask (H, W) with class indices
        class_colors: Dict mapping class ID to RGB color tuple
        save_path: Path to save figure
        alpha: Overlay transparency (0-1)
    """
    # Default colors
    if class_colors is None:
        class_colors = {
            0: (0, 0, 0),        # Background - Black
            1: (128, 128, 128),  # Tube - Gray
            2: (255, 0, 0)       # Gas - Red
        }

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create colored masks
    pred_colored = np.zeros_like(image)
    for class_id, color in class_colors.items():
        pred_colored[pred_mask == class_id] = color

    # Prepare subplots
    if true_mask is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # True mask colored
        true_colored = np.zeros_like(image)
        for class_id, color in class_colors.items():
            true_colored[true_mask == class_id] = color

        # Plot
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(true_colored)
        axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(pred_colored)
        axes[1, 0].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Overlay
        overlay = cv2.addWeighted(image, 1-alpha, pred_colored, alpha, 0)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred_colored)
        axes[1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        overlay = cv2.addWeighted(image, 1-alpha, pred_colored, alpha, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Segmentation visualization saved to {save_path}")

    plt.show()


def visualize_dual_gas_results(
    co2_frame: np.ndarray,
    ch4_frame: np.ndarray,
    co2_pred: np.ndarray,
    ch4_pred: np.ndarray,
    co2_true: Optional[np.ndarray] = None,
    ch4_true: Optional[np.ndarray] = None,
    class_label: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize dual-gas FUME results

    Args:
        co2_frame: CO2 input frame
        ch4_frame: CH4 input frame
        co2_pred: CO2 predicted mask
        ch4_pred: CH4 predicted mask
        co2_true: CO2 ground truth mask
        ch4_true: CH4 ground truth mask
        class_label: Predicted class label
        class_names: List of class names
        save_path: Path to save figure
    """
    class_colors = {0: (0, 0, 0), 1: (128, 128, 128), 2: (255, 0, 0)}

    if co2_true is not None:
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # CO2 column
    axes[0, 0].imshow(co2_frame, cmap='gray')
    axes[0, 0].set_title('CO2 Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    co2_colored = np.zeros((co2_pred.shape[0], co2_pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        co2_colored[co2_pred == class_id] = color
    axes[1, 0].imshow(co2_colored)
    axes[1, 0].set_title('CO2 Prediction', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # CH4 column
    axes[0, 1].imshow(ch4_frame, cmap='gray')
    axes[0, 1].set_title('CH4 Input', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    ch4_colored = np.zeros((ch4_pred.shape[0], ch4_pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        ch4_colored[ch4_pred == class_id] = color
    axes[1, 1].imshow(ch4_colored)
    axes[1, 1].set_title('CH4 Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Ground truth row
    if co2_true is not None:
        co2_true_colored = np.zeros((co2_true.shape[0], co2_true.shape[1], 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            co2_true_colored[co2_true == class_id] = color
        axes[2, 0].imshow(co2_true_colored)
        axes[2, 0].set_title('CO2 Ground Truth', fontsize=14, fontweight='bold')
        axes[2, 0].axis('off')

        ch4_true_colored = np.zeros((ch4_true.shape[0], ch4_true.shape[1], 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            ch4_true_colored[ch4_true == class_id] = color
        axes[2, 1].imshow(ch4_true_colored)
        axes[2, 1].set_title('CH4 Ground Truth', fontsize=14, fontweight='bold')
        axes[2, 1].axis('off')

    # Add classification result
    if class_label is not None and class_names is not None:
        fig.suptitle(f'Predicted Class: {class_names[class_label]}',
                    fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dual-gas visualization saved to {save_path}")

    plt.show()


def plot_per_class_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot per-class metrics (F1, Precision, Recall)

    Args:
        metrics_dict: Dict of model_name -> {class_name: metric_value}
        class_names: List of class names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['F1', 'Precision', 'Recall']

    for idx, metric in enumerate(metrics):
        data = []
        labels = []

        for model_name, class_metrics in metrics_dict.items():
            values = [class_metrics.get(f"{cn}_{metric.lower()}", 0) for cn in class_names]
            data.append(values)
            labels.append(model_name)

        x = np.arange(len(class_names))
        width = 0.8 / len(data)

        for i, (values, label) in enumerate(zip(data, labels)):
            offset = width * (i - len(data)/2 + 0.5)
            axes[idx].bar(x + offset, values, width, label=label, alpha=0.8)

        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Score by Class')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(class_names)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics plot saved to {save_path}")

    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['mIoU', 'Balanced_Acc', 'Macro_F1'],
    save_path: Optional[str] = None
):
    """
    Compare multiple models on key metrics

    Args:
        results: Dict of model_name -> {metric_name: value}
        metrics: List of metrics to compare
        save_path: Path to save figure
    """
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(metric, 0) for metric in metrics]
        offset = width * (i - len(model_names)/2 + 0.5)
        ax.bar(x + offset, values, width, label=model_name, alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Comparison on Key Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to {save_path}")

    plt.show()
