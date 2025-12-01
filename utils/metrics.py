"""
Metrics for FUME evaluation
- Segmentation: IoU, Dice, Pixel Accuracy
- Classification: Balanced Accuracy, F1, Confusion Matrix
"""

import torch
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score
)
from typing import Dict, List


class SegmentationMetrics:
    """Calculate segmentation metrics (IoU, Dice, Pixel Accuracy)"""

    def __init__(self, num_classes: int = 3, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions

        Args:
            preds: (B, H, W) predicted class indices
            targets: (B, H, W) ground truth class indices
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Remove ignored pixels
        valid_mask = targets != self.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        # Pixel accuracy
        self.correct_pixels += np.sum(preds == targets)
        self.total_pixels += len(targets)

        # IoU calculation
        for cls in range(self.num_classes):
            pred_mask = preds == cls
            target_mask = targets == cls

            self.intersection[cls] += np.sum(pred_mask & target_mask)
            self.union[cls] += np.sum(pred_mask | target_mask)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        # Pixel accuracy
        pixel_acc = self.correct_pixels / max(self.total_pixels, 1)

        # IoU per class
        iou_per_class = self.intersection / np.maximum(self.union, 1)

        # Mean IoU
        mean_iou = np.mean(iou_per_class)

        # Dice coefficient
        dice_per_class = 2 * self.intersection / np.maximum(self.intersection + self.union, 1)
        mean_dice = np.mean(dice_per_class)

        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'iou_background': iou_per_class[0],
            'iou_tube': iou_per_class[1],
            'iou_gas': iou_per_class[2],  # Most important!
            'dice_background': dice_per_class[0],
            'dice_tube': dice_per_class[1],
            'dice_gas': dice_per_class[2]
        }


class ClassificationMetrics:
    """Calculate classification metrics (Balanced Acc, F1, Confusion Matrix)"""

    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        self.num_classes = num_classes
        if class_names is None:
            self.class_names = ['Healthy', 'Transitional', 'Acidotic']
        else:
            self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions

        Args:
            preds: (B,) predicted class indices
            targets: (B,) ground truth class indices
        """
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_targets.extend(targets.cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # Overall accuracy
        accuracy = np.mean(preds == targets)

        # Balanced accuracy (critical for imbalanced dataset)
        balanced_acc = balanced_accuracy_score(targets, preds)

        # Per-class F1 scores
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )

        # Macro F1 (equal weight to all classes)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)

        # Weighted F1
        weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)

        # Cohen's Kappa
        kappa = cohen_kappa_score(targets, preds)

        # Confusion matrix
        cm = confusion_matrix(targets, preds)

        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'cohens_kappa': kappa,
            'confusion_matrix': cm
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            results[f'{class_name}_precision'] = precision[i]
            results[f'{class_name}_recall'] = recall[i]
            results[f'{class_name}_f1'] = f1[i]
            results[f'{class_name}_support'] = support[i]

        return results

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(np.array(self.all_targets), np.array(self.all_preds))


def compute_multi_task_metric(
    seg_metric: float,
    cls_metric: float,
    seg_weight: float = 0.5,
    cls_weight: float = 0.5
) -> float:
    """
    Compute combined multi-task metric

    Args:
        seg_metric: Segmentation metric (e.g., mIoU)
        cls_metric: Classification metric (e.g., balanced accuracy)
        seg_weight: Weight for segmentation
        cls_weight: Weight for classification

    Returns:
        Combined metric value
    """
    return seg_weight * seg_metric + cls_weight * cls_metric
