"""
Focal Loss and Dice Loss for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for classification
    FL(pt) = -α(1-pt)^γ log(pt)

    Handles class imbalance by down-weighting easy examples
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Args:
            alpha: Class weights tensor of shape (num_classes,)
                  For FUME: [1.0, 8.0, 1.2] for [Healthy, Transitional, Acidotic]
            gamma: Focusing parameter (default 2.0)
            reduction: 'none', 'mean', or 'sum'
            ignore_index: Ignore this class in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class indices

        Returns:
            Focal loss value
        """
        # Ensure targets are long dtype for indexing
        targets = targets.long()

        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Dice = 2*|X∩Y| / (|X| + |Y|)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) class indices

        Returns:
            Dice loss value
        """
        # Ensure targets are long dtype for one_hot encoding
        targets = targets.long()

        num_classes = inputs.shape[1]

        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Get probabilities
        inputs_soft = F.softmax(inputs, dim=1)

        # Flatten spatial dimensions
        inputs_flat = inputs_soft.view(inputs_soft.size(0), inputs_soft.size(1), -1)  # (B, C, H*W)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)

        # Calculate Dice coefficient per class
        intersection = (inputs_flat * targets_flat).sum(dim=2)  # (B, C)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        dice_loss = 1 - dice

        # Average over classes and batch
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss for segmentation
    Handles both class imbalance and boundary precision
    """

    def __init__(
        self,
        num_classes: int = 3,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) class indices

        Returns:
            Combined loss value
        """
        # Ensure targets are long dtype
        targets = targets.long()

        # Focal loss (pixel-wise classification)
        B, C, H, W = inputs.shape
        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        targets_flat = targets.view(-1)  # (B*H*W,)

        focal = self.focal_loss(inputs_flat, targets_flat)

        # Dice loss (region overlap)
        dice = self.dice_loss(inputs, targets)

        # Combined loss
        total_loss = self.focal_weight * focal + self.dice_weight * dice

        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Class-weighted Cross Entropy Loss
    Simple alternative to Focal Loss
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)

        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)
