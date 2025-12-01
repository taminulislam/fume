"""
Multi-Task Loss for FUME
Combines segmentation and classification losses
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .focal_loss import FocalLoss, FocalDiceLoss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for FUME model
    L_total = λ_seg * L_seg + λ_cls * L_cls

    Handles:
    - Dual segmentation (CO2 + CH4 masks)
    - Classification (pH class)
    - Loss balancing
    """

    def __init__(
        self,
        seg_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        cls_alpha: Optional[torch.Tensor] = None,
        cls_gamma: float = 2.0,
        seg_alpha: Optional[torch.Tensor] = None,
        seg_gamma: float = 2.0,
        use_focal_dice: bool = True
    ):
        """
        Args:
            seg_loss_weight: Weight for segmentation loss (λ_seg)
            cls_loss_weight: Weight for classification loss (λ_cls)
            cls_alpha: Class weights for classification [Healthy, Trans, Acidotic]
                      Default: [1.0, 8.0, 1.2] to handle Transitional imbalance
            cls_gamma: Focal loss gamma for classification
            seg_alpha: Class weights for segmentation [background, tube, gas]
            seg_gamma: Focal loss gamma for segmentation
            use_focal_dice: Use Focal+Dice loss for segmentation (True) or just Focal (False)
        """
        super().__init__()

        self.seg_loss_weight = seg_loss_weight
        self.cls_loss_weight = cls_loss_weight

        # Classification loss (Focal Loss for class imbalance)
        if cls_alpha is None:
            cls_alpha = torch.tensor([1.0, 8.0, 1.2])  # [Healthy, Transitional, Acidotic]

        self.cls_loss_fn = FocalLoss(alpha=cls_alpha, gamma=cls_gamma, reduction='mean')

        # Segmentation loss
        if use_focal_dice:
            self.seg_loss_fn = FocalDiceLoss(
                num_classes=3,  # background, tube, gas
                alpha=seg_alpha,
                gamma=seg_gamma,
                dice_weight=0.5,
                focal_weight=0.5
            )
        else:
            self.seg_loss_fn = FocalLoss(alpha=seg_alpha, gamma=seg_gamma, reduction='mean')

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary with keys:
                - 'cls_logits': (B, 3) classification logits
                - 'co2_seg_logits': (B, 3, H, W) CO2 segmentation logits
                - 'ch4_seg_logits': (B, 3, H, W) CH4 segmentation logits

            targets: Dictionary with keys:
                - 'class_label': (B,) class indices
                - 'co2_mask': (B, H, W) CO2 segmentation masks
                - 'ch4_mask': (B, H, W) CH4 segmentation masks
                - 'modality_mask': (B, 2) [use_co2, use_ch4] binary flags

        Returns:
            Dictionary with keys:
                - 'total_loss': Combined loss
                - 'cls_loss': Classification loss
                - 'seg_loss': Segmentation loss (average of CO2 and CH4)
                - 'co2_seg_loss': CO2 segmentation loss
                - 'ch4_seg_loss': CH4 segmentation loss
        """
        # Classification loss (always computed)
        cls_loss = self.cls_loss_fn(outputs['cls_logits'], targets['class_label'])

        # Segmentation losses (computed per modality)
        modality_mask = targets['modality_mask']  # (B, 2) [use_co2, use_ch4]

        # CO2 segmentation loss
        if modality_mask[:, 0].sum() > 0:  # If any samples have CO2
            co2_indices = modality_mask[:, 0] > 0
            co2_seg_loss = self.seg_loss_fn(
                outputs['co2_seg_logits'][co2_indices],
                targets['co2_mask'][co2_indices]
            )
        else:
            co2_seg_loss = torch.tensor(0.0, device=outputs['cls_logits'].device)

        # CH4 segmentation loss
        if modality_mask[:, 1].sum() > 0:  # If any samples have CH4
            ch4_indices = modality_mask[:, 1] > 0
            ch4_seg_loss = self.seg_loss_fn(
                outputs['ch4_seg_logits'][ch4_indices],
                targets['ch4_mask'][ch4_indices]
            )
        else:
            ch4_seg_loss = torch.tensor(0.0, device=outputs['cls_logits'].device)

        # Average segmentation loss
        num_modalities = (modality_mask[:, 0].sum() > 0).float() + (modality_mask[:, 1].sum() > 0).float()
        if num_modalities > 0:
            seg_loss = (co2_seg_loss + ch4_seg_loss) / num_modalities
        else:
            seg_loss = torch.tensor(0.0, device=outputs['cls_logits'].device)

        # Total loss
        total_loss = (
            self.cls_loss_weight * cls_loss +
            self.seg_loss_weight * seg_loss
        )

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'seg_loss': seg_loss,
            'co2_seg_loss': co2_seg_loss,
            'ch4_seg_loss': ch4_seg_loss
        }

    def update_weights(self, seg_weight: Optional[float] = None, cls_weight: Optional[float] = None):
        """
        Update loss weights during training (optional)
        Useful for curriculum learning or dynamic weight adjustment
        """
        if seg_weight is not None:
            self.seg_loss_weight = seg_weight
        if cls_weight is not None:
            self.cls_loss_weight = cls_weight
