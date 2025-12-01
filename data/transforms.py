"""
Data Augmentation and Transforms for FUME
Uses Albumentations library
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size=(640, 480), additional_targets=None):
    """
    Training augmentation pipeline

    Args:
        image_size: (width, height)
        additional_targets: Dict for multi-image augmentation
                          e.g., {'ch4_image': 'image', 'co2_mask': 'mask', 'ch4_mask': 'mask'}
    """
    if additional_targets is None:
        additional_targets = {
            'ch4_image': 'image',
            'co2_mask': 'mask',
            'ch4_mask': 'mask'
        }

    transforms = A.Compose([
        # Geometric transforms (apply to all images and masks)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),
        A.ElasticTransform(
            alpha=1.0,
            sigma=50,
            alpha_affine=50,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),

        # Pixel-level transforms (only for images, not masks)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Resize to standard size
        A.Resize(height=image_size[1], width=image_size[0], interpolation=cv2.INTER_LINEAR),

        # Normalize (assuming grayscale images)
        A.Normalize(
            mean=[0.5],  # Grayscale normalization
            std=[0.5],
            max_pixel_value=255.0
        ),

        # Convert to tensor
        ToTensorV2()
    ], additional_targets=additional_targets)

    return transforms


def get_val_transforms(image_size=(640, 480), additional_targets=None):
    """
    Validation/Test augmentation pipeline (no augmentation, only preprocessing)
    """
    if additional_targets is None:
        additional_targets = {
            'ch4_image': 'image',
            'co2_mask': 'mask',
            'ch4_mask': 'mask'
        }

    transforms = A.Compose([
        A.Resize(height=image_size[1], width=image_size[0], interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=[0.5],
            std=[0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets=additional_targets)

    return transforms


def get_test_time_augmentation():
    """
    Test-Time Augmentation (TTA) pipeline
    Returns list of transforms for ensemble prediction
    """
    tta_transforms = [
        # Original
        get_val_transforms(),

        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(height=480, width=640),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ], additional_targets={'ch4_image': 'image', 'co2_mask': 'mask', 'ch4_mask': 'mask'}),

        # Scale variations
        A.Compose([
            A.Resize(height=512, width=672),
            A.CenterCrop(height=480, width=640),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ], additional_targets={'ch4_image': 'image', 'co2_mask': 'mask', 'ch4_mask': 'mask'}),
    ]

    return tta_transforms


def get_minimal_transforms(image_size=(640, 480)):
    """
    Minimal transforms for baseline models (no augmentation)
    Used for ablation studies on augmentation impact
    """
    transforms = A.Compose([
        A.Resize(height=image_size[1], width=image_size[0]),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

    return transforms
