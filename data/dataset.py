"""
FUME Dataset Loader with Modality Dropout
Handles paired CO2-CH4 samples with missing modality support
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FUMEDataset(Dataset):
    """
    Dataset for FUME model with dual-gas support

    Features:
    - Loads paired CO2-CH4 frames
    - Handles missing modalities (zero-padding)
    - Supports modality dropout during training
    - Returns both frames, masks, and classification labels
    """

    def __init__(
        self,
        paired_csv: str,
        dataset_root: str,
        transform=None,
        modality_dropout: float = 0.0,
        is_training: bool = True,
        return_original: bool = False
    ):
        """
        Args:
            paired_csv: Path to paired annotations CSV
            dataset_root: Root directory of dataset
            transform: Albumentations transform
            modality_dropout: Probability of dropping one modality during training (0.0-1.0)
            is_training: Training mode flag
            return_original: Return original non-augmented samples only
        """
        self.paired_df = pd.read_csv(paired_csv)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.modality_dropout = modality_dropout if is_training else 0.0
        self.is_training = is_training
        self.return_original = return_original

        # Filter for original samples if requested
        if return_original:
            # This would require adding is_original flag to paired CSV
            # For now, we'll skip this filtering
            pass

        # Class mapping
        self.class_to_idx = {
            'Healthy': 0,
            'Transitional': 1,
            'Acidotic': 2
        }

        # Calculate class weights for weighted sampling
        class_counts = self.paired_df['class_name'].value_counts()
        self.class_weights = self._calculate_class_weights(class_counts)

        logger.info(f"Loaded {len(self.paired_df)} paired samples")
        logger.info(f"  Fully paired: {self.paired_df['is_paired'].sum()}")
        logger.info(f"  Modality dropout: {self.modality_dropout}")
        logger.info(f"  Class distribution: {class_counts.to_dict()}")

    def _calculate_class_weights(self, class_counts: pd.Series) -> Dict[str, float]:
        """Calculate inverse frequency weights for class balancing"""
        total = class_counts.sum()
        weights = {}
        for class_name, count in class_counts.items():
            weights[class_name] = total / (len(class_counts) * count)
        return weights

    def __len__(self) -> int:
        return len(self.paired_df)

    def _load_image(self, path: Optional[str]) -> np.ndarray:
        """Load grayscale image, return zero-padded array if path is None"""
        if path is None or pd.isna(path):
            # Return zero-padded image
            return np.zeros((480, 640), dtype=np.uint8)

        # Handle path format
        if isinstance(path, str):
            # Remove 'mmseg_dataset\' prefix and convert backslashes to forward slashes
            path = path.replace('mmseg_dataset\\', '').replace('mmseg_dataset/', '')
            path = path.replace('\\', '/')  # Convert Windows backslashes to Unix forward slashes
            full_path = self.dataset_root / path
        else:
            full_path = self.dataset_root / str(path)

        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}, using zero-padding")
            return np.zeros((480, 640), dtype=np.uint8)

        img = Image.open(full_path).convert('L')  # Grayscale
        return np.array(img)

    def _load_mask(self, path: Optional[str]) -> np.ndarray:
        """Load segmentation mask, return zero mask if path is None"""
        if path is None or pd.isna(path):
            return np.zeros((480, 640), dtype=np.uint8)

        # Handle path format
        if isinstance(path, str):
            path = path.replace('mmseg_dataset\\', '').replace('mmseg_dataset/', '')
            path = path.replace('\\', '/')  # Convert Windows backslashes to Unix forward slashes
            full_path = self.dataset_root / path
        else:
            full_path = self.dataset_root / str(path)

        if not full_path.exists():
            logger.warning(f"Mask not found: {full_path}, using zero mask")
            return np.zeros((480, 640), dtype=np.uint8)

        mask = Image.open(full_path).convert('L')
        return np.array(mask)

    def _apply_modality_dropout(self, row: pd.Series) -> Tuple[bool, bool]:
        """
        Apply modality dropout during training

        Returns:
            (use_co2, use_ch4): Boolean flags indicating which modalities to use
        """
        if self.modality_dropout == 0.0 or not self.is_training:
            return True, True

        # If sample is already unpaired, don't drop the available modality
        if not row['is_paired']:
            if row['missing_modality'] == 'co2':
                return False, True  # Only CH4 available
            else:
                return True, False  # Only CO2 available

        # For paired samples, randomly drop one modality
        if np.random.rand() < self.modality_dropout:
            # Randomly choose which modality to drop
            if np.random.rand() < 0.5:
                return True, False  # Drop CH4
            else:
                return False, True  # Drop CO2
        else:
            return True, True  # Use both

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset

        Returns:
            Dictionary with keys:
                - co2_frame: (1, H, W) tensor
                - ch4_frame: (1, H, W) tensor
                - co2_mask: (H, W) tensor with values [0, 1, 2]
                - ch4_mask: (H, W) tensor with values [0, 1, 2]
                - class_label: int (0=Healthy, 1=Transitional, 2=Acidotic)
                - ph_value: float
                - is_paired: bool
                - modality_mask: (2,) tensor [use_co2, use_ch4] as floats
        """
        row = self.paired_df.iloc[idx]

        # Apply modality dropout
        use_co2, use_ch4 = self._apply_modality_dropout(row)

        # Load images and masks
        co2_frame = self._load_image(row['co2_frame'] if use_co2 else None)
        ch4_frame = self._load_image(row['ch4_frame'] if use_ch4 else None)
        co2_mask = self._load_mask(row['co2_mask'] if use_co2 else None)
        ch4_mask = self._load_mask(row['ch4_mask'] if use_ch4 else None)

        # Apply transforms
        if self.transform is not None:
            # Albumentations expects HWC format, but grayscale is HW
            # We need to add channel dimension
            co2_frame_3ch = np.stack([co2_frame] * 3, axis=-1)  # Convert to 3-channel for albumentations
            ch4_frame_3ch = np.stack([ch4_frame] * 3, axis=-1)

            transformed = self.transform(
                image=co2_frame_3ch,
                ch4_image=ch4_frame_3ch,
                mask=co2_mask,
                co2_mask=co2_mask,
                ch4_mask=ch4_mask
            )

            # Extract single channel (albumentations will return 3xHxW tensor)
            co2_frame = transformed['image'][0:1]  # Take first channel only
            ch4_frame = transformed['ch4_image'][0:1]
            co2_mask = transformed['co2_mask']
            ch4_mask = transformed['ch4_mask']
        else:
            # Manual conversion to tensor
            co2_frame = torch.from_numpy(co2_frame).unsqueeze(0).float() / 255.0
            ch4_frame = torch.from_numpy(ch4_frame).unsqueeze(0).float() / 255.0
            co2_mask = torch.from_numpy(co2_mask).long()
            ch4_mask = torch.from_numpy(ch4_mask).long()

        # Get class label
        class_label = self.class_to_idx[row['class_name']]

        # Create modality mask (indicates which modalities are available)
        modality_mask = torch.tensor([float(use_co2), float(use_ch4)], dtype=torch.float32)

        return {
            'co2_frame': co2_frame,
            'ch4_frame': ch4_frame,
            'co2_mask': co2_mask,
            'ch4_mask': ch4_mask,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'ph_value': torch.tensor(row['ph_value'], dtype=torch.float32),
            'is_paired': torch.tensor(row['is_paired'], dtype=torch.bool),
            'modality_mask': modality_mask,
            'sample_id': idx
        }

    def get_sample_weights(self) -> np.ndarray:
        """
        Get sample weights for weighted random sampling
        Higher weights for underrepresented classes
        """
        weights = np.array([
            self.class_weights[row['class_name']]
            for _, row in self.paired_df.iterrows()
        ])
        return weights


class SingleGasDataset(Dataset):
    """
    Dataset for baseline models (single gas type)
    Used for CO2-only and CH4-only baselines
    """

    def __init__(
        self,
        annotations_csv: str,
        dataset_root: str,
        gas_type: str = 'co2',
        transform=None,
        is_training: bool = True
    ):
        """
        Args:
            annotations_csv: Path to original annotations CSV
            dataset_root: Root directory of dataset
            gas_type: 'co2' or 'ch4'
            transform: Albumentations transform
            is_training: Training mode flag
        """
        df = pd.read_csv(annotations_csv)
        self.df = df[df['gas_type'] == gas_type].reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.gas_type = gas_type
        self.is_training = is_training

        self.class_to_idx = {
            'Healthy': 0,
            'Transitional': 1,
            'Acidotic': 2
        }

        logger.info(f"Loaded {len(self.df)} {gas_type.upper()} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image and mask
        frame_path = row['frame_path'].replace('mmseg_dataset\\', '').replace('mmseg_dataset/', '')
        frame_path = frame_path.replace('\\', '/')  # Convert Windows backslashes to Unix forward slashes
        mask_path = row['mask_path'].replace('mmseg_dataset\\', '').replace('mmseg_dataset/', '')
        mask_path = mask_path.replace('\\', '/')  # Convert Windows backslashes to Unix forward slashes

        frame = Image.open(self.dataset_root / frame_path).convert('L')
        mask = Image.open(self.dataset_root / mask_path).convert('L')

        frame = np.array(frame)
        mask = np.array(mask)

        # Apply transforms
        if self.transform is not None:
            frame_3ch = np.stack([frame] * 3, axis=-1)
            transformed = self.transform(image=frame_3ch, mask=mask)
            frame = transformed['image'][0:1]  # Take first channel
            mask = transformed['mask']
        else:
            frame = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).long()

        class_label = self.class_to_idx[row['class_name']]

        return {
            'frame': frame,
            'mask': mask,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'ph_value': torch.tensor(row['ph_value'], dtype=torch.float32),
            'gas_type': self.gas_type
        }
