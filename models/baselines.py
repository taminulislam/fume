"""
Baseline Models for FUME Comparison

5 Baselines:
1. Segmentation-Only Model
2. Classification-Only Model
3. Gas-Aware Classifier
4. Early Fusion Model
5. Traditional ML Baseline
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from .backbones import ResNet50Encoder
from .heads import DeepLabV3PlusHead, ClassificationHead, SegmentationHead


# Baseline 1: Segmentation-Only Model
class SegmentationOnlyModel(nn.Module):
    """
    Baseline 1: Pure segmentation model
    Purpose: Establish segmentation performance ceiling

    Input: Single grayscale frame (CO2 or CH4)
    Output: 3-class segmentation mask only
    """

    def __init__(self, num_seg_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)
        self.seg_head = DeepLabV3PlusHead(
            in_channels=2048,
            num_classes=num_seg_classes,
            low_level_channels=256
        )

    def forward(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (frame.shape[2], frame.shape[3])
        features = self.encoder(frame)
        seg_logits = self.seg_head(features[-1], features[1], target_size)

        return {'seg_logits': seg_logits}


# Baseline 2: Classification-Only Model
class ClassificationOnlyModel(nn.Module):
    """
    Baseline 2: Pure classification model
    Purpose: Establish classification performance ceiling without segmentation

    Input: Single grayscale frame
    Output: 3-class pH prediction only
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)
        self.cls_head = ClassificationHead(in_channels=2048, num_classes=num_classes)

    def forward(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(frame)
        cls_logits = self.cls_head(features[-1])

        return {'cls_logits': cls_logits}


# Baseline 3: Gas-Aware Classifier
class GasAwareClassifier(nn.Module):
    """
    Baseline 3: Classification with explicit gas type embedding
    Purpose: Show importance of gas type information

    Input: Frame + gas type one-hot vector [CO2, CH4]
    Output: 3-class pH prediction
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained, in_channels=1)

        # Gas type embedding
        self.gas_embedding = nn.Embedding(2, 128)  # 2 gas types: CO2=0, CH4=1

        # Classifier with gas embedding
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 128, 512),  # Encoder features + gas embedding
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        frame: torch.Tensor,
        gas_type: torch.Tensor  # (B,) with values 0 (CO2) or 1 (CH4)
    ) -> Dict[str, torch.Tensor]:
        # Encode frame
        features = self.encoder(frame)
        pooled = self.global_pool(features[-1]).view(features[-1].size(0), -1)  # (B, 2048)

        # Get gas embedding
        gas_emb = self.gas_embedding(gas_type)  # (B, 128)

        # Concatenate
        combined = torch.cat([pooled, gas_emb], dim=1)  # (B, 2048+128)

        # Classify
        cls_logits = self.classifier(combined)

        return {'cls_logits': cls_logits}


# Baseline 4: Early Fusion Model
class EarlyFusionModel(nn.Module):
    """
    Baseline 4: Early fusion of CO2 and CH4 (concatenate then encode)
    Purpose: Show that dual-stream late fusion > early fusion

    Input: CO2 frame + CH4 frame concatenated as 2-channel input
    Output: Segmentation + Classification
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_seg_classes: int = 3,
        pretrained: bool = True
    ):
        super().__init__()

        # Single encoder for concatenated input (2 channels)
        self.encoder = ResNet50Encoder(pretrained=pretrained, in_channels=2)

        # Task heads
        self.seg_head = DeepLabV3PlusHead(
            in_channels=2048,
            num_classes=num_seg_classes,
            low_level_channels=256
        )
        self.cls_head = ClassificationHead(in_channels=2048, num_classes=num_classes)

    def forward(
        self,
        co2_frame: torch.Tensor,
        ch4_frame: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        target_size = (co2_frame.shape[2], co2_frame.shape[3])

        # Early fusion: concatenate along channel dimension
        fused_input = torch.cat([co2_frame, ch4_frame], dim=1)  # (B, 2, H, W)

        # Single encoder
        features = self.encoder(fused_input)

        # Task heads
        seg_logits = self.seg_head(features[-1], features[1], target_size)
        cls_logits = self.cls_head(features[-1])

        return {
            'cls_logits': cls_logits,
            'seg_logits': seg_logits  # Single segmentation for fused input
        }


# Baseline 5: Traditional ML Baseline
class TraditionalMLBaseline:
    """
    Baseline 5: Random Forest on hand-crafted features
    Purpose: Show deep learning necessity

    Features extracted from gas regions:
    - Gas region area
    - Mean intensity
    - Std intensity
    - Shape features (aspect ratio, compactness)
    - Gas type (one-hot)
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )

        self.is_fitted = False

    def extract_features(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        gas_type: int
    ) -> np.ndarray:
        """
        Extract hand-crafted features from frame and mask

        Args:
            frame: (H, W) grayscale image
            mask: (H, W) segmentation mask with classes [0=bg, 1=tube, 2=gas]
            gas_type: 0 for CO2, 1 for CH4

        Returns:
            features: (n_features,) feature vector
        """
        features = []

        # Gas region mask
        gas_mask = (mask == 2).astype(np.uint8)

        # 1. Area features
        gas_area = np.sum(gas_mask)
        total_area = frame.shape[0] * frame.shape[1]
        features.append(gas_area / total_area)  # Normalized gas area

        # 2. Intensity features
        if gas_area > 0:
            gas_pixels = frame[gas_mask == 1]
            features.append(np.mean(gas_pixels))  # Mean intensity
            features.append(np.std(gas_pixels))   # Std intensity
            features.append(np.median(gas_pixels))  # Median intensity
            features.append(np.percentile(gas_pixels, 25))  # Q1
            features.append(np.percentile(gas_pixels, 75))  # Q3
        else:
            features.extend([0, 0, 0, 0, 0])

        # 3. Shape features
        import cv2
        contours, _ = cv2.findContours(gas_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            # Compactness
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0
            features.append(compactness)

            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            if h > 0:
                aspect_ratio = w / h
            else:
                aspect_ratio = 0
            features.append(aspect_ratio)

            # Extent (area / bounding box area)
            bbox_area = w * h
            if bbox_area > 0:
                extent = area / bbox_area
            else:
                extent = 0
            features.append(extent)
        else:
            features.extend([0, 0, 0])

        # 4. Gas type (one-hot)
        features.append(1 if gas_type == 0 else 0)  # CO2
        features.append(1 if gas_type == 1 else 0)  # CH4

        return np.array(features)

    def fit(
        self,
        frames: list,
        masks: list,
        gas_types: list,
        labels: list
    ):
        """
        Train the Random Forest model

        Args:
            frames: List of (H, W) grayscale frames
            masks: List of (H, W) segmentation masks
            gas_types: List of gas type indices (0=CO2, 1=CH4)
            labels: List of class labels (0=Healthy, 1=Trans, 2=Acidotic)
        """
        # Extract features for all samples
        X = []
        for frame, mask, gas_type in zip(frames, masks, gas_types):
            feats = self.extract_features(frame, mask, gas_type)
            X.append(feats)

        X = np.array(X)
        y = np.array(labels)

        # Train Random Forest
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(
        self,
        frames: list,
        masks: list,
        gas_types: list
    ) -> np.ndarray:
        """
        Predict pH class for new samples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = []
        for frame, mask, gas_type in zip(frames, masks, gas_types):
            feats = self.extract_features(frame, mask, gas_type)
            X.append(feats)

        X = np.array(X)
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, frames: list, masks: list, gas_types: list) -> np.ndarray:
        """Get class probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = []
        for frame, mask, gas_type in zip(frames, masks, gas_types):
            feats = self.extract_features(frame, mask, gas_type)
            X.append(feats)

        X = np.array(X)
        probabilities = self.model.predict_proba(X)

        return probabilities
