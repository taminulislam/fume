"""
Backbone networks for FUME
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNet50Encoder(nn.Module):
    """ResNet-50 encoder for feature extraction"""

    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Modify first conv for grayscale (1-channel) input
        if in_channels == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Average RGB weights to initialize grayscale conv
                self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet stages
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # Output channels for each stage
        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns multi-scale features
        """
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # C0: 64 channels

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # C1: 256 channels

        x = self.layer2(x)
        features.append(x)  # C2: 512 channels

        x = self.layer3(x)
        features.append(x)  # C3: 1024 channels

        x = self.layer4(x)
        features.append(x)  # C4: 2048 channels

        return features
