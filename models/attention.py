"""
Cross-Modal Attention Mechanisms for FUME
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention between CO2 and CH4 streams

    Allows each stream to attend to the other stream's features
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        hidden_dim = max(in_channels // reduction, 64)

        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        """
        Args:
            x1: CO2 features (B, C, H, W)
            x2: CH4 features (B, C, H, W)

        Returns:
            (x1_attended, x2_attended): Attended features for both streams
        """
        B, C, H, W = x1.shape

        # x1 attends to x2
        query1 = self.query_conv(x1).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C')
        key2 = self.key_conv(x2).view(B, -1, H * W)  # (B, C', H*W)
        value2 = self.value_conv(x2).view(B, -1, H * W)  # (B, C, H*W)

        attention_12 = F.softmax(torch.bmm(query1, key2), dim=-1)  # (B, H*W, H*W)
        out1 = torch.bmm(value2, attention_12.permute(0, 2, 1))  # (B, C, H*W)
        out1 = out1.view(B, C, H, W)
        x1_attended = x1 + self.gamma * out1

        # x2 attends to x1
        query2 = self.query_conv(x2).view(B, -1, H * W).permute(0, 2, 1)
        key1 = self.key_conv(x1).view(B, -1, H * W)
        value1 = self.value_conv(x1).view(B, -1, H * W)

        attention_21 = F.softmax(torch.bmm(query2, key1), dim=-1)
        out2 = torch.bmm(value1, attention_21.permute(0, 2, 1))
        out2 = out2.view(B, C, H, W)
        x2_attended = x2 + self.gamma * out2

        return x1_attended, x2_attended


class SelfAttention(nn.Module):
    """Self-attention module for within-stream refinement"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H * W)
        value = self.value_conv(x).view(B, -1, H * W)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return x + self.gamma * out


class ChannelAttention(nn.Module):
    """Channel-wise attention (squeeze-and-excitation)"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Global average pooling
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)

        # Channel attention
        att = F.relu(self.fc1(gap))
        att = torch.sigmoid(self.fc2(att)).view(B, C, 1, 1)

        return x * att


class DualStreamFusion(nn.Module):
    """
    Fusion module for dual-stream features

    Combines:
    - Self-attention within each stream
    - Cross-attention between streams
    - Channel attention for refinement
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.self_att_co2 = SelfAttention(in_channels)
        self.self_att_ch4 = SelfAttention(in_channels)
        self.cross_att = CrossModalAttention(in_channels)
        self.channel_att = ChannelAttention(in_channels * 2)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        co2_feat: torch.Tensor,
        ch4_feat: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            co2_feat: (B, C, H, W) CO2 stream features
            ch4_feat: (B, C, H, W) CH4 stream features
            modality_mask: (B, 2) binary mask [use_co2, use_ch4]

        Returns:
            Fused features (B, C, H, W)
        """
        # Handle missing modalities
        use_co2 = modality_mask[:, 0].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        use_ch4 = modality_mask[:, 1].view(-1, 1, 1, 1)

        # Zero out missing modalities
        co2_feat = co2_feat * use_co2
        ch4_feat = ch4_feat * use_ch4

        # Self-attention within streams
        co2_self = self.self_att_co2(co2_feat)
        ch4_self = self.self_att_ch4(ch4_feat)

        # Cross-attention between streams (only for paired samples)
        both_present = (use_co2 * use_ch4).squeeze()
        if both_present.sum() > 0:
            co2_cross, ch4_cross = self.cross_att(co2_self, ch4_self)
        else:
            co2_cross, ch4_cross = co2_self, ch4_self

        # Concatenate streams
        fused = torch.cat([co2_cross, ch4_cross], dim=1)  # (B, 2C, H, W)

        # Channel attention
        fused = self.channel_att(fused)

        # Final fusion
        fused = self.fusion_conv(fused)

        return fused
