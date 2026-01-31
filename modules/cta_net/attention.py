"""
Module containing the Attention block for CTA-net.

Useful classes:
    - AttentionBlock
"""

from typing import override, Self

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention block.
    """

    @override
    def __init__(self: Self) -> None:
        """
        Initializes the Channel Attention block.
        """

        super().__init__()

        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias = False),
            nn.Sigmoid()
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        out: torch.Tensor = self.pool(x)
        out = out.squeeze(-1).transpose(1, 2)   # (B, C, 1, 1) -> (B, 1, C)
        out = self.pipe(out)
        out = out.transpose(1, 2).unsqueeze(-1) # (B, 1, C) -> (B, C, 1, 1)
        
        return out * x + x

class SpatialAttention(nn.Module):
    """
    Spatial Attention block.
    """

    @override
    def __init__(self: Self, channels: int) -> None:
        """
        Initializes the Spatial Attention block.

        Args:
            channels (int): Number of input and output channels.
        """

        super().__init__()

        self.stat_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size = 5, stride = 1, padding = 2),
            nn.PReLU()
        )

        # Input channels + 4 stats channels (Max, Min, Mean, Std)
        self.final_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels + 4, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        # Calculate stats for each channel
        max: torch.Tensor
        min: torch.Tensor
        max, _ = torch.max(x, dim = 1, keepdim = True)
        min, _ = torch.min(x, dim = 1, keepdim = True)
        mean: torch.Tensor = torch.mean(x, dim = 1, keepdim = True)
        std: torch.Tensor = torch.std(x, dim = 1, keepdim = True)
        stats: torch.Tensor = torch.cat((max, min, mean, std), dim = 1)
        
        # forward pass
        stats = self.stat_conv(stats)
        out: torch.Tensor = torch.cat((x, stats), dim = 1)
        out = self.final_conv(out)
        
        return x * out + x

class AttentionBlock(nn.Module):
    """
    Attention Block combining Channel and Spatial Attention.
    """

    @override
    def __init__(self: Self, channels: int) -> None:
        """
        Initializes the Attention Block.

        Args:
            channels (int): Number of input and output channels.
        """

        super().__init__()

        self.pipe: nn.Sequential = nn.Sequential(
            ChannelAttention(),
            SpatialAttention(channels)
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        out: torch.Tensor = self.pipe(x)
        return out
