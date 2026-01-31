"""
Module containing the CNN-Transformer (CT) mixed block.

Useful classes:
    - CTBlock
"""

from typing import override, Self

import torch
import torch.nn as nn


### CNN branch ###

class MultiscaleCNN(nn.Module):
    """
    Multi-Scale CNN block.
    """

    @override
    def __init__(self: Self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the Multi-Scale CNN block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        
        # Branch 1: 1x1 Conv
        self.branch1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        
        # Branch 2: 3x3 Conv
        self.branch2: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        
        # Branch 3: 2 stacked 3x3 Convs
        self.branch3: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        )
        
        # Branch 4: 3 stacked 3x3 Convs
        self.branch4: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        
        # Fusion Conv 1x1
        self.fusion: nn.Conv2d = nn.Conv2d(out_channels * 4, out_channels, kernel_size = 1, stride = 1, padding = 0)

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        # Multi-Scale Feature Extraction
        b1: torch.Tensor = self.branch1(x)
        b2: torch.Tensor = self.branch2(x)
        b3: torch.Tensor = self.branch3(x)
        b4: torch.Tensor = self.branch4(x)
        
        # Feature Fusion
        concat: torch.Tensor = torch.cat((b1, b2, b3, b4), dim = 1)
        out: torch.Tensor = self.fusion(concat)
        
        return out


### Transformer branch ###

class FFM(nn.Module):
    """
    Feed Forward Module.
    """

    @override
    def __init__(self: Self, channels: int, dropout: float) -> None:
        """
        Initializes the Feed Forward Module.

        Args:
            channels (int): Number of input and output channels.
            dropout (float): Dropout rate.
        """

        super().__init__()

        self.pipe: nn.Sequential = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
            nn.Dropout(dropout)
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        out: torch.Tensor = self.pipe(x)
        return out + x


class RelativePositionBias(nn.Module):
    """
    Bias for relative positional encoding in attention.
    """

    @override
    def __init__(self: Self, heads: int, window_size: int) -> None:
        """
        Initializes the Relative Position Bias module.

        Args:
            heads (int): Number of attention heads.
            window_size (int): Size of the square attention window.
        """

        super().__init__()
        self.window_size: int = window_size

        # Create a parameter table of relative position biases
        self.bias_table: nn.Parameter = nn.Parameter(
            torch.zeros(heads, (2 * window_size - 1) * (2 * window_size - 1))
        )

        # Generate position matrix
        coords_h_w: torch.Tensor = torch.arange(window_size)
        coords: torch.Tensor = torch.stack(torch.meshgrid([coords_h_w, coords_h_w], indexing = 'ij'))   # 2, H, W

        # Pairwise relative position index
        coords = coords.flatten(1)    # 2, H*W
        relative_coords: torch.Tensor = coords[:, :, None] - coords[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0)  # H*W, H*W, 2
        relative_coords += window_size - 1    # Shift to start from 0
        relative_coords[:, :, 0] *= (2 * window_size - 1)   # Convert from 2D to 1D index
        relative_position_index: torch.Tensor = relative_coords.sum(-1)  # H*W, H*W
        self.register_buffer('relative_position_index', relative_position_index)    # Save as non-parameter buffer

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        # Fetch relative position biases
        relative_position_bias: torch.Tensor = self.bias_table[:, self.relative_position_index] # type: ignore
        relative_position_bias = relative_position_bias.repeat(x.size(0), 1, 1)  # B*heads, H*W, H*W

        return relative_position_bias

class MHSA(nn.Module):
    """
    Multi-Head Self Attention block.
    """

    @override
    def __init__(self: Self, channels: int, heads: int, window_size: int, dropout: float) -> None:
        """
        Initializes the Multi-Head Self Attention block.

        Args:
            channels (int): Number of input and output channels.
            heads (int): Number of attention heads.
            window_size (int): Size of the square attention window.
            dropout (float): Dropout rate for attention weights and output.
        """

        super().__init__()

        self.norm: nn.LayerNorm = nn.LayerNorm(channels)
        self.rp_bias: RelativePositionBias = RelativePositionBias(heads, window_size)
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim = channels, num_heads = heads, batch_first = True)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        out: torch.Tensor = self.norm(x)
        bias: torch.Tensor = self.rp_bias(x)
        out, _ = self.attn(out, out, out, attn_mask = bias)
        out = self.dropout(out)

        return out + x


class ConformerCNN(nn.Module):
    """
    CNN block within Conformer
    """

    @override
    def __init__(self: Self, channels: int, dropout: float) -> None:
        """
        Initializes the Conformer CNN block.

        Args:
            channels (int): Number of input and output channels.
            dropout (float): Dropout rate.
        """

        super().__init__()

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(channels)
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size = 1),
            nn.GLU(dim = 1),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size = 1),
            nn.Dropout(dropout)
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        out: torch.Tensor = self.layer_norm(x)

        B: int
        L: int
        C: int
        B, L, C = out.size()
        side: int = int(L ** 0.5)
        out = out.transpose(1, 2).reshape(B, C, side, side) # (B, L, C) -> (B, C, H, W)
        out = self.pipe(out)
        out = out.flatten(2).transpose(1, 2)    # (B, C, H, W) -> (B, L, C)

        return out + x


class TransformerBlock(nn.Module):
    """
    Conformer-based Transformer block.
    """

    @override
    def __init__(self: Self, channels: int, heads: int, window_size: int, dropout: float) -> None:
        """
        Initializes the Conformer-based Transformer block.

        Args:
            channels (int): Number of input and output channels.
            heads (int): Number of attention heads.
            window_size (int): Size of the square attention window.
            dropout (float): Dropout rate.
        """
        
        super().__init__()
        
        self.pipe: nn.Sequential = nn.Sequential(
            FFM(channels, dropout),
            MHSA(channels, heads, window_size, dropout = dropout),
            ConformerCNN(channels, dropout = dropout),
            FFM(channels, dropout),
            nn.LayerNorm(channels)
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        out: torch.Tensor = x.flatten(2).transpose(1, 2)    # (B, C, H, W) -> (B, L, C)
        out = self.pipe(out)
        out = out.transpose(1, 2).reshape_as(x) # (B, L, C) -> (B, C, H, W)
        return out


### CNN-Transformer Mixed block ###

class CTBlock(nn.Module):
    """
    CNN-Transformer Mixed block.
    """

    @override
    def __init__(self: Self, in_channels: int, out_channels: int, heads: int, window_size: int, dropout: float) -> None:
        """
        Initializes the CNN-Transformer Mixed block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            heads (int): Number of attention heads.
            window_size (int): Size of the square attention window.
            dropout (float): Dropout rate.
        """

        super().__init__()

        self.start_conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.mscnn: MultiscaleCNN = MultiscaleCNN(out_channels, out_channels)
        self.transformer: TransformerBlock = TransformerBlock(out_channels, heads, window_size, dropout)
        self.end_conv: nn.Conv2d = nn.Conv2d(out_channels * 2, out_channels, kernel_size = 1, stride = 1, padding = 0)
    
    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        x = self.start_conv(x)

        cnn_out: torch.Tensor = self.mscnn(x)
        transformer_out: torch.Tensor = self.transformer(x)

        out: torch.Tensor = torch.cat((cnn_out, transformer_out), dim = 1)
        out = self.end_conv(out)

        return out + x
