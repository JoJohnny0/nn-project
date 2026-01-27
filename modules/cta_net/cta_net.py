"""
Module containing the CTA-Net architecture.

Useful classes:
    - CTA_Lightning
"""

from typing import override, Literal, Self

from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torchmetrics.classification import MulticlassAccuracy

from modules.cta_net.attention import AttentionBlock
from modules.cta_net.ct_mixed import CTBlock


class CTA_Net(nn.Module):
    """
    CTA-Net Architecture.
    """

    @override
    def __init__(self: Self, in_channels: int, hidden_channels: int, out_channels: int, heads: int, window_size: int, dropout: float = 0.) -> None:
        """
        Initializes the CTA-Net Architecture.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            heads (int): Number of attention heads.
            window_size (int): Size of the square attention window.
            dropout (float): Dropout rate.
        """

        super().__init__()

        self.pipe: nn.Sequential = nn.Sequential(
            CTBlock(in_channels, hidden_channels, heads, window_size, dropout),
            AttentionBlock(hidden_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, out_channels)
        )

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        out: torch.Tensor = self.pipe(x)
        return out


class CTA_Lightning(LightningModule):
    """
    CTA-Net Lightning Module.
    """

    @override
    def __init__(self: Self, in_channels: int, hidden_channels: int, out_channels: int, heads: int, window_size: int, lr: float, dropout: float = 0.) -> None:
        """
        Initializes the CTA-Net Lightning Module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            heads (int): Number of attention heads.
            window_size (int): Size of the square attention window.
            lr (float): Learning rate for the optimizer.
            dropout (float): Dropout rate.
        """

        super().__init__()
        self.save_hyperparameters()

        self.model: CTA_Net = CTA_Net(in_channels, hidden_channels, out_channels, heads, window_size, dropout)
        self.loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(out_channels, average = 'micro')

    @override
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
    
    @override
    def predict_step(self: Self, batch: list[torch.Tensor]) -> torch.Tensor:

        return self(batch[0]).argmax(dim = 1)
    
    def step(self: Self, batch: list[torch.Tensor], stage: Literal['train', 'val', 'test']) -> torch.Tensor:

        x: torch.Tensor
        target: torch.Tensor
        x, target = batch

        preds: torch.Tensor = self(x)

        # Compute metrics
        loss: torch.Tensor = self.loss(preds, target)
        accuracy: torch.Tensor = self.accuracy(preds, target)

        self.log(f'{stage}_loss', loss, prog_bar = True)
        self.log(f'{stage}_accuracy', accuracy, prog_bar = True)

        return loss
    
    @override
    def training_step(self: Self, batch: list[torch.Tensor]) -> torch.Tensor:

        return self.step(batch, 'train')
    
    @override
    def validation_step(self: Self, batch: list[torch.Tensor]) -> None:

        self.step(batch, 'val')
    
    @override
    def test_step(self: Self, batch: list[torch.Tensor]) -> None:

        self.step(batch, 'test')
    
    @override
    def configure_optimizers(self: Self) -> Adam:

        return Adam(self.parameters(),
                    lr = self.hparams.lr    # type: ignore
                    )
