"""
Module containing the datasets classes for hyperspectral images.

Useful classes:
    - LazyHyperspectralDataset
    - AugmentedHyperspectralDataset
"""

from math import ceil
from typing import override, Self

import numpy as np
from numpy.typing import NDArray
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as T


class LazyHyperspectralDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Class to handle hyperspectral image datasets, loading patches on-the-fly.

    Call set_normalization(mean, std) to apply per-channel normalization.
    """

    @override
    def __init__(self: Self, image: NDArray[np.integer|np.floating], labels: NDArray[np.integer], patch_size: int) -> None:
        """
        Initialize the hyperspectral dataset.

        Args:
            image (NDArray[integer|floating]): Hyperspectral image data of shape (H, W, C).
            labels (NDArray[integer]): Labels of shape (H, W), where 0 indicates unlabeled pixels.
            patch_size (int): Size of the square patch to extract around each pixel. Must be odd.
        """

        # Ensure the target is centered
        if patch_size % 2 == 0:
            raise ValueError("Patch size must be odd.")

        super().__init__()

        self.patch_size: int = patch_size
        
        # Load data
        self.image: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        self.labels: torch.Tensor = torch.from_numpy(labels).long()

        # Find valid indices
        self.indices: torch.Tensor = torch.nonzero(self.labels)
        
        # Padding
        pad: int = patch_size // 2
        self.image = F.pad(self.image,  (pad, pad, pad, pad), mode = 'reflect')

        # Make labels start from 0
        self.labels = self.labels - 1

        # Optional normalization (per-channel)
        self.mean: torch.Tensor|None = None
        self.std: torch.Tensor|None = None

    def set_normalization(self: Self, mean: torch.Tensor|None, std: torch.Tensor|None, eps: float = 1e-6) -> None:
        """
        Set per-channel normalization stats.

        Args:
            mean (Tensor|None): Mean per channel. If None, no normalization is applied.
            std (Tensor|None): Std per channel. If None, no normalization is applied.
            eps (float): Small value to avoid division by zero in case of zero std.
        """

        self.mean = mean.view(-1, 1, 1) if mean is not None else None
        self.std = std.view(-1, 1, 1).clamp(min = eps) if std is not None else None
        
    def __len__(self: Self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        
        return self.indices.size(0)

    @override
    def __getitem__(self: Self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        x: torch.Tensor
        y: torch.Tensor
        x, y = self.indices[i]

        # Extract patch centered at (x, y) (padding already applied)
        patch: torch.Tensor = self.image[:, x : x + self.patch_size, y : y + self.patch_size]

        # Apply normalization if set
        if self.mean is not None and self.std is not None:
            patch = (patch - self.mean) / self.std
        
        # Get label
        label: torch.Tensor = self.labels[x, y]
        
        return patch, label
    

class AugmentedHyperspectralDataset(TensorDataset):
    """
    Class to handle hyperspectral image datasets.
    Implements data augmentation through random rotations, noise addition (not in the central zone), and mixup.
    """

    @override
    def __init__(self: Self, x: torch.Tensor, y: torch.Tensor, max_sigma: float, central_region_size: int) -> None:
        """
        Initialize the hyperspectral dataset.

        Args:
            x (Tensor): Input patches of shape (N, C, H, W).
            y (Tensor): Labels of shape (N,).
            max_sigma (float): Maximum standard deviation for noise addition.
            central_region_size (int): Size of the central region to avoid noise addition.
        """

        super().__init__(x, y)
        self.max_sigma: float = max_sigma
        self.central_region_size: int = central_region_size

        # Rotation transform
        side: int = x.size(2)
        padding: int = ceil(side * (2**0.5 - 1) / 2)
        self.reflect_rotation: T.Compose = T.Compose([
            T.Pad(padding, padding_mode = 'reflect'),
            T.RandomRotation(180, interpolation = T.InterpolationMode.BILINEAR),
            T.CenterCrop(side)
        ])

        # Noise mask (to avoid adding noise in the central region)
        border: int = (side - self.central_region_size) // 2
        self.noise_mask: torch.Tensor = torch.ones((side, side))
        self.noise_mask[border:-border, border:-border] = 0

        # Store for each class the indices of its samples
        self.class_indices: dict[int, np.ndarray] = {}
        for c in torch.unique(y):
            label: int = int(c.item())
            self.class_indices[label] = np.where(y.numpy() == label)[0]

    @override
    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        x: torch.Tensor
        y: torch.Tensor
        x, y = super().__getitem__(index)

        # Random rotation
        if random.random() < 0.5:
            x = self.reflect_rotation(x)

        # Noise
        if random.random() < 0.5:
            sigma: float = random.uniform(0., self.max_sigma)
            noise: torch.Tensor = torch.randn_like(x) * sigma * self.noise_mask
            x = x + noise

        # Mixup
        if random.random() < 0.5:
            class_sample_indices: NDArray[np.int64] = self.class_indices[int(y.item())]
            mix_index: int = int(np.random.choice(class_sample_indices))
            x2: torch.Tensor = super().__getitem__(mix_index)[0]
            alpha: float = random.random()
            x = alpha * x + (1 - alpha) * x2

        return x, y
