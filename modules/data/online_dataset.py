"""
Module for handling hyperspectral image datasets.
Implements online data augmentation.

Useful functions:
    - get_loaders
"""

from math import ceil
from typing import override, Self

import numpy as np
from numpy.typing import NDArray
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


class OnlineHyperspectralDataset(TensorDataset):
    """
    Class to handle hyperspectral image datasets.
    Implements data augmentation through random rotations, noise addition (not in the central zone), and mixup.
    """

    @override
    def __init__(self: Self, x: torch.Tensor, y: torch.Tensor, augment: bool = False) -> None:
        """
        Initialize the hyperspectral dataset.

        Args:
            x (Tensor): Input patches of shape (N, C, H, W).
            y (Tensor): Labels of shape (N,).
            augment (bool): Whether to apply data augmentation.
        """

        super().__init__(x, y)
        self.augment: bool = augment

        # Rotation transform
        side: int = x.size(2)
        padding: int = ceil(side * (2**0.5 - 1) / 2)
        self.reflect_rotation: T.Compose = T.Compose([
            T.Pad(padding, padding_mode = 'reflect'),
            T.RandomRotation(180, interpolation = T.InterpolationMode.BILINEAR),
            T.CenterCrop(side)
        ])

        # Noise mask (to avoid adding noise in the central region)
        border: int = (side - 3) // 2   # Central 3x3 region
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

        if self.augment:
            return x, y

        # Random rotation
        if random.random() < 0.5:
            x = self.reflect_rotation(x)

        # Noise
        if random.random() < 0.5:
            sigma: float = random.uniform(0.0, 0.1)
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


def get_loaders(image: NDArray[np.integer|np.floating],
                labels: NDArray[np.integer],
                patch_size: int,
                train_samples_per_class: int,
                val_samples_per_class: int,
                batch_size: int
                ) -> tuple[DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]]]:
    """
    Split, normalize and augment the dataset, returning data loaders for training, validation, and testing containing the specified number of samples per class.

    Args:
        image (NDArray[integer|floating]): Hyperspectral image data of shape (H, W, C).
        labels (NDArray[integer]): Labels of shape (H, W), where 0 indicates unlabeled pixels.
        patch_size (int): Size of the square patch to extract around each pixel. Must be odd.
        train_samples_per_class (int): Number of training samples per class.
        val_samples_per_class (int): Number of validation samples per class.
        batch_size (int): Batch size for the data loaders.

    Returns:
        loaders (tuple[DataLoader[list[Tensor]], DataLoader[list[Tensor]], DataLoader[list[Tensor]]]): training, validation, and test data loaders.
    """

    if patch_size % 2 == 0:
        raise ValueError("Patch size must be odd.")

    num_classes: int = int(labels.max())

    # Convert to torch
    image_tensor: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    labels_tensor: torch.Tensor = torch.from_numpy(labels).long() - 1   # Make labels start from 0

    # Range of indices for each class
    train_end: int = train_samples_per_class
    val_end: int = train_end + val_samples_per_class

    train_coords_list: list[torch.Tensor] = []
    val_coords_list: list[torch.Tensor] = []
    test_coords_list: list[torch.Tensor] = []
    for c in range(num_classes):

        class_coords: torch.Tensor = torch.nonzero(labels_tensor == c)

        # Error if not enough samples
        if class_coords.size(0) < val_end + 1:
            error_msg: str = f"Not enough samples for class {c + 1} to allocate {train_samples_per_class} training and {val_samples_per_class} validation and at least 1 test sample.\n"
            error_msg += f"Available samples: {class_coords.size(0)}."
            raise ValueError(error_msg)

        # Shuffle coordinates
        perm: torch.Tensor = torch.randperm(class_coords.size(0))
        class_coords = class_coords[perm]
        
        # Add to splits
        train_coords_list.append(class_coords[:train_end])
        val_coords_list.append(class_coords[train_end:val_end])
        test_coords_list.append(class_coords[val_end:])
    train_coords: tuple[torch.Tensor, torch.Tensor] = tuple(torch.cat(train_coords_list).T) # type: ignore
    val_coords: tuple[torch.Tensor, torch.Tensor] = tuple(torch.cat(val_coords_list).T) # type: ignore
    test_coords: tuple[torch.Tensor, torch.Tensor] = tuple(torch.cat(test_coords_list).T)   # type: ignore
    
    # Extract patches
    pad: int = patch_size // 2
    padded_image: torch.Tensor = F.pad(image_tensor, (pad, pad, pad, pad), mode = 'reflect')
    patches: torch.Tensor = padded_image.unfold(1, patch_size, 1).unfold(2, patch_size, 1).permute(1, 2, 0, 3, 4)   # Shape: (H, W, C, patch_size, patch_size)
    train_patches: torch.Tensor = patches[train_coords]
    val_patches: torch.Tensor = patches[val_coords]
    test_patches: torch.Tensor = patches[test_coords]

    # Normalize
    mean: torch.Tensor = train_patches.mean(dim = (0, 2, 3)).view(-1, 1, 1)
    std: torch.Tensor = train_patches.std(dim = (0, 2, 3)).view(-1, 1, 1)
    train_patches = (train_patches - mean) / std
    val_patches = (val_patches - mean) / std
    test_patches = (test_patches - mean) / std

    # Create datasets
    train_dataset: OnlineHyperspectralDataset = OnlineHyperspectralDataset(train_patches, labels_tensor[train_coords])
    val_dataset: OnlineHyperspectralDataset = OnlineHyperspectralDataset(val_patches, labels_tensor[val_coords])
    test_dataset: OnlineHyperspectralDataset = OnlineHyperspectralDataset(test_patches, labels_tensor[test_coords])

    # Create data loaders
    train_loader: DataLoader[list[torch.Tensor]] = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)   # type: ignore
    val_loader: DataLoader[list[torch.Tensor]] = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)  # type: ignore
    test_loader: DataLoader[list[torch.Tensor]] = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)    # type: ignore

    return train_loader, val_loader, test_loader
