"""
Module for handling hyperspectral image datasets.

Useful functions:
    - get_balanced_loaders
"""

from typing import override, Self

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class HyperspectralDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Class to handle hyperspectral image datasets.
    """

    @override
    def __init__(self: Self, image: NDArray[np.integer|np.floating], labels: NDArray[np.integer], patch_size: int) -> None:
        """
        Initialize the hyperspectral dataset.

        Args:
            image (NDArray[np.integer|np.floating]): Hyperspectral image data of shape (H, W, C).
            labels (NDArray[np.integer]): Labels of shape (H, W), where 0 indicates unlabeled pixels.
            patch_size (int): Size of the square patch to extract around each pixel. Must be odd.
        """

        # Ensure the target is centered
        if patch_size % 2 == 0:
            raise ValueError("Patch size must be odd.")

        super().__init__()

        self.patch_size: int = patch_size
        
        # Load data
        self.image: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        self.labels: torch.Tensor = torch.from_numpy(labels)

        # Find valid indices
        self.indices: torch.Tensor = torch.nonzero(self.labels)
        
        # Normalize
        mean: torch.Tensor = self.image.mean(dim = (1, 2), keepdim = True)
        std: torch.Tensor = self.image.std(dim = (1, 2), keepdim = True)
        self.image = (self.image - mean) / std
        
        # Padding
        pad: int = patch_size // 2
        self.image = F.pad(self.image,  (pad, pad, pad, pad), mode = 'reflect')

        # Make labels start from 0
        self.labels = self.labels - 1
        
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
        
        # Get label
        label: torch.Tensor = self.labels[x, y]
        
        return patch, label


def get_balanced_loaders(image: NDArray[np.integer|np.floating],
                          labels: NDArray[np.integer],
                          patch_size: int,
                          train_samples_per_class: int,
                          val_samples_per_class: int,
                          batch_size: int
                          ) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Splits the dataset into training, validation, and test sets with the same number of samples per class in the training and validation sets.

    Args:
        image (NDArray[np.integer|np.floating]): Hyperspectral image data of shape (H, W, C).
        labels (NDArray[np.integer]): Labels of shape (H, W), where 0 indicates unlabeled pixels.
        patch_size (int): Size of the square patch to extract around each pixel. Must be odd.
        train_samples_per_class (int): Number of training samples per class.
        val_samples_per_class (int): Number of validation samples per class.
        batch_size (int): Batch size for the data loaders.

    Returns:
        tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]: Training, validation, and test datasets.
    """

    full_dataset: HyperspectralDataset = HyperspectralDataset(image, labels, patch_size)
    num_classes: int = int(labels.max())
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    # Flatten labels for easier indexing
    all_labels: torch.Tensor = full_dataset.labels[full_dataset.indices[:, 0], full_dataset.indices[:, 1]]

    for c in range(num_classes):
        class_indices: torch.Tensor = torch.nonzero(all_labels == c).squeeze()

        # Shuffle indices
        perm: torch.Tensor = torch.randperm(class_indices.size(0))
        class_indices = class_indices[perm]

        # Select samples
        train_end: int = train_samples_per_class
        val_end: int = train_end + val_samples_per_class

        if class_indices.size(0) < val_end + 1:
            error_msg: str = f"Not enough samples for class {c + 1} to allocate {train_samples_per_class} training and {val_samples_per_class} validation and at least 1 test sample.\n"
            error_msg += f"Available samples: {class_indices.size(0)}."
            raise ValueError(error_msg) 

        train_indices.extend(class_indices[:train_end].tolist())
        val_indices.extend(class_indices[train_end:val_end].tolist())
        test_indices.extend(class_indices[val_end:].tolist())

    # Create data loaders
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(Subset(full_dataset, train_indices), batch_size = batch_size, shuffle = True)
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(Subset(full_dataset, val_indices), batch_size = batch_size, shuffle = False)
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(Subset(full_dataset, test_indices), batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader
