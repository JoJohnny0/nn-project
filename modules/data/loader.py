"""
Module containing data loading and dataset expansion functions.

Useful functions:
    - get_loaders
"""

from math import ceil

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from torchvision import transforms as T

from modules.data.dataset import AugmentedHyperspectralDataset, LazyHyperspectralDataset


def amplify_dataset(x: torch.Tensor, y: torch.Tensor, sigma: float, central_region_size: int) -> TensorDataset:
    """
    Applies gaussian noise, random rotations and linear combinations to amplify the given dataset.
    The gaussian noise will not be applied in the central region of each patch.

    Args:
        x (Tensor): Input patches of shape (N, C, H, W).
        y (Tensor): Labels of shape (N,).
        sigma (float): Standard deviation of the gaussian noise to add.
        central_region_size (int): Size of the central region where no noise will be added during augmentation. Must be odd, less than the patch size.
    """

    side: int = x.size(2)

    if central_region_size % 2 == 0:
        raise ValueError("Central region size must be odd.")
    if central_region_size < 1 or central_region_size >= side:
        raise ValueError(f"Central region size must be in [1, {side}), got {central_region_size}.")

    # Mask for the gaussian noise
    mask: torch.Tensor = torch.full_like(x[0], sigma)
    border: int = (side - central_region_size) // 2
    mask[:, border : -border, border : -border] = 0.0

    # Rotation transform
    padding: int = ceil(side * (2**0.5 - 1) / 2)
    reflect_rotation: T.Compose = T.Compose([
        T.Pad(padding, padding_mode = 'reflect'),
        T.RandomRotation(180, interpolation = T.InterpolationMode.BILINEAR),
        T.CenterCrop(side)
    ])

    new_samples: list[torch.Tensor] = []
    new_labels: list[torch.Tensor] = []
    for patch, label in zip(x, y):

        # Original sample
        new_samples.append(patch)
        new_labels.append(label)

        # Gaussian noise
        noise: torch.Tensor = torch.randn_like(patch) * mask
        new_samples.append(patch + noise)
        new_labels.append(label)

        # Rotations
        rotated_patch: torch.Tensor = reflect_rotation(patch)
        new_samples.append(rotated_patch)
        new_labels.append(label)

    # Split the dataset based on the labels
    for label in torch.unique(y):
        
        # Shuffle samples of the same class
        class_indices: torch.Tensor = torch.nonzero(y == label).squeeze()
        class_indices = class_indices[torch.randperm(class_indices.size(0))]

        # Linear Summations (here we use average to keep variance stable)
        for i in range(-1, class_indices.size(0) -1):  # pair each sample with the next one in the shuffled order to avoid duplicates
            x1: torch.Tensor = x[class_indices[i]]
            x2: torch.Tensor = x[class_indices[i + 1]]
            mixed_patch: torch.Tensor = (x1 + x2) / 2.0
            new_samples.append(mixed_patch)
            new_labels.append(label)

    # Create new dataset
    new_x: torch.Tensor = torch.stack(new_samples)
    new_y: torch.Tensor = torch.stack(new_labels)
    return TensorDataset(new_x, new_y)


def get_loaders(image: NDArray[np.integer|np.floating],
                labels: NDArray[np.integer],
                patch_size: int,
                train_samples_per_class: int,
                val_samples_per_class: int,
                online_augmentation: bool,
                sigma: float,
                central_region_size: int,
                batch_size: int
                ) -> tuple[DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]]]:
    """
    Split, normalize and expand the dataset, returning data loaders for training, validation, and testing containing the specified number of samples per class.

    Args:
        image (NDArray[integer|floating]): Hyperspectral image data of shape (H, W, C).
        labels (NDArray[integer]): Labels of shape (H, W), where 0 indicates unlabeled pixels.
        patch_size (int): Size of the square patch to extract around each pixel. Must be odd.
        train_samples_per_class (int): Number of training samples per class.
        val_samples_per_class (int): Number of validation samples per class.
        online_augmentation (bool): Whether to apply data augmentation online during training or to amplify the dataset beforehand.
        sigma (float): Standard deviation of the gaussian noise to add. If online_augmentation is True this is the maximum sigma for random noise addition.
        central_region_size (int): Size of the central region to avoid noise addition.
        batch_size (int): Batch size for the data loaders.

    Returns:
        loaders (tuple[DataLoader[list[Tensor]], DataLoader[list[Tensor]], DataLoader[list[Tensor]]]): training, validation, and test data loaders.
    """

    if patch_size % 2 == 0:
        raise ValueError("Patch size must be odd.")

    full_dataset: LazyHyperspectralDataset = LazyHyperspectralDataset(image, labels, patch_size)
    num_classes: int = int(labels.max())

    # flattened labels for indexing
    flat_labels: torch.Tensor = full_dataset.labels[full_dataset.indices[:, 0], full_dataset.indices[:, 1]]

    # Range of indices for each class
    train_end: int = train_samples_per_class
    val_end: int = train_end + val_samples_per_class

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for c in range(num_classes):

        class_indices: torch.Tensor = torch.nonzero(flat_labels == c).squeeze()

        # Error if not enough samples
        if class_indices.size(0) < train_samples_per_class + val_samples_per_class + 1:
            error_msg: str = f"Not enough samples for class {c + 1} to allocate {train_samples_per_class} training and {val_samples_per_class} validation samples.\n"
            error_msg += f"Available samples: {class_indices.size(0)}."
            raise ValueError(error_msg)
        
        # Shuffle indices
        perm: torch.Tensor = torch.randperm(class_indices.size(0))
        class_indices = class_indices[perm]

        # Add to splits
        train_indices.extend(class_indices[:train_end].tolist())
        val_indices.extend(class_indices[train_end:val_end].tolist())
        test_indices.extend(class_indices[val_end:].tolist())
    
    # Extract train data
    train_tensor: torch.Tensor = torch.stack([full_dataset[i][0] for i in train_indices])
    train_labels: torch.Tensor = torch.tensor([full_dataset[i][1] for i in train_indices])

    # Normalize
    mean: torch.Tensor = train_tensor.mean(dim = (0, 2, 3)).view(-1, 1, 1)
    std: torch.Tensor = train_tensor.std(dim = (0, 2, 3)).view(-1, 1, 1)
    train_tensor = (train_tensor - mean) / std
    full_dataset.set_normalization(mean, std)

    # Create train dataset with augmentation
    train_dataset: TensorDataset
    if online_augmentation:
        train_dataset = AugmentedHyperspectralDataset(train_tensor, train_labels, sigma, central_region_size)
    else:
        train_dataset = amplify_dataset(train_tensor, train_labels, sigma, central_region_size)

    # Lazy validation and test datasets
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = Subset(full_dataset, val_indices)
    test_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader: DataLoader[list[torch.Tensor]] = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)   # type: ignore
    val_loader: DataLoader[list[torch.Tensor]] = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)  # type: ignore
    test_loader: DataLoader[list[torch.Tensor]] = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)    # type: ignore

    return train_loader, val_loader, test_loader
