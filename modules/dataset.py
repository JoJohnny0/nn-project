"""
Module for handling hyperspectral image datasets.

Useful functions:
    - get_loaders
"""

from math import ceil
from typing import override, Self

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset
import torchvision.transforms as T


class HyperspectralDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Class to handle hyperspectral image datasets.
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
    

def augment_dataset(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], sigma: float, central_region_size: int) -> TensorDataset:
    """
    Applies gaussian noise, random rotations and linear combinations to augment the dataset.
    The gaussian noise will not be applied in the central region of each patch.

    Args:
        dataset (Dataset[tuple[Tensor, Tensor]]): The dataset to augment.
        sigma (float): Standard deviation of the gaussian noise to add.
        central_region_size (int): Size of the central region where no noise will be added during augmentation. Must be odd.
    """

    if central_region_size % 2 == 0:
        raise ValueError("Central region size must be odd.")

    side: int = dataset[0][0].size(1)

    # Mask for the gaussian noise
    mask: torch.Tensor = torch.full_like(dataset[0][0], sigma)
    border: int = (side - central_region_size) // 2
    mask[:, border : -border, border : -border] = 0.0

    # Rotation transform
    padding: int = ceil(side * (2**0.5 - 1) / 2)
    reflect_rotation: T.Compose = T.Compose([
        T.Pad(padding, padding_mode = 'reflect'),
        T.RandomRotation(180, interpolation = T.InterpolationMode.BILINEAR),
        T.CenterCrop(side)
    ])

    augmented_samples: list[torch.Tensor] = []
    for x, _ in dataset:

        augmented_samples.append(x)

        # Gaussian noise
        noise: torch.Tensor = torch.randn_like(x) * mask
        augmented_samples.append(x + noise)

        # Rotations
        rotated_patch: torch.Tensor = reflect_rotation(x)
        augmented_samples.append(rotated_patch)
    
    # Linear combinations
    dataset_length: int = len(dataset)  # type: ignore
    couples: NDArray[np.long] = np.random.choice(dataset_length, size = (dataset_length, 2))
    for i, j in couples:
        x1: torch.Tensor = dataset[i][0]
        x2: torch.Tensor = dataset[j][0]
        alpha: float = np.random.uniform()
        new_x: torch.Tensor = alpha * x1 + (1 - alpha) * x2
        augmented_samples.append(new_x)
    
    # Build augmented dataset
    y: torch.Tensor = dataset[0][1]
    augmented_dataset: TensorDataset = TensorDataset(torch.stack(augmented_samples), y.repeat(len(augmented_samples)))
    return augmented_dataset    

def get_loaders(image: NDArray[np.integer|np.floating],
                labels: NDArray[np.integer],
                patch_size: int,
                train_samples_per_class: int,
                val_samples_per_class: int,
                sigma: float,
                central_region_size: int,
                batch_size: int
                ) -> tuple[DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]]]:
    """
    Splits and augments the dataset, returning data loaders for training, validation, and testing containing the specified number of samples per class.

    Args:
        image (NDArray[integer|floating]): Hyperspectral image data of shape (H, W, C).
        labels (NDArray[integer]): Labels of shape (H, W), where 0 indicates unlabeled pixels.
        patch_size (int): Size of the square patch to extract around each pixel. Must be odd.
        train_samples_per_class (int): Number of training samples per class.
        val_samples_per_class (int): Number of validation samples per class.
        sigma (float): Standard deviation of the gaussian noise to add for augmentation.
        central_region_size (int): Size of the central region where no noise will be added during augmentation. Must be odd.
        batch_size (int): Batch size for the data loaders.

    Returns:
        loaders (tuple[DataLoader[list[Tensor]], DataLoader[list[Tensor]], DataLoader[list[Tensor]]]): training, validation, and test data loaders.
    """

    full_dataset: HyperspectralDataset = HyperspectralDataset(image, labels, patch_size)
    num_classes: int = int(labels.max())
    train_sets: list[TensorDataset] = []
    val_sets: list[Subset[tuple[torch.Tensor, torch.Tensor]]] = []
    test_sets: list[Subset[tuple[torch.Tensor, torch.Tensor]]] = []

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

        # Error if not enough samples
        if class_indices.size(0) < val_end + 1:
            error_msg: str = f"Not enough samples for class {c + 1} to allocate {train_samples_per_class} training and {val_samples_per_class} validation and at least 1 test sample.\n"
            error_msg += f"Available samples: {class_indices.size(0)}."
            raise ValueError(error_msg) 
        
        # Augment training set
        train_set: Subset[tuple[torch.Tensor, torch.Tensor]] = Subset(full_dataset, class_indices[:train_end].tolist())
        augmented_train_set: TensorDataset = augment_dataset(train_set, sigma, central_region_size)

        # Append subsets to respective lists
        train_sets.append(augmented_train_set)
        val_sets.append(Subset(full_dataset, class_indices[train_end:val_end].tolist()))
        test_sets.append(Subset(full_dataset, class_indices[val_end:].tolist()))

    # Create data loaders
    train_loader: DataLoader[list[torch.Tensor]] = DataLoader(ConcatDataset(train_sets), batch_size = batch_size, shuffle = True)
    val_loader: DataLoader[list[torch.Tensor]] = DataLoader(ConcatDataset(val_sets), batch_size = batch_size, shuffle = False)
    test_loader: DataLoader[list[torch.Tensor]] = DataLoader(ConcatDataset(test_sets), batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader
