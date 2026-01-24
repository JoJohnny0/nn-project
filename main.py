from lightning.pytorch import Trainer
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader

from typeguard import install_import_hook
install_import_hook('modules')

from modules.cta_net.cta_net import CTA_Lightning
from modules.dataset import get_balanced_loaders


# Hyperparameters
train_samples_per_class: int = 10
val_samples_per_class: int = 5
patch_size: int = 15
hidden_channels: int = 128
heads: int = 2
dropout: float = 0.1
lr: float = 8e-5
batch_size: int = 32
epochs: int = 150


# Load data
image = loadmat('data/pavia_university/PaviaU.mat')['paviaU']
labels = loadmat('data/pavia_university/PaviaU_gt.mat')['paviaU_gt']
#image  = np.load('data/indian_pine/indianpinearray.npy')
#labels = np.load('data/indian_pine/IPgt.npy')

train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
train_loader, val_loader, test_loader = get_balanced_loaders(image,
                                                             labels,
                                                             patch_size,
                                                             train_samples_per_class = train_samples_per_class,
                                                             val_samples_per_class = val_samples_per_class,
                                                             batch_size = batch_size
                                                             )

# Train model
model: CTA_Lightning = CTA_Lightning(in_channels = image.shape[2],
                                     hidden_channels = hidden_channels,
                                     out_channels = int(labels.max()),
                                     heads = heads,
                                     window_size = patch_size,
                                     dropout = dropout,
                                     lr = lr
                                     )
trainer: Trainer = Trainer(max_epochs = epochs)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
