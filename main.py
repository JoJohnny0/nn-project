from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader

from typeguard import install_import_hook   # TODO: remove
install_import_hook('modules')

from modules.cta_net.cta_net import CTA_Lightning
from modules.dataset import get_loaders


# Set seed
pl.seed_everything(42)


# Hyperparameters
patch_size: int = 15
train_samples_per_class: int = 10
val_samples_per_class: int = 5
sigma: float = 0.1
central_region_size: int = 3
hidden_channels: int = 128
heads: int = 2
dropout: float = 0.1
lr: float = 8e-5
batch_size: int = 32
epochs: int = 150


# Select dataset
dataset: Literal['Pavia University', 'Indian Pine'] = 'Pavia University'

# Load dataset
image: NDArray[np.uint16]
labels: NDArray[np.uint8]
if dataset == 'Pavia University':
    image = loadmat('data/datasets/Pavia University/PaviaU.mat')['paviaU']
    labels = loadmat('data/datasets/Pavia University/PaviaU_gt.mat')['paviaU_gt']
else:
    image  = np.load('data/datasets/Indian Pine/indianpinearray.npy')
    labels = np.load('data/datasets/Indian Pine/IPgt.npy')

train_loader: DataLoader[list[torch.Tensor]]
val_loader: DataLoader[list[torch.Tensor]]
test_loader: DataLoader[list[torch.Tensor]]
train_loader, val_loader, test_loader = get_loaders(image,
                                                    labels,
                                                    patch_size,
                                                    train_samples_per_class = train_samples_per_class,
                                                    val_samples_per_class = val_samples_per_class,
                                                    sigma = sigma,
                                                    central_region_size = central_region_size,
                                                    batch_size = batch_size
                                                    )

# Initialize logger
wandb_logger: WandbLogger = WandbLogger(project = f'Hyperspectral Classification on {dataset} dataset', save_dir = 'data/wandb')
wandb_logger.experiment.define_metric('*', step_metric = 'epoch')

# Add best checkpoint callback
save_best: ModelCheckpoint = ModelCheckpoint(monitor = 'val_avg_accuracy',
                                             dirpath = 'data/checkpoints',
                                             filename = 'cta-net-epoch={epoch}',
                                             mode = 'max',
                                             save_weights_only = True
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
trainer: pl.Trainer = pl.Trainer(max_epochs = epochs, logger = wandb_logger, callbacks = [save_best], log_every_n_steps = len(train_loader))
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader, ckpt_path = 'best')
