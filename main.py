from argparse import ArgumentParser, Namespace
from typing import Literal

import kagglehub
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader

from typeguard import install_import_hook # TODO: remove
install_import_hook('modules')

from modules.cta_net.cta_net import CTA_Lightning
from modules.dataset import get_loaders


def main(dataset: Literal['PaviaUniversity', 'IndianPines'], seed: int|None = None) -> None:
    """
    Main function to train and evaluate CTA-Net on a hyperspectral image dataset.
    The results are logged using Weights & Biases.

    Args:
        dataset (Literal['PaviaUniversity', 'IndianPine']): The dataset to use.
        seed (int|None): Random seed for reproducibility.
    """

    # Data Hyperparameters
    patch_size: int = 15
    train_samples_per_class: int = 10
    val_samples_per_class: int = 5
    sigma: float = 0.1
    central_region_size: int = 3

    # Model Hyperparameters
    hidden_channels: int = 128
    heads: int = 2

    # Training Hyperparameters
    dropout: float = 0.1
    lr: float = 8e-5
    batch_size: int = 32
    epochs: int = 150


    # Set random seed
    if seed is not None:
        pl.seed_everything(seed)


    # Download dataset if needed
    image: NDArray[np.uint16]
    labels: NDArray[np.uint8]
    if dataset == 'PaviaUniversity':
        dataset_path: str = kagglehub.dataset_download('syamkakarla/pavia-university-hsi')
        image = loadmat(f'{dataset_path}/PaviaU.mat')['paviaU']
        labels = loadmat(f'{dataset_path}/PaviaU_gt.mat')['paviaU_gt']
    else:
        dataset_path: str = kagglehub.dataset_download('abhijeetgo/indian-pines-hyperspectral-dataset')
        image  = np.load(f'{dataset_path}/indianpinearray.npy')
        labels = np.load(f'{dataset_path}/IPgt.npy')

    # Get data loaders
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
    wandb_logger: WandbLogger = WandbLogger(project = f"Hybrid Models for Hyperspectral Image Classification",
                                            name = f"CTA-Net_{dataset}_seed={seed}",
                                            save_dir = 'data/wandb',
                                            # Parameters not logged by the trainer
                                            config = {'train_samples_per_class': train_samples_per_class,
                                                    'val_samples_per_class': val_samples_per_class,
                                                    'central_region_size': central_region_size,
                                                    'batch_size': batch_size
                                                    }
                                            )
    wandb_logger.experiment.define_metric('*', step_metric = 'epoch')

    # Add best checkpoint callback
    save_best: ModelCheckpoint = ModelCheckpoint(monitor = 'val_avg_accuracy',
                                                dirpath = 'data/checkpoints',
                                                filename = f'cta-net-epoch={{epoch}}-seed={seed}',
                                                mode = 'max',
                                                save_weights_only = True
                                                )

    # Initialize model and trainer
    model: CTA_Lightning = CTA_Lightning(in_channels = image.shape[2],
                                        hidden_channels = hidden_channels,
                                        out_channels = int(labels.max()),
                                        heads = heads,
                                        window_size = patch_size,
                                        dropout = dropout,
                                        lr = lr
                                        )
    trainer: pl.Trainer = pl.Trainer(max_epochs = epochs, logger = wandb_logger, callbacks = save_best, log_every_n_steps = len(train_loader))

    # Train and test
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path = 'best')


if __name__ == '__main__':

    # Argument parser
    parser: ArgumentParser = ArgumentParser(description = "Train and evaluate CTA-Net on a hyperspectral image dataset.")
    parser.add_argument('dataset', type = str, choices = ('PaviaUniversity', 'IndianPines'), help = "The dataset to use.")
    parser.add_argument('--seed', type = int, help = "Random seed for reproducibility.")
    args: Namespace = parser.parse_args()

    # Run main
    main(args.dataset, seed = args.seed)
