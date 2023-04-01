import multiprocessing as mp
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Run IPython magic commands
from IPython.core.getipython import get_ipython
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

ipython = get_ipython()
if ipython is not None:
    # Only works in interactive mode
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


def mnist(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create loaders for MNIST dataset"""

    # Load and transform to tensor
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Loaders
    # The loading can be a large bottleneck for the training speed. Increase number of workers and pin memory.
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=mp.cpu_count(),
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=True,
    )

    return train_loader, test_loader
