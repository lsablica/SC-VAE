from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_mnist_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    data_root = str(Path(data_dir))

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    eval_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader
