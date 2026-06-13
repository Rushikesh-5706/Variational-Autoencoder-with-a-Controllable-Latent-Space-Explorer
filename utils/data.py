import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size: int = 64, data_root: str = "data"):
    """
    Build DataLoaders for FashionMNIST.

    Downloads the dataset to `data_root` if not already present.
    Applies ToTensor() which scales pixels to [0, 1] — correct for
    BCE + Sigmoid reconstruction loss.

    Args:
        batch_size: samples per batch (default 64)
        data_root:  directory to store / load the raw dataset from

    Returns:
        (train_loader, test_loader): both are DataLoader instances
    """
    os.makedirs(data_root, exist_ok=True)
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, test_loader
