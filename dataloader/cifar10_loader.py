# Importing Libraries
import torch
import torchvision
from torch.utils.data import DataLoader

from utils.utils import collate_fn

def load_cifar10(
    batch_size: int = 16,
    image_size: int = 28,
    num_workers: int = 4,
    save_path: str = "data",
    split: str = "train",
    logger=None,
) -> DataLoader:
    """Load the Cifar 10 data and returns the dataloaders (train ). The data is downloaded if it does not exist.

    Args:
        batch_size (int): The batch size.
        image_size (int): The image size.
        num_workers (int): The number of workers to use for the dataloader.
        save_path (str): The path to save the data to.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    # Load the data
    if split == 'train':
        dataset = torchvision.datasets.CIFAR10(
                root=save_path,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((image_size, image_size)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    else:
        dataset = torchvision.datasets.CIFAR10(
                root=save_path,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((image_size, image_size)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        
    if logger is not None:
        logger.info(f"Number of {split} samples: {len(dataset)}")
    else:
        print(f"Number of {split} samples: {len(dataset)}")


    return dataloader, dataset
