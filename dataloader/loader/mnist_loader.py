# Importing Libraries
import torch
import torchvision
from torch.utils.data import DataLoader

from utils.utils import collate_fn


def load_mnist(
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 4,
    save_path: str = "data",
    split: str = 'train',
    logger=None,
) -> DataLoader:
    """Load the MNIST data and returns the dataloaders (train ). The data is downloaded if it does not exist.

    Args:
        batch_size (int): The batch size.
        image_size (int): The image size.
        num_workers (int): The number of workers to use for the dataloader.
        save_path (str): The path to save the data to.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """
    if split not in ['train', 'val']:
        raise ValueError(f"Split must be either 'train' or 'val'. Got {split}.")
    if split == 'train':
        train = True
    else:
        train = False
    # Load the data
    mnist_data = torchvision.datasets.MNIST(
        root=save_path,
        train=train,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    if split ==  'train':
        # Reduced set for faster training
        mnist_data_reduced = torch.utils.data.Subset(mnist_data, list(range(0, 40000)))
        shuffle = True
        drop_last = True
    else:
        mnist_data_reduced = torch.utils.data.Subset(mnist_data, list(range(0, 5000)))
        shuffle = False
        drop_last = False

    
    if logger is not None:
        logger.info(f"Number of {split} samples: {len(mnist_data_reduced)}")
    else:
        print(f"Number of {split} samples: {len(mnist_data_reduced)}")


    dataloader = DataLoader(
        mnist_data_reduced,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader, mnist_data_reduced
