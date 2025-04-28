# Importing Libraries
import torch
from torch.utils.data import DataLoader

from dataloader.loader.cifar10_loader import load_cifar10
from dataloader.loader.mnist_loader import load_mnist
from dataloader.loader.InterHand26M_loader import load_InterHand26M
from dataloader.loader.Oxford102Flower_loader import load_OxfordFlowers


def load_dataloader(
    name: str = "mnist",
    split: str = 'train',
    logger=None,
    config = None,
    save_path: str = None,
) -> DataLoader:
    """Load the data loader for the given name.

    Args:
        name (str, optional): The name of the data loader. Defaults to "mnist".
        batch_size (int, optional): The batch size. Defaults to 2.
        image_size (int, optional): The image size. Defaults to 256.
        num_workers (int, optional): The number of workers to use for the dataloader. Defaults to 4.
        save_path (str, optional): The path to save the data to. Defaults to "data".

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """
    model_name = config['architecture']['model_name']
    dataset_name = config['dataset']['dataset_name']
    img_size = config["dataset"]["img_size"][dataset_name]
    in_channels = config['dataset']['img_channels'][dataset_name]
    batch_size = config['dataset']["batch_size"][model_name][dataset_name]    
    num_workers = config['trainer']['num_workers']
    # print(f"Loading {name} dataset")
    if name == "mnist":
        return load_mnist(
            batch_size=batch_size,
            image_size=img_size,
            num_workers=num_workers,
            # save_path=save_path,
            split=split,
            logger=logger,
            config = config,
        )

    elif name == "cifar10":
        return load_cifar10(
            batch_size=batch_size,
            image_size=img_size,
            num_workers=num_workers,
            # save_path=save_path,
            split=split,
            logger=logger,
            config = config,
        )
    elif name == "InterHand26M":
        return load_InterHand26M(
            batch_size=batch_size,
            image_size=img_size,
            num_workers=num_workers,
            # save_path=save_path,
            split=split,
            logger=logger,
            config = config,
        )
    elif name == "Oxford102Flower":
        return load_OxfordFlowers(
            batch_size=batch_size,
            image_size=img_size,
            num_workers=num_workers,
            # save_path=save_path,
            split=split,
            logger=logger,
            config = config,
        )