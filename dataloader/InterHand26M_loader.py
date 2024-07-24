import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
from glob import glob
from pycocotools.coco import COCO
from PIL import Image

from torch.utils.data import DataLoader
import torchvision

from torchvision import transforms

import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from dataloader.dataset.InterHand26M import InterHand26M

def load_InterHand26M(
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 6,
    save_path: str = None,
    split: str = 'train',
    logger=None,
    return_annotation=False,
) -> DataLoader:
    """Load the InterHand26M data and returns the dataloaders (train ). The data is downloaded if it does not exist.

    Args:
        batch_size (int): The batch size.
        image_size (int): The image size.
        num_workers (int): The number of workers to use for the dataloader.
        save_path (str): The path to save the data to.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    assert split in ['train', 'val', 'test']
    if save_path is None:
        data_dir = '/scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1'
    else:
        data_dir = save_path
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    if split == 'train':
        data_transforms = transforms.Compose([    
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p = 0.2),
            transforms.RandomVerticalFlip(p = 0.2),
            transforms.RandomApply([transforms.RandomRotation(25)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([    
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    dataset = InterHand26M(data_dir, split, transform=data_transforms, logger=logger)
    
    if logger is not None:
        logger.info(f"Number of {split} samples: {len(dataset)}")
    else:
        print(f"Number of {split} samples: {len(dataset)}")

    if split == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    # num_workers = 6
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


    return dataloader, dataset


if __name__ == '__main__':
    import logging

    log_path = 'InterHand26M.log'
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger(f'InterHand26M-LOG')
    
    batch_size = 500

    dataloader, dataset = load_InterHand26M(batch_size = batch_size, image_size=20, num_workers = 10, split= 'train', logger=logger)
    num_batches = len(dataset)//batch_size
    for i, data in enumerate(dataloader):
        print(f'{i}/{num_batches}, {data.shape}')
