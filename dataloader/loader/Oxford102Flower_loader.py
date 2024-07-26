import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision

import sys

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.dataset.Oxford102Flower import OxfordFlowersDataset

def load_OxfordFlowers(
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 4,
    save_path: str = None,
    split: str = 'train',
    logger=None,
    config = None,
) -> DataLoader:
    """Load the Oxford102Flower data and returns the dataloaders (train ). The data is downloaded if it does not exist.

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
        data_dir = '/scratch/rhong5/dataset/Oxford102Flower'
    else:
        data_dir = save_path
        
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    mean = config['dataset']['mean']
    std = config['dataset']['std']


    if split == 'train':
        data_transforms = transforms.Compose([    
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p = 0.2),
            transforms.RandomVerticalFlip(p = 0.2),
            transforms.RandomApply([transforms.RandomRotation(25)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        data_transforms = transforms.Compose([    
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


    dataset = OxfordFlowersDataset(data_dir, split, transform=data_transforms, config=config)
    
    if logger is not None:
        logger.info(f"Number of {split} samples: {len(dataset)}, Number of classes: {len(dataset.classes)}")
        # logger.info(f"{split} Classes: {dataset.classes}")
        # logger.info(f"{split} Class to index mapping: {dataset.class_to_idx}")
    else:
        print(f"Number of {split} samples: {len(dataset)}, Number of classes: {len(dataset.classes)}")
        print(f"{split} Classes: {dataset.classes}")
        print(f"{split} Class to index mapping: {dataset.class_to_idx}")


    if split == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    num_workers = 6
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


    return dataloader, dataset


# 检查DataLoader是否工作正常


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    # save
    plt.savefig('Oxford102Flower.jpg')

if __name__ == '__main__':

    train_dataloader, train_dataset = load_OxfordFlowers(batch_size=4, image_size=256, num_workers=4, 
                                        split='train',
                                        return_annotation=True
                                        )

    # 获取一个批次的训练数据
    inputs, classes = next(iter(train_dataloader))

    # 创建网格
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[train_dataset.classes[x] for x in classes])