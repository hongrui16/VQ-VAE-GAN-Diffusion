import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, config = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert split in ['train', 'val']
        if split == 'val':
            split = 'valid'
        root_dir = os.path.join(root_dir, split)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        self.return_labels = config['dataset']['return_annotations']

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                ### check if the file is an image (.jpg, .jpeg, .png, .tif, .bmp), if not, skip it
                if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                    continue
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        if self.return_labels:
            return image, label
        else:
            return image
