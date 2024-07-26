# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
from glob import glob
from pycocotools.coco import COCO
from PIL import Image
from rembg import remove
import time
from torch.utils.data import DataLoader
import torchvision

from torchvision import transforms

import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class InterHand26M(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, transform = None, logger=None, config = None):
        assert split in ['train', 'val', 'test'] # data_split: train, val, test
        self.transform = transform
        self.data_split = split
        self.logger = logger
        # self.img_path = osp.join('..', 'data', 'InterHand26M', 'images')
        # self.annot_path = osp.join('..', 'data', 'InterHand26M', 'annotations')
        self.root_dir = root_dir
        self.img_path = osp.join(root_dir, 'images')
        self.annot_path = osp.join(root_dir, 'annotations')

        self.get_hand_mask = config['dataset']['get_hand_mask'] if not config is None else False
        self.return_annotations = config['dataset']['return_annotations'] if not config is None else False
        max_train_samples = float(config['dataset']['max_train_samples'])
        max_val_samples = float(config['dataset']['max_val_samples'])
        if split == 'train':
            self.max_samples = max_train_samples
        elif split == 'val':
            self.max_samples = max_val_samples
        else:
            self.max_samples = float('inf')
        # if self.get_hand_mask:
        #     dilation_size = 10
        #     self.dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)

        # # IH26M joint set
        # self.joint_set = {
        #                 'joint_num': 42, 
        #                 'joints_name': ('R_Thumb_4', 'R_Thumb_3', 'R_Thumb_2', 'R_Thumb_1', 'R_Index_4', 'R_Index_3', 'R_Index_2', 'R_Index_1', 'R_Middle_4', 'R_Middle_3', 'R_Middle_2', 'R_Middle_1', 'R_Ring_4', 'R_Ring_3', 'R_Ring_2', 'R_Ring_1', 'R_Pinky_4', 'R_Pinky_3', 'R_Pinky_2', 'R_Pinky_1', 'R_Wrist', 'L_Thumb_4', 'L_Thumb_3', 'L_Thumb_2', 'L_Thumb_1', 'L_Index_4', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Middle_4', 'L_Middle_3', 'L_Middle_2', 'L_Middle_1', 'L_Ring_4', 'L_Ring_3', 'L_Ring_2', 'L_Ring_1', 'L_Pinky_4', 'L_Pinky_3', 'L_Pinky_2', 'L_Pinky_1', 'L_Wrist'),
        #                 'flip_pairs': [ (i,i+21) for i in range(21)]
        #                 }
        # self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        # self.joint_set['root_joint_idx'] = {'right': self.joint_set['joints_name'].index('R_Wrist'), 'left': self.joint_set['joints_name'].index('L_Wrist')}
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_data.json'))
        # with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_camera.json')) as f:
        #     cameras = json.load(f)
        # with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_joint_3d.json')) as f:
        #     joints = json.load(f)
        # with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_MANO_NeuralAnnot.json')) as f:
        #     mano_params = json.load(f)
        aid_list = list(db.anns.keys())
        
        datalist = []
        for i, aid in enumerate(aid_list):
            if i >= self.max_samples:
                break
            ann = db.anns[aid]
            image_id = ann['image_id']
            bbox = ann['bbox']
            img = db.loadImgs(image_id)[0]
            # img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])
            img_name = img['file_name'].split('/')[-1]
       
            bbox = np.array(bbox, dtype=np.float32).astype(int)
            bbox[2:] += bbox[:2] # xywh -> xyxy
            datalist.append({
                'img_path': img_path,
                'img_name': img_name,
                'hand_bbox': bbox.tolist(),
            })
        
        del db # release memory

        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # data = copy.deepcopy(self.datalist[idx])
        data = self.datalist[idx]
        img_path = data['img_path']
        img_name = data['img_name']
        # if not osp.exists(img_path):
        #     self.logger.info(f'Error: {img_path} not found')
        #     idx = random.randint(0, len(self.datalist))
        #     return self.__getitem__(idx)
        
        img = cv2.imread(img_path)
        # if not isinstance(img, np.ndarray):
        #     self.logger.info(f'Error: {img_path} is not a valid image')
        #     idx = random.randint(0, len(self.datalist))
        #     return self.__getitem__(idx)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            hand_bbox_list = data['hand_bbox']
            hand_bbox = np.array(hand_bbox_list, dtype=np.int32)
            ### crop hand based on hand_bbox
            x1, y1, x2, y2 = hand_bbox
            h, w = img.shape[:2]
            
            margin = 15
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            hand_img = img[y1:y2, x1:x2, :]
        except:
            self.logger.info(f'Error: {img_path}, hand_bbox is not valid')
            self.logger.info(f'hand_bbox: {hand_bbox_list}')
            idx = random.randint(0, len(self.datalist))
            return self.__getitem__(idx)
        
        if self.get_hand_mask:
            # start_time = time.time()
            # output = remove(hand_img) ### time-consuming, 
            # hand_mask = output[:, :, 3].copy()
            # hand_mask[hand_mask>0] = 1
            # hand_mask = cv2.dilate(hand_mask, self.dilation_kernel, iterations=1)
            # print('mask time:', hand_img.shape, time.time() - start_time) # (154, 140, 3) 17.662569761276245
            pass

        ## pad to square
        h, w = hand_img.shape[:2]
        max_size = max(h, w)
        hand_img = cv2.copyMakeBorder(hand_img, 0, max_size - h, 0, max_size - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # if self.get_hand_mask:
        #     hand_mask = cv2.copyMakeBorder(hand_mask, 0, max_size - h, 0, max_size - w, cv2.BORDER_CONSTANT, value=(0))

        if not isinstance(hand_img, np.ndarray):
            self.logger.info(f'Error: {img_path}, not a valid ndarray image')
            idx = random.randint(0, len(self.datalist))
            return self.__getitem__(idx)


            
        hand_img = Image.fromarray(hand_img)        
        if not isinstance(hand_img, Image.Image):
            self.logger.info(f'Error: {img_path}')
            idx = random.randint(0, len(self.datalist))
            return self.__getitem__(idx)
        
        h, w = hand_img.size

        if self.transform:
            hand_img = self.transform(hand_img)
            h, w = hand_img.size(1), hand_img.size(2)

        
        # if self.get_hand_mask:
        #     hand_mask = cv2.resize(hand_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        #     hand_mask = torch.from_numpy(hand_mask).unsqueeze(0).float()
        #     hand_img = torch.cat([hand_img, hand_mask], dim=0)

        # hand_data = {
        #     'img': hand_img,
        #     'img_name': img_name
        # }
        # return hand_data
        return hand_img
        


if __name__ == '__main__':

    from torchvision import transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import cv2
    import logging

    data_dir = '/scratch/rhong5/dataset/InterHand/InterHand2.6M_5fps_batch1'
    log_path = 'InterHand26M.log'
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger(f'InterHand26M-LOG')

    trans = transforms.Compose([    
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p = 0.2),
        transforms.RandomVerticalFlip(p = 0.2),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = InterHand26M(root_dir=data_dir, split= 'val', transform=trans, logger=logger)


    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, img in enumerate(tqdm(dataloader)):
        # print('img:', img.shape)
        img = img.squeeze().permute(1, 2, 0).numpy()        
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = img_gray > 20
        mask = mask.astype(np.uint8) * 255
        mask = cv2.merge([mask, mask, mask])
        img = np.concatenate([img, mask], axis=1)
        # print('img:', img.shape)
        cv2.imwrite(f'./2_{i}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if i > 20:
            break

    print('done')

