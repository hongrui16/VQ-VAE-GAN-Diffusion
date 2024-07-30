import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import torchvision

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset
import time
from einops import rearrange, reduce

from accelerate import Accelerator
from ema_pytorch import EMA

import tqdm
import os, sys
import numpy as np
import cv2
import torch



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class GaussianDiffusionWorker(object):
    def __init__(
        self,
        diffusion_model,
        device = 'cpu',
        train_dataset: Dataset = None,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        adam_betas = (0.9, 0.99),
        experiment_dir: str = "experiments",
        save_img_dir: str = "samples",
        amp = True,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1., 
        args = None,
        config = None,
        logger = None,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        if 'img_size' in config['architecture']['gaussiandiffusion']:
            self.img_size = config['architecture']['gaussiandiffusion']['img_size']  
        else:
            self.img_size = 256
        
        if 'train_gaussiandiffusion' in config['architecture']['gaussiandiffusion']:
            train_gaussiandiffusion = config['architecture']['gaussiandiffusion']['train_gaussiandiffusion']
        else:
            train_gaussiandiffusion = True

        self.logger = logger
        self.model = diffusion_model
        self.device = device
        self.args = args

        self.input_dim = self.model.input_dim
        num_samples = 9
            # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples

        self.batch_size = config['dataset']['batch_size'] if 'batch_size' in config['dataset'] else 1
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        
        if train_gaussiandiffusion:
            num_iters_per_epoch = len(train_dataset)//self.batch_size
            self.save_step = 100
            if num_iters_per_epoch < 0.1*self.save_step:
                self.save_step = 1
            elif num_iters_per_epoch < 0.5*self.save_step:
                self.save_step = 5
            elif num_iters_per_epoch < 1.5*self.save_step:
                self.save_step = 10
            elif num_iters_per_epoch < 10*self.save_step:
                self.save_step = 50
            elif num_iters_per_epoch < 50*self.save_step:
                self.save_step = 100
            else:
                self.save_step = 200

            self.logger.info(f"Save step set to {self.save_step}")          

            # optimizer

            self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.results_folder = experiment_dir
        os.makedirs(self.results_folder, exist_ok = True)

        self.save_img_folder = save_img_dir
        os.makedirs(self.save_img_folder, exist_ok = True)


        self.step = 0

        ema_decay = 0.9999
        ema_update_every = 10
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)




    def save(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
        }

        if self.accelerator.is_main_process:
            data['ema'] = self.ema.state_dict()


        torch.save(data, os.path.join(self.results_folder, f'model.pt'))

    def load(self, resume_path):
        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['opt'])
        if 'ema' in checkpoint and self.accelerator.is_main_process:
            self.ema.load_state_dict(checkpoint['ema'])


    def train(self,
        dataloader,
        epochs: int = 1,):

        self.model.train()

        for epoch in range(epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for index, imgs in enumerate(tqdm_bar):
                self.opt.zero_grad()
                if imgs.dim() > self.input_dim:
                    imgs = imgs.squeeze(1)
                imgs = imgs.to(device=self.device)
                
                out = self.model(imgs)
                loss = out['loss']
                
                self.accelerator.backward(loss)
                
                if (index + 1) % self.gradient_accumulate_every == 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    if self.accelerator.is_main_process and self.ema:
                        self.ema.update()

            
                if self.step % self.save_step == 0:
                    loginfo =f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Loss : {loss.item():.4f}"
                    # Log the information
                    self.logger.info(loginfo)



                self.step += 1
                if self.args.debug and self.step > 100:
                    break

            self.generate_images(epoch = epoch)
            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break


    
    def generate_images(self, n_images: int = 1, epoch = -1, generate_bs = 4):

        self.logger.info(f"gaussianDiffusion Generating {n_images} images...")
        # Use EMA model for generation if available
        if hasattr(self, 'ema'):
            model_to_use = self.ema.ema_model
        else:
            model_to_use = self.model

        model_to_use.eval()
        model_to_use.to(self.device)

        with torch.no_grad():
            for i in range(n_images):
                random_imgs = torch.rand(generate_bs, self.img_size, self.img_size).to(self.device)
                sampled_imgs = model_to_use.sample(generate_bs, random_imgs)
                sampled_imgs = sampled_imgs.detach()#.cpu().permute(1, 2, 0).numpy()
                sampled_imgs = (sampled_imgs - sampled_imgs.min()) / (sampled_imgs.max() - sampled_imgs.min())
                if self.input_dim == 3:
                    sampled_imgs = sampled_imgs.unsqueeze(1)

                torchvision.utils.save_image(
                    sampled_imgs,
                    os.path.join(self.save_img_folder, f"Generating_epoch{epoch:03d}_{i}.jpg"),
                    nrow=4,
                )
    