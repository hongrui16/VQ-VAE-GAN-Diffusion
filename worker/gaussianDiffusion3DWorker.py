import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import time
import tqdm
import os
import math
import argparse

from utils.utils import ExponentialMovingAverage


class GaussianDiffusion3DWorker(object):
    def __init__(
        self,
        model,
        device = 'cpu',
        train_dataset = None,
        experiment_dir: str = "experiments",
        save_img_dir: str = "samples",
        args = None,
        config = None,
        logger = None,
    ):
        super().__init__()    # device="cpu" if args.cpu else "cuda"
        self.device = device

        model_name = config['architecture']['model_name']
        self.model_name = model_name
        self.img_size = config['architecture'][model_name]['img_size']
        self.num_epochs = config['trainer']['num_epochs']
        self.batch_size = config['trainer'][model_name]['batch_size'] 
        self.model_ema_steps = config['trainer'][model_name]['model_ema_steps']
        self.model_ema_decay = config['trainer'][model_name]['model_ema_decay']
        lr = config['trainer'][model_name]['learning_rate']

        resume_path = config['architecture'][model_name]['resume_path']
        self.logger = logger
        self.model = model
        self.device = device
        self.args = args
        self.experiment_dir = experiment_dir
        self.save_img_dir = save_img_dir

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.save_img_dir, exist_ok=True)

        self.model.to(self.device)
        #torchvision ema setting
        #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
        adjust = 1* self.batch_size * self.model_ema_steps / self.num_epochs
        alpha = 1.0 - self.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        self.model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        #load checkpoint
        if resume_path and os.path.exists(resume_path):
            ckpt=torch.load(resume_path)
            self.model_ema.load_state_dict(ckpt["model_ema"])
            self.model.load_state_dict(ckpt["model"])

        self.global_steps=0
        
        self.batch_size = config['trainer'][model_name]['batch_size'] 
        train_model = config['architecture'][model_name]['train_model']        
        if train_model:
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

            
            self.optimizer=AdamW(model.parameters(),lr=lr)
            self.scheduler=OneCycleLR(self.optimizer,lr,total_steps=self.num_epochs*num_iters_per_epoch
                                      ,pct_start=0.25,anneal_strategy='cos')
            self.loss_fn=nn.MSELoss(reduction='mean')
            self.no_clip = config['trainer'][model_name]['no_clip']

        self.n_samples = config['architecture'][model_name]['n_samples']
        
        


    def train(self, dataloader, epochs: int = 1):

        self.model.train()

        for epoch in range(epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for index, image in enumerate(tqdm_bar):
                # noise=torch.randn_like(image).to(self.device)
                image=image.to(self.device)
                loss=self.model(image)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                if self.global_steps % self.model_ema_steps==0:
                    self.model_ema.update_parameters(self.model)
                
                
            
                if self.global_steps % self.save_step == 0:
                    loginfo =f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Loss : {loss.item():.4f}"
                    # Log the information
                    self.logger.info(loginfo)

                self.global_steps += 1
                if self.args.debug and self.global_steps > 100:
                    break

            ckpt={"model":self.model.state_dict(),
                    "model_ema":self.model_ema.state_dict()}

            torch.save(ckpt, f"{self.experiment_dir}/model.pt")

            
            self.generate_images(epoch = epoch)
            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break

    
    
    def generate_images(self, n_images: int = -1, epoch = -1, generate_bs = 4):

        n_images = self.n_samples if n_images == -1 else n_images
        self.logger.info(f"{self.model_name} Generating {n_images} images...")
    
        self.model_ema.eval()
        samples=self.model_ema.module.sampling(n_images)
        save_image(samples,f"{self.save_img_dir}/Generated_epoch_{epoch:03d}.jpg", nrow=int(math.sqrt(n_images)))


    