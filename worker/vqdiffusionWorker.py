# Importing Libraries
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Run, Image
import os
import time
import tqdm
from torch.optim.lr_scheduler import OneCycleLR

from utils.utils import print_gpu_memory_usage
from utils.utils import ExponentialMovingAverage

class VQDiffusionWorker:
    def __init__(
        self,
        model: nn.Module,
        run: Run = None,
        experiment_dir: str = "experiments",
        device: str = "cpu",
        logger = None,
        train_dataset = None,
        save_img_dir = None,
        args = None,
        val_dataloader = None,
        config = None,
    ):
        model_name = config['architecture']['model_name']

        
        self.num_codebook_vectors = config['architecture']['vqvae']['num_codebook_vectors']
        self.seq_len = config['architecture']['vqvae']['latent_channels']

        train_model = config['architecture'][model_name]['train_model']
        learning_rate = config['trainer'][model_name]['learning_rate']
        
        self.img_size = config['architecture'][model_name]['img_size']
        self.num_epochs = config['trainer']['num_epochs']
        self.batch_size = config['trainer'][model_name]['batch_size'] 
        self.model_ema_steps = config['trainer'][model_name]['model_ema_steps']
        self.model_ema_decay = config['trainer'][model_name]['model_ema_decay']
        self.batch_size = config['trainer'][model_name]['batch_size'] 
        resume_path = config['architecture'][model_name]['resume_path']
        diffusion_resume_path = config['architecture'][model_name]['resume_path']
        self.return_all_timestamps = config['architecture'][model_name]['return_all_timestamps']

        beta1 = config['trainer'][model_name]['beta1']
        beta2 = config['trainer'][model_name]['beta2']

        self.vqdiffusion = model
        self.run = run
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.model_name = model_name
        self.save_img_dir = save_img_dir
        self.args = args
        self.val_dataloader = val_dataloader
        self.global_step = 0
        self.epoch = 0

        self.vqdiffusion.to(device)
        self.device = device

        adjust = 1* self.batch_size * self.model_ema_steps / self.num_epochs
        alpha = 1.0 - self.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        self.model_ema = ExponentialMovingAverage(self.vqdiffusion.diffusion, device=device, decay=1.0 - alpha)

        if not diffusion_resume_path is None and os.path.exists(diffusion_resume_path):
            ckpt=torch.load(diffusion_resume_path)
            if 'diffusion' in ckpt:
                self.vqdiffusion.diffusion.load_state_dict(ckpt['diffusion'])
            else:
                self.vqdiffusion.diffusion.load_state_dict(ckpt)
            if 'optimizer' in ckpt:
                self.optim.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            if 'global_step' in ckpt:
                self.global_step = ckpt['global_step']
            if 'epoch' in ckpt:
                self.epoch = ckpt['epoch']
            
            self.logger.info(f"Diffusion model loaded from {diffusion_resume_path}")

        if train_model:
            self.batch_size = config['trainer'][model_name]['batch_size'] 
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

            self.optim = torch.optim.AdamW(self.vqdiffusion.diffusion.parameters(), lr=learning_rate, betas=(beta1, beta2))
            self.scheduler=OneCycleLR(self.optim, learning_rate, total_steps=self.num_epochs*num_iters_per_epoch
                                      ,pct_start=0.25,anneal_strategy='cos')
            self.no_clip = config['trainer'][model_name]['no_clip']

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int):
        self.vqdiffusion.train()
        for epoch in range(self.epoch, epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for index, imgs in enumerate(tqdm_bar):
                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                # print('imgs.shape', imgs.shape)
                loss = self.vqdiffusion(imgs)
                
                loss.backward()
                self.optim.step()
                self.scheduler.step()
                if self.global_step % self.model_ema_steps==0:
                    self.model_ema.update_parameters(self.vqdiffusion.diffusion)
                
                if not self.run is None:
                    self.run.track(
                        loss,
                        name="diffusion_loss",
                        step=index,
                        context={"stage": "Diffusion"},
                    )

                if self.global_step % self.save_step == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Diffusion Loss : {loss:.4f}"
                    )

                    _, sampled_imgs = self.vqdiffusion.log_images(imgs[0][None])

                    if self.run is not None:
                        self.run.track(
                            Image(
                                torchvision.utils.make_grid(sampled_imgs)
                                .mul(255)
                                .add_(0.5)
                                .clamp_(0, 255)
                                .permute(1, 2, 0)
                                .to("cpu", torch.uint8)
                                .numpy()
                            ),
                            name="Images",
                            step=index,
                            context={"stage": "Diffusion"},
                        )

                if self.args.debug:
                    break
                
                self.global_step += 1

            if epoch == 0:
                print_gpu_memory_usage(self.logger)
            # self.save_checkpoint(self.experiment_dir)

            self.generate_images(epoch=epoch)
            torch.cuda.empty_cache()

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break
            
    
    def generate_images(self, n_images: int = 16, epoch = -1, flag ='generate'):

        self.logger.info(f"{self.model_name} vqdiffusion Generating {n_images} images...")
        
        self.vqdiffusion.eval()

        self.vqdiffusion = self.vqdiffusion.to(self.device)
        with torch.no_grad():
            sample_indices = self.vqdiffusion.sample(n_images)
            # print('sample_indices', sample_indices.shape) #torch.Size([16, 10, 256])
            # , 'return_all_timestamps', self.return_all_timestamps) 
            if self.return_all_timestamps and sample_indices.dim() == 3:                
                num_timestamps = sample_indices.shape[1]
                bs = sample_indices.shape[0]
                nrow = int(num_timestamps**0.5)
                for i in range(bs):
                    sampled_imgs = self.vqdiffusion.z_to_image(sample_indices[i])
                    sampled_imgs = sampled_imgs.detach()
                    sampled_imgs = (sampled_imgs - sampled_imgs.min()) / (sampled_imgs.max() - sampled_imgs.min())
                    torchvision.utils.save_image(
                        sampled_imgs,
                        os.path.join(self.save_img_dir, f"{self.model_name}_{flag}_epoch{epoch:03d}_bs{i:02d}.jpg"),
                        nrow=nrow
                        )
                    if i > 8:
                        break
            else:
                sampled_imgs = self.vqdiffusion.z_to_image(sample_indices)
                sampled_imgs = sampled_imgs.detach()#.cpu().permute(1, 2, 0).numpy()
                sampled_imgs = (sampled_imgs - sampled_imgs.min()) / (sampled_imgs.max() - sampled_imgs.min())


                torchvision.utils.save_image(
                    sampled_imgs,
                    os.path.join(self.save_img_dir, f"{self.model_name}_{flag}_epoch{epoch:03d}.jpg"),
                    nrow=4,
                )
        
    def save_checkpoint(self, checkpoint_dir: str, epoch: int = -1):
        """Saves the vqgan model checkpoints"""
        # filepath = os.path.join(checkpoint_dir, f"{self.model_name}Trans.pt")
        # if os.path.exists(filepath):
        #     os.remove(filepath)
        # torch.save(self.vqdiffusion.state_dict(), filepath)
        # self.logger.info(f"Checkpoint saved at {checkpoint_dir}")

        # save transformer model only
        weight_path = os.path.join(checkpoint_dir, 'diffusion.pt')
        if os.path.exists(weight_path):
            os.remove(weight_path)
        ckpt_dict = {}
        ckpt_dict['diffusion'] = self.vqdiffusion.diffusion.state_dict()
        ckpt_dict['optimizer'] = self.optim.state_dict()
        ckpt_dict['scheduler'] = self.scheduler.state_dict()
        ckpt_dict['global_step'] = self.global_step
        ckpt_dict['epoch'] = epoch
        torch.save(ckpt_dict, weight_path)
        self.logger.info(f"Diffusion model saved at {checkpoint_dir}")

