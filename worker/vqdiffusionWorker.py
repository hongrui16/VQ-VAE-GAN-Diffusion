# Importing Libraries
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Run, Image
import os
import time
import tqdm

from utils.utils import print_gpu_memory_usage

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
        learning_rate = config['trainer']['diffusion']['learning_rate']
        beta1 = config['trainer']['diffusion']['beta1']
        beta2 = config['trainer']['diffusion']['beta2']
        
        self.num_codebook_vectors = config['architecture']['vqvae']['num_codebook_vectors']
        self.seq_len = config['architecture']['vqvae']['latent_channels']

        self.vqdiffusion = model
        self.run = run
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.model_name = model_name
        self.save_img_dir = save_img_dir
        self.args = args
        self.val_dataloader = val_dataloader
        self.global_step = 0


        self.vqdiffusion.to(device)
        self.device = device

        train_diffusion = config['architecture']['diffusion']['train_diffusion']
        if train_diffusion:
            self.optim = self.configure_optimizers(
                learning_rate=learning_rate, beta1=beta1, beta2=beta2
            )

            num_iters_per_epoch = len(train_dataset)//config['dataset']['batch_size']
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


    def configure_optimizers(
        self, learning_rate: float = 4.5e-06, beta1: float = 0.9, beta2: float = 0.95
    ):               
        optimizer = torch.optim.AdamW(self.vqdiffusion.diffusion.parameters(), lr=learning_rate, betas=(beta1, beta2))
        return optimizer

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int):
        self.vqdiffusion.train()
        for epoch in range(epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for index, imgs in enumerate(tqdm_bar):
                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                loss = self.vqdiffusion(imgs)
                
                loss.backward()
                self.optim.step()

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
            self.save_checkpoint(self.experiment_dir)

            self.generate_images(epoch=epoch)
            torch.cuda.empty_cache()

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break
            
    
    def generate_images(self, n_images: int = 4, epoch = -1):

        self.logger.info(f"{self.model_name} Transformer Generating {n_images} images...")
        
        self.vqdiffusion.eval()

        self.vqdiffusion = self.vqdiffusion.to(self.device)
        with torch.no_grad():
            for i in range(n_images):
                sample_indices = self.vqdiffusion.sample(n_images)
                sampled_imgs = self.vqdiffusion.z_to_image(sample_indices)
                torchvision.utils.save_image(
                    sampled_imgs,
                    os.path.join(self.save_img_dir, f"{self.model_name}Trans_epoch{epoch:03d}_{i}.jpg"),
                    nrow=4,
                )
    
    def save_checkpoint(self, checkpoint_dir: str):
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
        torch.save(self.vqdiffusion.diffusion.state_dict(), weight_path)
        self.logger.info(f"Diffusion model saved at {checkpoint_dir}")

