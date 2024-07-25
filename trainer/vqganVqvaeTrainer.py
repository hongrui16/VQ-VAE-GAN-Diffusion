"""
https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
"""

# Importing Libraries
import os

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from aim import Image, Run
import time
import tqdm
import cv2

from utils.utils import weights_init
from utils.utils import print_gpu_memory_usage, denormalize
from network.vqgan.discriminator import Discriminator


class VQGANVQVAETrainer:
    """Trainer class for VQGAN, contains step, train methods"""

    def __init__(
        self,
        model: torch.nn.Module,
        run: Run,
        device: str or torch.device = "cuda",
        experiment_dir: str = "./experiments",
        logger = None,
        train_dataset = None,
        save_img_dir = None,
        args = None,
        val_dataloader = None,
        config = None,
    ):
        model_name = config['architecture']['model_name']
        
        learning_rate = config['trainer']['vqvae']['learning_rate']
        beta1 = config['trainer']['vqvae']['beta1']
        beta2 = config['trainer']['vqvae']['beta2']        
        perceptual_loss_factor = config['trainer']['vqvae']['perceptual_loss_factor']
        rec_loss_factor = config['trainer']['vqvae']['rec_loss_factor']
        perceptual_model = config['trainer']['vqvae']['perceptual_model']

        disc_factor = config['trainer']['descriminator']['disc_factor']
        disc_start = config['trainer']['descriminator']['disc_start']

        self.mean = config['dataset']['mean']
        self.std = config['dataset']['std']
        self.get_hand_mask = config['dataset']['get_hand_mask'] ### only for InterHand26M dataset
        self.dataset_name = config['dataset']['dataset_name']

        self.run = run
        self.device = device
        self.logger = logger
        self.model_name = model_name
        self.save_img_dir = save_img_dir
        self.args = args
        self.val_dataloader = val_dataloader




        num_training_samples = len(train_dataset)
        save_max_sample = 50
        if num_training_samples <= save_max_sample:
            self.save_every = 1
        else:
            self.save_every = num_training_samples // save_max_sample
        
        # VQGAN parameters
        self.vqvae = model
            
        self.vqvae.to(self.device)
        if "vqgan" in model_name.lower():
            # Discriminator parameters
            self.discriminator = Discriminator(image_channels=self.vqvae.img_channels).to(self.device)
            self.discriminator.apply(weights_init)
            
            discrimator_weight_path = config['trainer']['descriminator']['resume_path']
            if not discrimator_weight_path is None:
                if os.path.exists(discrimator_weight_path):
                    self.discriminator.load_state_dict(torch.load(discrimator_weight_path))
                    self.logger.info(f"Discriminator loaded from {discrimator_weight_path}")

        # Loss parameters
        self.perceptual_loss = lpips.LPIPS(net=perceptual_model).to(self.device)

        # Optimizers
        self.opt_vqvae, self.opt_disc = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

        # Hyperprameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.rec_loss_factor = rec_loss_factor

        # Save directory
        self.expriment_save_dir = experiment_dir

        # Miscellaneous
        self.global_step = 0
        self.sample_batch = None
        self.gif_images = []
        self.logger = logger

    def configure_optimizers(
        self, learning_rate: float = 2.25e-05, beta1: float = 0.5, beta2: float = 0.9
    ):
        opt_vqvae = torch.optim.Adam(
            list(self.vqvae.encoder.parameters())
            + list(self.vqvae.decoder.parameters())
            + list(self.vqvae.codebook.parameters())
            + list(self.vqvae.quant_conv.parameters())
            + list(self.vqvae.post_quant_conv.parameters()),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )
        if "vqgan" in self.model_name:
            opt_disc = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=learning_rate,
                eps=1e-08,
                betas=(beta1, beta2),
            )
        else:
            opt_disc = None

        return opt_vqvae, opt_disc

    def step(self, imgs: torch.Tensor, masks:torch.Tensor = None) -> torch.Tensor:
        """Performs a single training step from the dataloader images batch

        For the VQGAN, it calculates the perceptual loss, reconstruction loss, and the codebook loss and does the backward pass.

        For the discriminator, it calculates lambda for the discriminator loss and does the backward pass.

        Args:
            imgs: input tensor of shape (batch_size, channel, H, W)
            masks: hand mask tensor of shape (batch_size, H, W), only for InterHand26M dataset

        Returns:
            decoded_imgs: output tensor of shape (batch_size, channel, H, W)
        """
        


        # Getting decoder output
        decoded_images, _, q_loss = self.vqvae(imgs)

        """
        =======================================================================================================================
        VQ Loss
        """
        perceptual_loss = self.perceptual_loss(imgs, decoded_images)            
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            self.perceptual_loss_factor * perceptual_loss
            + self.rec_loss_factor * rec_loss
        )
        # print('perceptual_rec_loss', perceptual_rec_loss.shape) #torch.Size([bs, 3, 256, 256])
        if masks is not None:
            # print('masks', masks.shape) ## torch.Size([bs, 256, 256])
            masks = masks.unsqueeze(1)
            # print('masks', masks.shape)
            # print('before mask, perceptual_rec_loss', perceptual_rec_loss.mean())
            perceptual_rec_loss = perceptual_rec_loss * masks
        perceptual_rec_loss = perceptual_rec_loss.mean()
        # print('after mask, perceptual_rec_loss', perceptual_rec_loss.mean())

        if self.model_name == "vqgan":
            """
            =======================================================================================================================
            Discriminator Loss
            """
            disc_real = self.discriminator(imgs)
            disc_fake = self.discriminator(decoded_images)

            disc_factor = self.vqvae.adopt_weight(
                self.disc_factor, self.global_step, threshold=self.disc_start
            )

            g_loss = -torch.mean(disc_fake)

            λ = self.vqvae.calculate_lambda(perceptual_rec_loss, g_loss)
            vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

            d_loss_real = torch.mean(F.relu(1.0 - disc_real))
            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
            gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
        else:
            gan_loss = None
            vq_loss = perceptual_rec_loss + q_loss
        # ======================================================================================================================
        # Tracking metrics

        self.run.track(
            perceptual_rec_loss,
            name="Perceptual & Reconstruction loss",
            step=self.global_step,
            context={"stage": f"{self.model_name}"},
        )

        self.run.track(
            vq_loss, name="VQ Loss", step=self.global_step, context={"stage": f"{self.model_name}"}
        )

        if self.model_name == "vqgan":
            self.run.track(
                gan_loss, name="GAN Loss", step=self.global_step, context={"stage": f"{self.model_name}"}
            )

        # =======================================================================================================================
        # Backpropagation
        self.opt_vqvae.zero_grad()
        vq_loss.backward(retain_graph=True)  # retain_graph is used to retain the computation graph for the discriminator loss

        if self.model_name == "vqgan":                    
            self.opt_disc.zero_grad()
            gan_loss.backward()
            self.opt_disc.step()

        self.opt_vqvae.step()


        # self.opt_disc.zero_grad()
        # gan_loss.backward()

        # self.opt_vqgan.step()
        # self.opt_disc.step()

        return decoded_images, vq_loss, gan_loss


    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 1,
    ):
        """Trains the VQGAN for the given number of epochs

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader to use.
            epochs (int, optional): number of epochs to train for. Defaults to 100.
        """
        
        self.vqvae.train()
        if self.model_name == "vqgan":
            self.discriminator.train()
        for epoch in range(epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for index, imgs in enumerate(tqdm_bar):
                # Training step
                imgs = imgs.to(self.device)
                if self.get_hand_mask and self.dataset_name == 'InterHand26M':
                    images = denormalize(imgs, self.mean, self.std)
                    hand_masks = images[:,0]>(20/255) ## 20/255 is the threshold for hand mask
                    # img_gray = (images[0,0].detach().cpu().numpy()*255).astype(np.uint8)
                    # mask = (hand_masks.detach().cpu().numpy()*255).astype(np.uint8).squeeze()
                    # print('img_gray', img_gray.shape)
                    # print('mask', mask.shape)
                    # composed_img = np.concatenate((img_gray, mask), axis=1)
                    # cv2.imwrite(f'./hand_mask_{index}.jpg', composed_img)
                else:
                    hand_masks = None
                # print('imgs', imgs.shape)
                decoded_images, vq_loss, gan_loss = self.step(imgs, hand_masks)

                # Updating global step
                self.global_step += 1


                if index % self.save_every == 0:
                    if self.model_name == "vqgan":
                        loginfo = f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | VQ Loss : {vq_loss:.4f} | Discriminator Loss: {gan_loss:.4f}"
                    else:
                        loginfo =f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | VQ Loss : {vq_loss:.4f}"
                    # Log the information
                    self.logger.info(loginfo)

                    # Only saving the gif for the first 2000 save steps
                    if self.global_step // self.save_every <= 2000:
                        self.sample_batch = (
                            imgs[:] if self.sample_batch is None else self.sample_batch
                        )

                        with torch.no_grad():
                            
                            """
                            Note : Lots of efficiency & cleaning needed here
                            """

                            gif_img = (
                                torchvision.utils.make_grid(
                                    torch.cat(
                                        (
                                            self.sample_batch,
                                            self.vqvae(self.sample_batch)[0],
                                        ),
                                    )
                                )
                                .detach()
                                .cpu()
                                .permute(1, 2, 0)
                                .numpy()
                            )

                            gif_img = (gif_img - gif_img.min()) * (
                                255 / (gif_img.max() - gif_img.min())
                            )
                            gif_img = gif_img.astype(np.uint8)

                            self.run.track(
                                Image(
                                    torchvision.utils.make_grid(
                                        torch.cat(
                                            (
                                                imgs,
                                                decoded_images,
                                            ),
                                        )
                                    ).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                                ),
                                name=f"{self.model_name} Reconstruction",
                                step=self.global_step,
                                context={"stage": f"{self.model_name}"},
                            )

                            self.gif_images.append(gif_img)

                        imageio.mimsave(
                            os.path.join(self.expriment_save_dir, "reconstruction.gif"),
                            self.gif_images,
                            fps=5,
                        )
                if self.args.debug:
                    break

            if epoch == 0:
                print_gpu_memory_usage(self.logger)
            self.save_checkpoint(self.expriment_save_dir)    

            self.generate_images(n_images=6, epoch=epoch)


            torch.cuda.empty_cache()

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break
    
    def generate_images(self, n_images: int = 6, dataloader: torch.utils.data.DataLoader = None, 
                               epoch = -1):

        self.logger.info(f"{self.model_name} Generating {n_images} images...")

        self.vqvae.eval()
        self.vqvae.to(self.device)

        if dataloader is None:
            dataloader = self.val_dataloader
        
        with torch.no_grad():
            for i, input_image in enumerate(dataloader):
                if i >= n_images:
                    break

                input_image = input_image.to(self.device)
                # print('input_image', input_image.shape)
                generated_imgs, codebook_indices, codebook_loss = self.vqvae(input_image)
                if input_image.shape[1] == 3: # RGB image
                    input_image = denormalize(input_image, self.mean, self.std)

                # 确保所有图像在 [0, 1] 范围内
                input_image = input_image.clamp(0, 1)
                generated_imgs = generated_imgs.clamp(0, 1)


                # 将原图和生成的图像拼接在一起
                combined_image = torch.cat((input_image, generated_imgs), dim=3)
                torchvision.utils.save_image(
                    combined_image,
                    os.path.join(self.save_img_dir, f"{self.model_name}_epoch{epoch:03d}_{i}.jpg"),
                    nrow=1,
                )
        

    def save_checkpoint(self, checkpoint_dir: str):
        """Saves the vqvae model checkpoints"""
        filepath = os.path.join(checkpoint_dir, "vqvae.pt")
        if os.path.exists(filepath):
            os.remove(filepath)
        torch.save(self.vqvae.state_dict(), filepath)


        if 'vqgan' in self.model_name.lower():
            filepath = os.path.join(checkpoint_dir, "discriminator.pt")
            torch.save(
                self.discriminator.state_dict(), filepath)
        self.logger.info(f"Checkpoint saved at {checkpoint_dir}")