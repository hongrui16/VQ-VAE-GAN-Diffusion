import os
import time
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import imageio

from utils.utils import weights_init, print_gpu_memory_usage, denormalize
from network.vae.vae import VAE
from aim import Image, Run


class VAEWorker:
    def __init__(
        self,
        run: Run = None,
        device: str = "cpu",
        experiment_dir=None,
        logger=None,
        train_dataset=None,
        args=None,
        val_dataloader=None,
        config=None,
        save_ckpt_dir=None,
    ):
        """Initializes the VAE trainer.

        Args:
            run: AIM run for logging.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
            experiment_dir: Directory to save checkpoints and GIFs.
            logger: Logger for training information.
            train_dataset: Training dataset.
            save_img_dir: Directory to save generated images.
            args: Command-line arguments (e.g., debug mode).
            val_dataloader: DataLoader for validation data.
            config: Configuration dictionary with model and training parameters.
        """
        model_name = 'vae'
        dataset_name = config['dataset']['dataset_name']
        img_size = config["dataset"]["img_size"][dataset_name]
        # img_channels = config['dataset']['img_channels'][dataset_name]
        batch_size = config['dataset']["batch_size"][model_name][dataset_name]

        self.img_size = img_size
        self.batch_size = batch_size
        
        learning_rate = config['trainer']['vae'].get('learning_rate', 2.25e-05)  # Default to 2.25e-05
        beta1 = config['trainer']['vae'].get('beta1', 0.5)  # Default to 0.5 for Adam optimizer
        beta2 = config['trainer']['vae'].get('beta2', 0.9)  # Default to 0.9 for Adam optimizer
        rec_loss_factor = config['trainer']['vae'].get('rec_loss_factor', 1.0)
        kld_weight = config['trainer']['vae'].get('kld_weight', 0.1)  # Default to 0.1 for better regularization

        # self.mean = config['dataset']['mean']
        # self.std = config['dataset']['std']
        self.mask_fn = config['dataset'].get('mask_fn', None)  # Generalized mask function
        self.dataset_name = dataset_name

        self.run = run
        self.device = device
        self.logger = logger
        self.model_name = model_name

        self.args = args
        self.val_dataloader = val_dataloader
        self.experiment_dir = experiment_dir
        if save_ckpt_dir is not None:
            self.checkpoint_dir = save_ckpt_dir
        else:
            self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        
        self.save_img_dir = os.path.join(experiment_dir, "images")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.save_img_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"Image directory: {self.save_img_dir}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"VAEWorker initialized with device: {self.device}")
        self.logger.info(f"Model name: {self.model_name}")

        self.vae = VAE(config=config).to(self.device)
        self.logger.info(f"VAE model created")

        # Optimizers
        self.configure_optimizers(learning_rate, beta1, beta2)
        
        # Hyperparameters
        self.rec_loss_factor = rec_loss_factor
        self.kld_weight = kld_weight

        # Training state
        self.global_step = 0

        self.gif_images = []
        self.save_step = config['trainer'].get('save_step', max(1, len(train_dataset) // self.batch_size // 10))
        self.logger.info(f"Save step set to {self.save_step}")
        

    def configure_optimizers(self, learning_rate: float = 2.25e-05, beta1: float = 0.5, beta2: float = 0.9):
        self.opt_vae = torch.optim.Adam(
            list(self.vae.encoder.parameters())
            + list(self.vae.decoder.parameters())
            + list(self.vae.fc_mu.parameters())
            + list(self.vae.fc_logvar.parameters()),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )

    def step(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a single training step for the VAE.

        Args:
            imgs: Input images (batch_size, channels, height, width).

        Returns:
            Tuple of (decoded images, VAE loss).
        """
        assert imgs.dim() == 4, f"Expected 4D input tensor, got {imgs.dim()}D"
        assert imgs.shape[1] == self.vae.in_channels, f"Expected {self.vae.in_channels} channels, got {imgs.shape[1]}"

        self.vae.train()
        decoded_images, mu, logvar = self.vae(imgs)
        recon_loss = F.mse_loss(decoded_images, imgs, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / torch.numel(imgs)
        vae_loss = self.rec_loss_factor * recon_loss + self.kld_weight * kld

        if self.run is not None:
            self.run.track(recon_loss, name="Reconstruction Loss", step=self.global_step)
            self.run.track(kld, name="KL Divergence", step=self.global_step)
            self.run.track(vae_loss, name="VAE Loss", step=self.global_step)

        self.opt_vae.zero_grad()
        vae_loss.backward()
        self.opt_vae.step()
        return decoded_images, vae_loss

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        self.vae.train()

        for epoch in range(epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for index, imgs in enumerate(tqdm_bar):
                imgs = imgs.to(self.device)

                decoded_images, vae_loss = self.step(imgs)

                if self.global_step % self.save_step == 0:
                    loginfo = f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | VAE Loss: {vae_loss:.4f}"
                    self.logger.info(loginfo)

                    self.save_reconstruction_gif(imgs, decoded_images)

                self.global_step += 1

                if self.args.debug:
                    break

            if epoch == 0:
                print_gpu_memory_usage(self.logger)

            self.save_checkpoint(epoch=epoch)


            torch.cuda.empty_cache()

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break

    @torch.no_grad()
    def generate_images(self, n_images: int = 16, dataloader=None, epoch=-1, sample_from_normal_dist=False):
        self.logger.info(f"{self.model_name} Generating {n_images} images...")

        self.vae.eval()

        with torch.no_grad():
            if sample_from_normal_dist:
                img_filepath = os.path.join(self.save_img_dir, f"{self.model_name}_epoch{epoch:03d}_sample.jpg") if epoch != -1 else os.path.join(self.save_img_dir, f"{self.model_name}_sample.jpg")
                z = torch.randn((n_images, self.vae.latent_channels, self.vae.latent_size, self.vae.latent_size)).to(self.device)

                generated_imgs = self.vae.decode(z)
                generated_imgs = (generated_imgs - generated_imgs.min()) / (generated_imgs.max() - generated_imgs.min())
                vutils.save_image(
                    generated_imgs,
                    img_filepath,
                    nrow=int(np.sqrt(n_images)),
                )
                return

            if dataloader is not None:
                imgs = next(iter(dataloader)).to(self.device)
                decoded_images, _ = self.vae(imgs)
                decoded_images = (decoded_images - decoded_images.min()) / (decoded_images.max() - decoded_images.min())
                vutils.save_image(
                    decoded_images,
                    os.path.join(self.save_img_dir, f"{self.model_name}_epoch{epoch:03d}_reconstruction.jpg"),
                    nrow=int(np.sqrt(n_images)),
                )

    def save_reconstruction_gif(self, imgs, decoded_images):
        batch_size = min(10, imgs.shape[0])

        horizontal_combined_input = torch.cat([imgs[i] for i in range(batch_size)], dim=2)
        horizontal_combined_output = torch.cat([decoded_images[i] for i in range(batch_size)], dim=2)
        vertical_combined = torch.cat((horizontal_combined_input, horizontal_combined_output), dim=1)

        gif_img = vutils.make_grid(vertical_combined.unsqueeze(0), nrow=1)
        gif_img = gif_img.detach().cpu().permute(1, 2, 0).numpy()
        gif_img = (gif_img - gif_img.min()) * (255 / (gif_img.max() - gif_img.min()))
        gif_img = gif_img.astype(np.uint8)

        if self.run is not None:
            self.run.track(
                Image(
                    vutils.make_grid(
                        torch.cat((imgs, decoded_images))
                    ).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                ),
                name=f"{self.model_name} Reconstruction",
                step=self.global_step,
                context={"stage": f"{self.model_name}"},
            )

        self.gif_images.append(gif_img)

        if len(self.gif_images) > 50:
            self.gif_images = self.gif_images[-50:]

        imageio.mimsave(
            os.path.join(self.experiment_dir, "reconstruction.gif"),
            self.gif_images,
            fps=5,
        )

    def save_checkpoint(self, epoch: int = -1):
        filepath = os.path.join(self.checkpoint_dir, "vae.pt")
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.opt_vae.state_dict(),
            'epoch': epoch,
        }, filepath)
        self.logger.info(f"Checkpoint saved at {filepath}")
