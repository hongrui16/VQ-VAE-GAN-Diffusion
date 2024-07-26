"""
https://github.com/dome272/VQGAN-pytorch/blob/main/transformer.py
"""

# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from network.vqDiffusion.submodule.diffusion_vq_official import Diffusion_VQ_Official
from network.vqDiffusion.submodule.diffusion_continuous import GaussianDiffusion2D
from network.vqDiffusion.submodule.unet2d import Unet2D



class VQDiffusion(nn.Module):
    def __init__(
        self,
        vqvae: nn.Module,
        device: str = "cpu",
        logger = None,
        config = None,
        
    ):
        super().__init__()
        diffusion_type = config['architecture']['diffusion']['diffusion_type']
        time_steps = config['architecture']['diffusion']['diffusion_steps']
        sampling_timesteps = config['architecture']['diffusion']['sampling_steps']
        objective = config['architecture']['diffusion']['objective']
        seq_length = config['architecture']['vqvae']['latent_channels']
        codebook_size = config['architecture']['vqvae']['num_codebook_vectors']
        indices_to_dist_fn = config['architecture']['diffusion']['indices_to_dist_fn']
        gaussian_dim = config['architecture']['diffusion']['gaussian_dim']

        assert diffusion_type in ['VQ_Official', 'Continuous'], "Diffusion type should be either 'VQ_Official' or 'Continuous'"
        assert indices_to_dist_fn in ['one_hot', 'lookup_table'], 'indices_to_dist_fn must be either one_hot or lookup_table'

        self.device = device
        self.vqvae = vqvae
        self.logger = logger

        distribute_dim = 1 # -1 or 1

        if diffusion_type == 'VQ_Official':
            unet_channels = codebook_size
            unet_out_dim = codebook_size - 1
        else:
            if indices_to_dist_fn == 'one_hot':
                if distribute_dim == 1:
                    unet_channels = codebook_size
                    unet_out_dim = codebook_size
                else:
                    unet_channels = seq_length
                    unet_out_dim = seq_length
            else:
                if distribute_dim == 1:
                    unet_channels = seq_length
                    unet_out_dim = seq_length
                else:
                    unet_channels = gaussian_dim
                    unet_out_dim = gaussian_dim

        self.unet = Unet2D(
                            dim = 64,
                            dim_mults = (1, 2, 4, 8),
                            channels = unet_channels,
                            out_dim= unet_out_dim
                            )
        
        if diffusion_type == 'VQ_Official':
            self.diffusion = Diffusion_VQ_Official(
                                            self.unet, diffusion_step = time_steps, 
                                            vocab_size = codebook_size,
                                            seq_len = seq_length, device = device,)
        else:                
            self.diffusion = GaussianDiffusion2D(self.unet,  seq_length = seq_length,
                                            timesteps = time_steps, sampling_timesteps = sampling_timesteps,vocab_size = codebook_size,
                                            distribute_dim = distribute_dim, gaussian_dim = gaussian_dim, 
                                            indices_to_dist_fn = indices_to_dist_fn,
                                            )
        
        diffusion_resume_path = config['architecture']['diffusion']['resume_path']
        if not diffusion_resume_path is None:
            if os.path.exists(diffusion_resume_path):
                self.diffusion.load_state_dict(torch.load(diffusion_resume_path))
                self.logger.info(f"diffusion loaded weight from {diffusion_resume_path}")
        
        freeze_weights = config['architecture']['diffusion']['freeze_weights'] or not config['architecture']['diffusion']['train_diffusion']
        if freeze_weights:
            for param in self.diffusion.parameters():
                param.requires_grad = False
            logger.info(f"Diffusion model is freezed")

    @torch.no_grad()
    def encode_to_z(self, x: torch.tensor) -> torch.tensor:
        """Processes the input batch ( containing images ) to encoder and returning flattened quantized encodings

        Args:
            x (torch.tensor): the input batch b*c*h*w

        Returns:
            torch.tensor: the flattened quantized encodings
        """
        # print('x:', x.shape) # x: torch.Size([bs, c, 256, 256])
        codebook_mapping, codebook_indices, q_loss = self.vqvae.encode(x) 
        # print('codebook_mapping:', codebook_mapping.shape) # codebook_mapping: torch.Size([bs, 256, 16, 16])
        # print('codebook_indices:', codebook_indices.shape) # codebook_indices: torch.Size([bs*256])
        codebook_indices = codebook_indices.view(codebook_mapping.shape[0], -1)
        # print('codebook_indices:', codebook_indices.shape) # codebook_indices: torch.Size([bs, 256])
        return codebook_mapping, codebook_indices

    @torch.no_grad()
    def z_to_image(
        self, indices: torch.tensor, p1: int = 16, p2: int = 16
    ) -> torch.Tensor:
        """Returns the decoded image from the indices for the codebook embeddings

        Args:
            indices (torch.tensor): the indices of the vectors in codebook to use for generating the decoder output
            p1 (int, optional): encoding size. Defaults to 16.
            p2 (int, optional): encoding size. Defaults to 16.

        Returns:
            torch.tensor: generated image from decoder
        """

        ix_to_vectors = self.vqvae.codebook.codebook(indices).reshape(
            indices.shape[0], p1, p2, 256
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqvae.decode(ix_to_vectors)
        return image

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        transformer model forward pass 

        Args:
            x (torch.tensor): Batch of images
        """

        # Getting the codebook indices of the image
        _, indices = self.encode_to_z(x)
        # print('indices', indices.shape)
        out = self.diffusion(indices)
        loss = out['loss']
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generating sample indices from the transformer

        Args:
            x (torch.Tensor): the batch of images

        Returns:
            torch.Tensor: _description_
        """
        self.diffusion.eval()
        if batch_size > 5:
            batch_size = 5
        sampling_indices = self.diffusion.sample(batch_size=batch_size)
        return sampling_indices

    @torch.no_grad()
    def log_images(self, x:torch.Tensor):
        """ Generating images using the transformer and decoder. Also uses encoder to complete partial images.   

        Args:
            x (torch.Tensor): batch of images

        Returns:
            Retures the input and generated image in dictionary and in a simple concatenated image
        """
        log = dict()

        batch_size = x.shape[0]
        if batch_size > 4:
            batch_size = 4
            x = x[:4]

        _, indices = self.encode_to_z(x) # Getting the indices of the quantized encoding


        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        # log["half_sample"] = half_sample
        # log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec))

    def load_checkpoint(self, path):
        """Loads the checkpoint from the given path."""

        self.load_state_dict(torch.load(path))

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)
