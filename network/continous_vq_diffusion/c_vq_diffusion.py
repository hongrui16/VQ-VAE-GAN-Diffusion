"""
https://github.com/dome272/VQGAN-pytorch/blob/main/transformer.py
"""

# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from network.diffusion.unet_2d import Unet2D
from network.diffusion.gaussian_diffusion import  DiffusionModel, EMA


class C_VQDiffusion(nn.Module):
    def __init__(
        self,
        vqvae: nn.Module,
        device: str = "cpu",
        logger = None,
        config = None,
        
    ):
        super().__init__()
        seq_length = config['architecture']['vqvae']['num_latent_vec']
        codebook_size = config['architecture']['vqvae']['num_codebook_vectors']

        model_name = config['architecture']['model_name']
        # diffusion_type = config['architecture'][model_name]['diffusion_type']
        # time_steps = config['architecture'][model_name]['diffusion_steps']
        # sampling_timesteps = config['architecture'][model_name]['sampling_steps']
        # indices_to_dist_fn = config['architecture'][model_name]['indices_to_dist_fn']
        # gaussian_dim = config['architecture'][model_name]['gaussian_dim']
        # distribute_dim = config['architecture'][model_name]['distribute_dim']
        diffusion_resume_path = config['architecture'][model_name]['resume_path']
        freeze_weights = config['architecture'][model_name]['freeze_weights']
        # clipped_reverse_diffusion = config['architecture'][model_name]['clipped_reverse_diffusion']
        # unet_dim = config['architecture'][model_name]['unet_dim']
        # sample_method = config['architecture'][model_name]['sample_method']
        # loss_fn = config['architecture'][model_name]['loss_fn']
        # return_all_timestamps = config['architecture'][model_name]['return_all_timestamps']
        # compute_indices_recon_loss = config['architecture'][model_name]['compute_indices_recon_loss']

        # assert diffusion_type in ['VQ_Official', 'gaussiandiffusion2d', 'gaussiandiffusion3d'], "Diffusion type should be either 'VQ_Official', 'gaussiandiffusion2d', 'gaussiandiffusion3d'"
        # assert indices_to_dist_fn in ['one_hot', 'lookup_table'], 'indices_to_dist_fn must be either one_hot or lookup_table'

        self.device = device
        self.vqvae = vqvae
        self.logger = logger

        self.codebook_size = codebook_size
        self.seq_length = seq_length
        self.indices_width = config['architecture'][model_name]['indices_width']

        self.unet = Unet2D(
                    dim = 64,
                    dim_mults = (1, 2, 4, 8),
                    channels = self.indices_width,
                    )
    
        self.diffusion = DiffusionModel(self.unet, device=self.device)


        if not diffusion_resume_path is None:
            if os.path.exists(diffusion_resume_path):
                self.diffusion.load_state_dict(torch.load(diffusion_resume_path))
                self.logger.info(f"diffusion loaded weight from {diffusion_resume_path}")
        
        if freeze_weights:
            for param in self.diffusion.parameters():
                param.requires_grad = False
            logger.info(f"vqdiffusion model is freezed")

        self.diffusion.to(device)

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
        vqdiffusion model forward pass 

        Args:
            x (torch.tensor): Batch of images
        """

        # Getting the codebook indices of the image
        _, indices = self.encode_to_z(x)
        # print('indices', indices.shape)# # indices torch.Size([bs, 256])
        # print('max indices', indices.max()) # max indices tensor(2048-1)
        # print('min indices', indices.min()) # min indices tensor(0)
        ## add one dimension to the indices
        # indices = indices.unsqueeze(1) # torch.Size([bs, 1, 256])

        #exppand the indices 1st dimension to the indices_width
        indices = indices.unsqueeze(1).expand(-1, self.indices_width, -1) # torch.Size([bs, 1, 256])

        # print('indices', indices.shape) # # indices torch.Size([bs, 1, 256])
        # print('indices', indices[0,0])
        ## convert the indices to float
        indices = indices.float()
        indices = indices/self.codebook_size
        batch_size = indices.shape[0]
        # print('indices', indices[0])

        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(indices)
            
        loss, _ = self.diffusion(indices, condition=None, t=t, noise=noise) #x_0, condition, t, noise
        
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generating sample indices from the vqdiffusion

        Args:
            x (torch.Tensor): the batch of images

        Returns:
            torch.Tensor: _description_
        """
        self.diffusion.eval()
        sampling_indices = self.diffusion.ddim_sample(
            condition=None,
            x_t=torch.randn(batch_size, self.indices_width, self.seq_length).to(self.device),
            eta=0.0,
            sampling_timesteps=500,
            disable_print=True,
        )
        # print('sampling_indices:', sampling_indices.shape) # sampling_indices: torch.Size([bs, 1, 256])
        # sampling_indices = sampling_indices.squeeze(1)

        # compute the mean of the sampling indices on the 1st dimension
        sampling_indices = sampling_indices.mean(dim=1) # sampling_indices: torch.Size([bs, 256])

        # print('sampling_indices:', sampling_indices.shape) # sampling_indices: torch.Size([bs, 256])
        # print('sampling_indices:', sampling_indices[0]) # sampling_indices: tensor(0.0000)
        # clip the values to be between 0 and 1
        
        sampling_indices = sampling_indices * self.codebook_size
        sampling_indices = torch.clamp(sampling_indices, 0, self.codebook_size - 1)
        # print('max sampling_indices', sampling_indices.max()) # max sampling_indices tensor(255)
        # print('min sampling_indices', sampling_indices.min()) # min sampling_indices tensor(0)
        # convert the indices to int
        sampling_indices = sampling_indices.long()
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
