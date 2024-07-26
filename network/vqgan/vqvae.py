"""

Implementing the main VQGAN, containing forward pass, lambda calculation, and to "enable" discriminator loss after a certain number of global steps.
"""

# Importing Libraries
import torch
import torch.nn as nn
import os, sys
from network.vqgan.encoder import Encoder
from network.vqgan.decoder import Decoder
from network.vqgan.codebook import CodeBook


class VQVAE(nn.Module):
    """
    VQGAN class

    Args:
        img_channels (int, optional): Number of channels in the input image. Defaults to 3.
        img_size (int, optional): Size of the input image. Defaults to 256.
        latent_channels (int, optional): Number of channels in the latent vector. Defaults to 256.
        latent_size (int, optional): Size of the latent vector. Defaults to 16.
        intermediate_channels (list, optional): List of channels in the intermediate layers of encoder and decoder. Defaults to [128, 128, 256, 256, 512].
        num_residual_blocks_encoder (int, optional): Number of residual blocks in the encoder. Defaults to 2.
        num_residual_blocks_decoder (int, optional): Number of residual blocks in the decoder. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        attention_resolution (list, optional): Resolution of the attention mechanism. Defaults to [16].
        num_codebook_vectors (int, optional): Number of codebook vectors. Defaults to 1024.
    """

    def __init__(
        self,
        logger = None,
        config = None,
    ):
        super().__init__()

        img_channels = config["architecture"]["vqvae"]["img_channels"]
        img_size = config["architecture"]["vqvae"]["img_size"]
        latent_channels = config["architecture"]["vqvae"]["latent_channels"]
        latent_size = config["architecture"]["vqvae"]["latent_size"]
        intermediate_channels = config["architecture"]["vqvae"]["intermediate_channels"]
        num_residual_blocks_encoder = config["architecture"]["vqvae"]["num_residual_blocks_encoder"]
        num_residual_blocks_decoder = config["architecture"]["vqvae"]["num_residual_blocks_decoder"]
        dropout = config["architecture"]["vqvae"]["dropout"]
        attention_resolution = config["architecture"]["vqvae"]["attention_resolution"]
        num_codebook_vectors = config["architecture"]["vqvae"]["num_codebook_vectors"]

        self.img_channels = img_channels
        self.num_codebook_vectors = num_codebook_vectors

        self.encoder = Encoder(
            img_channels=img_channels,
            image_size=img_size,
            latent_channels=latent_channels,
            intermediate_channels=intermediate_channels[:], # shallow copy of the link
            num_residual_blocks=num_residual_blocks_encoder,
            dropout=dropout,
            attention_resolution=attention_resolution,
        )

        self.decoder = Decoder(
            img_channels=img_channels,
            latent_channels=latent_channels,
            latent_size=latent_size,
            intermediate_channels=intermediate_channels[:], # shallow copy of the link
            num_residual_blocks=num_residual_blocks_decoder,
            dropout=dropout,
            attention_resolution=attention_resolution,
        )
        self.codebook = CodeBook(
            num_codebook_vectors=num_codebook_vectors, latent_dim=latent_channels
        )

        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        
        vqvae_resume_path = config["architecture"]["vqvae"]["resume_path"]
        if not vqvae_resume_path is None:
            if os.path.exists(vqvae_resume_path):
                self.load_checkpoint(vqvae_resume_path)
                logger.info(f"VQVAE loaded from {vqvae_resume_path}")

        freeze_weights = config['architecture']['vqvae']['freeze_weights'] or not config['architecture']['vqvae']['train_vqvae']

        if freeze_weights:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.codebook.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False
            logger.info(f"VAE model is freezed")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a single step of training on the input tensor x

        Args:
            x (torch.Tensor): Input tensor to the encoder.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """

        encoded_images = self.encoder(x)
        # print('encoded_images:', encoded_images.shape) #torch.Size([bs, 256, 16, 16])
        quant_x = self.quant_conv(encoded_images)
        # print('quant_x:', quant_x.shape) # torch.Size([bs, 256, 16, 16])

        codebook_mapping, codebook_indices, codebook_loss = self.codebook(quant_x)

        post_quant_x = self.post_quant_conv(codebook_mapping)
        # print('post_quant_x:', post_quant_x.shape) # post_quant_x: torch.Size([bs, 256, 16, 16])
        decoded_images = self.decoder(post_quant_x)

        return decoded_images, codebook_indices, codebook_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        quant_x = self.quant_conv(x)

        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_x)

        return codebook_mapping, codebook_indices, q_loss

    def decode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.post_quant_conv(x)
        x = self.decoder(x)

        return x

    def calculate_lambda(self, perceptual_loss, gan_loss):
        """Calculating lambda shown in the eq. 7 of the paper

        Args:
            perceptual_loss (torch.Tensor): Perceptual reconstruction loss.
            gan_loss (torch.Tensor): loss from the GAN discriminator.
        """

        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight

        # Because we have multiple loss functions in the networks, retain graph helps to keep the computational graph for backpropagation
        # https://stackoverflow.com/a/47174709
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lmda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lmda = torch.clamp(
            lmda, 0, 1e4
        ).detach()  # Here, we are constraining the value of lambda between 0 and 1e4,

        return 0.8 * lmda  # Note: not sure why we are multiplying it by 0.8... ?

    @staticmethod
    def adopt_weight(
        disc_factor: float, i: int, threshold: int, value: float = 0.0
    ) -> float:
        """Starting the discrimator later in training, so that our model has enough time to generate "good-enough" images to try to "fool the discrimator".

        To do that, we before eaching a certain global step, set the discriminator factor by `value` ( default 0.0 ) .
        This discriminator factor is then used to multiply the discriminator's loss.

        Args:
            disc_factor (float): This value is multiple to the discriminator's loss.
            i (int): The current global step
            threshold (int): The global step after which the `disc_factor` value is retured.
            value (float, optional): The value of discriminator factor before the threshold is reached. Defaults to 0.0.

        Returns:
            float: The discriminator factor.
        """

        if i < threshold:
            disc_factor = value

        return disc_factor

    def load_checkpoint(self, path):
        """Loads the checkpoint from the given path."""

        self.load_state_dict(torch.load(path))

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)
