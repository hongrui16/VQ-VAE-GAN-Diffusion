import torch
import torch.nn as nn
import os, sys


if __name__ == "__main__":
    sys.path.append("../..")

from network.common.encoder import Encoder
from network.common.decoder import Decoder

class VAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        
        intermediate_channels = config["architecture"]["vae"].get("intermediate_channels", [128, 128, 256, 256, 512])
        num_residual_blocks_encoder = config["architecture"]["vae"].get("num_residual_blocks_encoder", 2)
        num_residual_blocks_decoder = config["architecture"]["vae"].get("num_residual_blocks_decoder", 3)
        dropout = config["architecture"]["vae"].get("dropout", 0.0)
        attention_resolution = config["architecture"]["vae"].get("attention_resolution", [32])
        latent_size = config["architecture"]["vae"].get("latent_size", 16) #### latent size is shape of encoded image feature, e.g.  batch_size x 256 x 16 x 16. 16 is the size of the latent feature
        latent_channels = config["architecture"]["vae"].get("latent_channels", 256) ## latent channels is the number of channels in the latent feature, e.g. 256



        dataset_name = config['dataset']['dataset_name']
        img_size = config["dataset"]["img_size"][dataset_name]
        img_channels = config['dataset']['img_channels'][dataset_name]

        self.encoder = Encoder(
            img_channels=img_channels,
            image_size=img_size,
            latent_channels=latent_channels,
            intermediate_channels=intermediate_channels[:],
            num_residual_blocks=num_residual_blocks_encoder,
            dropout=dropout,
            attention_resolution=attention_resolution,
        )
        ## if input image is 256x256, the output of encoder is 256x16x16

        self.decoder = Decoder(
            img_channels=img_channels,
            latent_channels=latent_channels,
            latent_size=latent_size,
            intermediate_channels=intermediate_channels[:],
            num_residual_blocks=num_residual_blocks_decoder,
            dropout=dropout,
            attention_resolution=attention_resolution,
        )
        ## if input for decoder is 256x16x16, the output of decoder is 3x256x256

        self.in_channels = img_channels
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        # Latent projection layers
        self.fc_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar) 
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)



if __name__ == "__main__":
    # Test the VAE class
    config = {
        "architecture": {
            "vae": {
                "intermediate_channels": [128, 128, 256, 256, 512],
                "num_residual_blocks_encoder": 2,
                "num_residual_blocks_decoder": 3,
                "dropout": 0.0,
                "attention_resolution": [32],
                "latent_size": 16,
                "latent_channels": 256,
            }
        },
        "dataset": {
            "dataset_name": "example_dataset",
            "img_size": {"example_dataset": 256},
            "img_channels": {"example_dataset": 3},
        },
    }
    vae = VAE(config=config)
    x = torch.randn(1, 3, 256, 256)  # Example input tensor
    output = vae(x)
    print(output[0].shape)  # Should be (1, 3, 256, 256)
    print(output[1].shape)  # Should be (1, 256, 16, 16)
    print(output[2].shape)  # Should be (1, 256, 16, 16)
    print("VAE model created successfully")