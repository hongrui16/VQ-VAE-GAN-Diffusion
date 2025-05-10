
# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from network.diffusion.unet_2d import Unet2D
from network.diffusion.gaussian_diffusion import  DiffusionModel, EMA

class V_VQDiffusion(nn.Module):
    def __init__(
        self,
        vqvae: nn.Module,
        device: str = "cpu",
        logger=None,
        config=None,
    ):
        super().__init__()
        seq_length = config['architecture']['vqvae']['num_latent_vec']
        codebook_size = config['architecture']['vqvae']['num_codebook_vectors']

        model_name = config['architecture']['model_name']
        diffusion_resume_path = config['architecture'][model_name]['resume_path']
        freeze_weights = config['architecture'][model_name]['freeze_weights']

        self.device = device
        self.vqvae = vqvae
        self.logger = logger

        self.codebook_size = codebook_size
        self.seq_length = seq_length

        # 获取embedding维度
        self.embedding_dim = self.vqvae.latent_vec_dim
        # 如果没有embedding_dim属性，可以从codebook.weight获取：
        # self.embedding_dim = self.vqvae.codebook.codebook.weight.shape[1]

        # 修改Unet2D的输入通道为embedding_dim
        self.unet = Unet2D(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=self.embedding_dim,
        )

        self.diffusion = DiffusionModel(self.unet, device=self.device)

        if diffusion_resume_path is not None:
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
        codebook_mapping, codebook_indices, q_loss = self.vqvae.encode(x)
        codebook_indices = codebook_indices.view(codebook_mapping.shape[0], -1)
        return codebook_mapping, codebook_indices

    @torch.no_grad()
    def z_to_image(
        self, indices: torch.tensor, p1: int = 16, p2: int = 16
    ) -> torch.Tensor:
        ix_to_vectors = self.vqvae.codebook.codebook(indices).reshape(
            indices.shape[0], p1, p2, 256
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqvae.decode(ix_to_vectors)
        return image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = self.encode_to_z(x)  # indices: [batch_size, seq_length]

        # 将indices映射为embedding向量
        embeddings = self.vqvae.codebook.codebook(indices)  # [batch_size, seq_length, embedding_dim]

        # 调整维度为 [batch_size, embedding_dim, seq_length]
        embeddings = embeddings.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length]

        batch_size = embeddings.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(embeddings)

        # 在embedding空间上进行Gaussian diffusion
        loss, _ = self.diffusion(embeddings, condition=None, t=t, noise=noise)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        self.diffusion.eval()

        # 从噪声开始采样，生成embedding向量
        sampled_embeddings = self.diffusion.ddim_sample(
            condition=None,
            x_t=torch.randn(batch_size, self.embedding_dim, self.seq_length).to(self.device),
            eta=0.0,
            sampling_timesteps=500,
            disable_print=True,
        )  # sampled_embeddings: [batch_size, embedding_dim, seq_length]

        # 调整维度为 [batch_size, seq_length, embedding_dim]
        sampled_embeddings = sampled_embeddings.permute(0, 2, 1)  # [batch_size, seq_length, embedding_dim]

        # 计算去噪后的embedding与codebook中所有embedding的距离
        codebook_embeddings = self.vqvae.codebook.codebook.weight  # [codebook_size, embedding_dim]
        sampled_embeddings = sampled_embeddings.unsqueeze(2)  # [batch_size, seq_length, 1, embedding_dim]
        codebook_embeddings = codebook_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, codebook_size, embedding_dim]

        # 计算欧氏距离
        distances = torch.sum((sampled_embeddings - codebook_embeddings) ** 2, dim=-1)  # [batch_size, seq_length, codebook_size]

        # 最近邻量化：选择距离最小的index
        predicted_indices = distances.argmin(dim=-1)  # [batch_size, seq_length]

        return predicted_indices

    @torch.no_grad()
    def log_images(self, x: torch.Tensor):
        log = dict()

        batch_size = x.shape[0]
        if batch_size > 4:
            batch_size = 4
            x = x[:4]

        _, indices = self.encode_to_z(x)
        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec

        return log, torch.concat((x, x_rec))

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)