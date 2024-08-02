import torch.nn as nn
import torch
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys



if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '..'))
    

from network.vqDiffusion.submodule.unet3d import Unet3D


class GaussianDiffusion3D(nn.Module):
    def __init__(self,
                 image_sizes,
                 in_channels,
                 time_embedding_dim=256,
                 timesteps=1000,
                 sampling_timesteps = 1000,
                 base_dim=64,
                 dim_mults= [1, 2, 4, 8],
                 device	= 'cpu',
                 ):
        
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_sizes=image_sizes

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model=Unet3D(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)
        self.device=device

    def forward(self,x,noise):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x_t,t)

        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda"):
        x_t=torch.randn((n_samples,self.in_channels,self.image_sizes[0],self.image_sizes[1])).to(self.device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(self.device)
            t=torch.tensor([i for _ in range(n_samples)]).to(self.device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    

class VQGaussianDiffusion3DWrapper(nn.Module):
    def __init__(self,
        seq_length = 256,
        timesteps = 1000,
        sampling_timesteps = None,
        vocab_size = 1024,
        gaussian_dim = 512,
        indices_to_dist_fn = 'lookup_table',
        time_embedding_dim = 256,
        base_dim = 64,
        device = 'cpu',
        clipped_reverse_diffusion = True,
        ):
        
        super().__init__()
        self.device = device
        image_sizes = [seq_length, gaussian_dim] 
        self.diffusion=GaussianDiffusion3D(
            image_sizes, 
            1,
            time_embedding_dim,
            timesteps,
            sampling_timesteps,
            base_dim,
            device = device)
        
        self.codebook_size=vocab_size
        self.clipped_reverse_diffusion = clipped_reverse_diffusion

        self.register_buffer('gaussian_lookup_table', torch.rand(self.codebook_size, gaussian_dim))
        self.loss_fn=nn.MSELoss(reduction='mean')
    
    def indices_to_gaussian(self, indices):
        return self.gaussian_lookup_table[indices]

    def gaussian_to_indices(self, gaussian):

        # 确保输入的高斯分布向量形状为 [batch_size, num_indices, gaussian_dim]
        batch_size, num_indices, gaussian_dim = gaussian.shape

        # 展平gaussian为 [batch_size * num_indices, gaussian_dim]
        gaussian_flat = gaussian.view(-1, gaussian_dim)

        # 显式计算距离
        distances = (
            torch.sum(gaussian_flat**2, dim=-1, keepdim=True)
            + torch.sum(self.gaussian_lookup_table**2, dim=-1)
            - 2 * torch.matmul(gaussian_flat, self.gaussian_lookup_table.t())
        )

        # 找到距离最近的索引
        closest_indices = torch.argmin(distances, dim=-1)

        # 将索引还原为原始形状 [batch_size, num_indices]
        closest_indices = closest_indices.view(batch_size, num_indices)

        return closest_indices
    
    def forward(self,indices_x0):
        x0=self.indices_to_gaussian(indices_x0)
        if len(x0.shape)==3:
            x0=x0.unsqueeze(1)
        noise=torch.randn_like(x0).to(x0.device)
        pred_noise = self.diffusion(x0,noise)
        loss = self.loss_fn(pred_noise, noise)
        out = {}
        out['loss'] = loss
        return out
    
    @torch.no_grad()
    def sample(self, batch_size = 16, xt = None):
        samples_dist = self.diffusion.sampling(batch_size, clipped_reverse_diffusion = self.clipped_reverse_diffusion)
        if len(samples_dist.shape)==4:
            samples_dist=samples_dist.squeeze(1)
        sample_indices = self.gaussian_to_indices(samples_dist)
        return sample_indices