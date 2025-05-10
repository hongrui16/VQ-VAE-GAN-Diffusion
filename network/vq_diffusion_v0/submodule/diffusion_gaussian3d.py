import torch.nn as nn
import torch
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '..'))
    

from network.vq_diffusion_v0.submodule.unet3d import Unet3D

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gram_schmidt(vectors):
    orthogonal_vectors = []
    for v in vectors:
        for u in orthogonal_vectors:
            v -= np.dot(v, u) * u
        v /= np.linalg.norm(v)
        orthogonal_vectors.append(v)
    return np.array(orthogonal_vectors)

def orthogonal_vector_sampling(dim, num_vectors, device = 'cpu'):
    vectors = np.random.randn(num_vectors, dim)
    vectors = gram_schmidt(vectors)
    return torch.tensor(vectors, dtype=torch.float32, device=device) # shape: [num_vectors, dim]

def positional_encoding(dim, num_vectors, device='cpu'):
    position = np.arange(num_vectors)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe = np.zeros((num_vectors, dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float32, device=device)

def orthogonal_vector_sampling_with_positional_encoding(dim, num_vectors, device='cpu'):
    # 生成正交向量的查找表
    vectors = np.random.randn(num_vectors, dim)
    vectors = gram_schmidt(vectors)
    lookup_table = torch.tensor(vectors, dtype=torch.float32, device=device) # shape: [num_vectors, dim]
    
    # 生成位置编码
    pos_encoding = positional_encoding(dim, num_vectors, device=device)
    
    # 将位置编码加到查找表向量中
    lookup_table_with_pos = lookup_table + pos_encoding
    return lookup_table_with_pos


class GaussianDiffusion3D(nn.Module):
    def __init__(self,
                 image_sizes,
                 in_channels,
                 time_embedding_dim=256,
                 timesteps=1000,
                 sampling_timesteps = 500,
                 base_dim=64,
                 dim_mults= [1, 2, 4, 8],
                 device	= 'cpu',
                 auto_normalize = True,
                 loss_fn = 'noise_mse', # 'noise_mse' or 'elbo'
                 sample_method = 'ddim', # 'ddim' or 'ddpm'

                 ):
        
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_sizes=image_sizes
        self.sampling_timesteps = sampling_timesteps
        self.loss_fn = loss_fn
        assert loss_fn in ['noise_mse', 'elbo']

        self.ddim_sampling_eta = 0.0
        self.self_condition = False
        self.sample_method = sample_method

        self.num_timestamps = 24

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.model=Unet3D(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults).to(device)
        self.device=device

    
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
        pred=self.model(x_t,t = t)

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
        pred=self.model(x_t,t = t)
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
    
    def q_posterior(self, x_0, x_t, t):
        '''
        q(x_{t-1}|x_{0},x_{t})
        '''
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        posterior_mean = (1./torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * x_0)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t-1).reshape(x_t.shape[0], 1, 1, 1)
            posterior_variance = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            posterior_variance = torch.full_like(beta_t, fill_value=1e-20)  # 使用一个非常小的常数代替0

        posterior_log_variance_clipped  = self.posterior_log_variance_clipped.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
    #     pred_noise = self.model(x, t, x_self_cond)
    #     x_start = self.predict_start_from_noise(x, t, pred_noise)
    #     return pred_noise, x_start

    def predict_start_from_noise(self, x_t, t, pred_noise):
        """
        从给定的噪声图像 x_t 和预测的噪声 pred_noise 来估计原始的图像 x_0。
        """
        # # 计算 sqrt_recip_alphas_cumprod 和 sqrt_recipm1_alphas_cumprod
        # print('x_t.shape:', x_t.shape) #torch.Size([16, 1, 256, 512])
        # print('t:',t, t.shape) #torch.Size([])
        # print('self.sqrt_recip_alphas_cumprod.shape:', self.sqrt_recip_alphas_cumprod.shape) #torch.Size([500])
        # print('self.sqrt_recipm1_alphas_cumprod.shape:', self.sqrt_recipm1_alphas_cumprod.shape) #torch.Size([500])
        # t
        # sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        # sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        
        # # 从噪声图像 x_t 和预测的噪声 pred_noise 计算 x_0
        # pred_x_start = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * pred_noise
        
        # return pred_x_start
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x_t.shape[0])
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise
        )

    def p_mean_variance(self, x_t, t):
        self_cond = x_t if self.self_condition else None
        pred_noise=self.model(x_t, self_cond, t)
        pred_x_start = self.predict_start_from_noise(x_t, t, pred_noise)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_x_start, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance, pred_x_start
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        model_mean, posterior_variance, model_log_variance, pred_x_start  = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t, device=x_t.device) if t > 0 else 0. # no noise if t == 0
        x = model_mean + (0.5 * model_log_variance).exp() * noise
        return x
    
    @torch.no_grad()
    def ddpm_sample(self, n_samples, return_all_timestamps = False, clipped_reverse_diffusion=True):
        x_t = torch.randn((n_samples, self.in_channels, self.image_sizes[0], self.image_sizes[1])).to(self.device)
        imgs = [x_t]
        save_step = self.timesteps // self.num_timestamps
        if save_step == 0:
            save_step = 1

        for i in tqdm(range(self.timesteps-1, -1, -1), desc = "DDPM Sampling"):
            x = self.p_sample(x_t, i)
            if clipped_reverse_diffusion:
                x.clamp_(-1., 1.)

            x_t = x
            if return_all_timestamps and i % save_step == 0:
                imgs.append(x)
        if return_all_timestamps:
            ret = torch.cat(imgs, dim = 1)
        else:
            ret = x
        ret = self.unnormalize(ret)
        return ret
    
    @torch.no_grad()
    def ddim_sample(self, n_samples, return_all_timestamps = False, clipped_reverse_diffusion=True):
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, self.timesteps - 1, steps = self.sampling_timesteps)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # print('time_pairs:', time_pairs)
        shape = (n_samples, self.in_channels, self.image_sizes[0], self.image_sizes[1])
        img = torch.randn(shape, device = self.device)
        imgs = [img]

        x_start = None

        save_step = self.sampling_timesteps // self.num_timestamps
        if save_step == 0:
            save_step = 1
        i = 0
        for time, time_next in tqdm(time_pairs, desc = 'DDIM Sampling'):
            time = torch.tensor(time, device=self.device)
            time_next = torch.tensor(time_next, device=self.device)

            # print('time:', time, 'time_next:', time_next) # time: tensor(276, device='cuda:0') time_next: tensor(221, device='cuda:0')
            self_cond = x_start if self.self_condition else None
            # print('type(time):', type(time), 'time:', time)
            pred_noise=self.model(img, self_cond, time)
            # print('time:', time, 'pred_noise:', pred_noise.shape)  # time: tensor(276, device='cuda:0') pred_noise: torch.Size([16, 1, 256, 512])
            x_start = self.predict_start_from_noise(img, time, pred_noise)
            if clipped_reverse_diffusion:
                x_start.clamp_(-1., 1.)
            # print('img:', img.shape, 'x_start:', x_start.shape) #img: torch.Size([16, 1, 256, 512]) x_start: torch.Size([16, 1, 256, 512])
            if time_next < 0:
                img = x_start
                if return_all_timestamps and i % save_step == 0:
                    imgs.append(img)
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if return_all_timestamps and i % save_step == 0:
                # print('img.shape:', img.shape) #img.shape: torch.Size([16, 1, 256, 512])
                imgs.append(img)
            i += 1

        if return_all_timestamps:
            ret = torch.cat(imgs, dim = 1)
        else:
            ret = img
        # print('ret.shape:', ret.shape)
        ret = self.unnormalize(ret)
        return ret


    def kl_divergence(self, posterior_mean, posterior_variance, model_mean, model_variance):
        """
        Compute the KL divergence between the true posterior q(x_{t-1} | x_t, x_0) 
        and the model posterior p(x_{t-1} | x_t) for a given timestep t.
        """
        # posterior_mean and posterior_variance are the true posterior's mean and variance
        # model_mean and model_variance are the model's predicted mean and variance

        kl_div = 0.5 * (
            torch.log(model_variance) - torch.log(posterior_variance) + 
            (posterior_variance + (posterior_mean - model_mean).pow(2)) / model_variance - 1
        )

        # Sum over all dimensions
        kl_div = kl_div.sum(dim=(1, 2, 3)).mean()  # Mean over the batch
        return kl_div


    def negative_log_likelihood(self, xt, posterior_mean, posterior_log_variance):
        """
        计算给定 xt 和模型预测的后验均值、方差的负对数似然。
        
        xt: 模型在时间步 t 生成的样本
        posterior_mean: 模型预测的后验均值
        posterior_log_variance: 模型预测的后验对数方差
        
        返回值:
        NLL 损失值, 越小表示模型生成的样本与真实样本更接近。
        """
        # 计算负对数似然
        nll = 0.5 * torch.exp(-posterior_log_variance) * (xt - posterior_mean).pow(2) + 0.5 * posterior_log_variance
        
        # 求和并取平均
        return nll.sum(dim=(1, 2, 3)).mean()  # Mean over the batch
    
    def compute_elbo_loss(self, x_start, xt, t, noise = None):
        """
        Compute the ELBO loss for a given batch of data.
        """
        # 1. 计算真实的后验分布
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, xt, t)
        
        # 2. 计算模型预测的后验分布
        model_mean, model_variance, _, _ = self.p_mean_variance(xt, t)
        
        # 3. 计算KL散度
        kl_loss = self.kl_divergence(posterior_mean, posterior_variance, model_mean, model_variance)
        
        # 4. 计算负对数似然（重构误差）
        nll_loss = self.negative_log_likelihood(xt, posterior_mean, posterior_log_variance)
        
        # 5. 求和得到ELBO
        elbo_loss = kl_loss + nll_loss
        return elbo_loss
    
    
    def compute_noise_loss(self, x_start, xt, t, noise = None):
        # print('log_prob, x.shape', x.shape, x.unique()) #torch.Size([bs, 1, 32, 64]) tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')
        self_cond = x_start if self.self_condition else None
        pred_noise = self.model(xt, self_cond, t)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        return loss

    # def forward(self, x_start):
    #     b, c, h, w = x_start.shape
    #     t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
    #     noise = torch.randn_like(x_start, device=self.device)
    #     xt = self._forward_diffusion(x_start, t, noise)

    #     if self.loss_fn == 'elbo':
    #         loss = self.compute_elbo_loss(x_start, xt, t, noise)
    #     elif self.loss_fn == 'noise_mse':
    #         loss = self.compute_noise_loss(x_start, xt, t, noise)
    #     return loss

    # @torch.no_grad()
    # def sampling(self, batch_size = 16, return_all_timestamps = False, clipped_reverse_diffusion=True):
    #     # print('sample_method:', self.sample_method, 'return_all_timestamps:', return_all_timestamps)
    #     if self.sample_method == 'ddim':
    #         return self.ddim_sample(batch_size, return_all_timestamps, clipped_reverse_diffusion)
    #     elif self.sample_method == 'ddpm':
    #         return self.ddpm_sample(batch_size, return_all_timestamps, clipped_reverse_diffusion)
    #     else:
    #         raise ValueError(f"Invalid sample method: {self.sample_method}")

    
    def forward(self,x_start):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x_start.shape[0],)).to(x_start.device)
        noise = torch.randn_like(x_start, device=self.device)

        x_t=self._forward_diffusion(x_start,t,noise)
        pred_noise = self.model(x_t,t = t)
        loss = torch.nn.functional.mse_loss(pred_noise,noise)
        return loss

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True):
        x_t=torch.randn((n_samples,self.in_channels,self.image_sizes[0], self.image_sizes[1])).to(self.device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(self.device)
            t=torch.tensor([i for _ in range(n_samples)]).to(self.device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t



class VQGaussianDiffusion3DWrapper(nn.Module):
    def __init__(self,
        seq_length = 256,
        timesteps = 1000,
        sampling_timesteps = 500,
        vocab_size = 1024,
        gaussian_dim = 512,
        indices_to_dist_fn = 'lookup_table',
        time_embedding_dim = 256,
        base_dim = 64,
        device = 'cpu',
        sample_method = 'ddim', # 'ddim' or 'ddpm'
        loss_fn = 'noise_mse',# 'noise_mse' or 'elbo'
        return_all_timestamps = False,
        clipped_reverse_diffusion = False,
        compute_indices_recon_loss = False,
 
        ):
        
        super().__init__()
        
        assert loss_fn in ['noise_mse', 'elbo']
        assert sample_method in ['ddim', 'ddpm']

        self.loss_fn = loss_fn
        self.sample_method = sample_method

        self.device = device
        self.codebook_size=vocab_size
        self.timesteps = timesteps
        self.return_all_timestamps = return_all_timestamps
        self.in_channels = 1
        self.compute_indices_recon_loss = compute_indices_recon_loss
        self.belta = 0.01

        image_sizes = [seq_length, gaussian_dim] 
        self.diffusion=GaussianDiffusion3D(
            image_sizes, 
            self.in_channels,
            time_embedding_dim,
            timesteps,
            sampling_timesteps,
            base_dim,
            device = device,
            loss_fn=loss_fn,
            sample_method=sample_method,
            )
        
        # look_up_table = orthogonal_vector_sampling(gaussian_dim, vocab_size)
        # look_up_table = orthogonal_vector_sampling_with_positional_encoding(gaussian_dim, vocab_size, device = device)
        look_up_table = positional_encoding(gaussian_dim, vocab_size, device = device)
        
        self.register_buffer('gaussian_lookup_table', look_up_table)
    
    # def gaussian_to_indices(self, gaussian):
    #     # print('gaussian.shape', gaussian.shape) #torch.Size([16, 256, 512])
    #     # 确保输入的高斯分布向量形状为 [batch_size, num_indices, gaussian_dim]
    #     batch_size, num_indices, gaussian_dim = gaussian.shape

    #     # 展平gaussian为 [batch_size * num_indices, gaussian_dim]
    #     gaussian_flat = gaussian.reshape(-1, gaussian_dim)

    #     # 显式计算距离
    #     distances = (
    #         torch.sum(gaussian_flat**2, dim=-1, keepdim=True)
    #         + torch.sum(self.gaussian_lookup_table**2, dim=-1)
    #         - 2 * torch.matmul(gaussian_flat, self.gaussian_lookup_table.t())
    #     )

    #     # 找到距离最近的索引
    #     closest_indices = torch.argmin(distances, dim=-1)

    #     # 将索引还原为原始形状 [batch_size, num_indices]
    #     closest_indices = closest_indices.reshape(batch_size, num_indices)

    #     return closest_indices

    def indices_to_gaussian(self, indices):
        return self.gaussian_lookup_table[indices]

    def gaussian_to_indices(self, gaussian):
        # 确保输入的高斯分布向量形状为 [batch_size, num_indices, gaussian_dim]

        if len(gaussian.shape)==4:
            gaussian=gaussian.squeeze(1)
        # print('gaussian.shape', gaussian.shape) # torch.Size([16, 256, 512])
        
        batch_size, num_indices, gaussian_dim = gaussian.shape

        # 展平gaussian为 [batch_size * num_indices, gaussian_dim]
        gaussian_flat = gaussian.reshape(-1, gaussian_dim)

        # 规范化查找表向量
        self.gaussian_lookup_table = F.normalize(self.gaussian_lookup_table, p=2, dim=-1)

        # 规范化输入高斯向量（如果需要）
        gaussian_flat = F.normalize(gaussian_flat, p=2, dim=-1)

        # 使用torch.cdist计算距离
        distances = torch.cdist(gaussian_flat, self.gaussian_lookup_table)

        # 找到距离最近的索引
        closest_indices = torch.argmin(distances, dim=-1)

        # 将索引还原为原始形状 [batch_size, num_indices]
        closest_indices = closest_indices.reshape(batch_size, num_indices)

        return closest_indices


    def forward(self,indices_x0):
        x0=self.indices_to_gaussian(indices_x0)
        if len(x0.shape)==3:
            x0=x0.unsqueeze(1)
                
        t=torch.randint(0,self.timesteps,(x0.shape[0],)).to(x0.device)
        noise = torch.randn_like(x0, device=self.device)

        x_t=self.diffusion._forward_diffusion(x0,t,noise)
        pred_noise = self.diffusion.model(x_t, t = t)

        loss = torch.nn.functional.mse_loss(pred_noise,noise)

        if self.compute_indices_recon_loss:
            pred_x0 = self.diffusion.predict_start_from_noise(x_t, t , pred_noise)
            # print('pred_x0.shape:', pred_x0.shape) #torch.Size([16, 1, 256, 512])
            pred_indices = self.gaussian_to_indices(pred_x0)
            # 计算 indices 的重构损失
            recon_loss = F.mse_loss(pred_indices.float(), indices_x0.float())
            # 计算总损失
            loss = loss + self.belta * recon_loss
            

        out = {}
        out['loss'] = loss
        return out
    
    @torch.no_grad()
    def sample(self, batch_size = 16, xt = None):
        samples_dist = self.diffusion.sampling(batch_size, self.return_all_timestamps)
        sample_indices = []
        # print('samples_dist.shape', samples_dist.shape) #torch.Size([16, 10, 256, 512])
        if len(samples_dist.shape)==4:
            if self.return_all_timestamps:
                num_timestamps = samples_dist.shape[1]//self.in_channels
                for i in range(num_timestamps):
                    dist = samples_dist[:, i]
                    # print('dist.shape', dist.shape) # torch.Size([16, 256, 512])
                    sample_indice = self.gaussian_to_indices(dist)
                    sample_indices.append(sample_indice.unsqueeze(1))
                sample_indices = torch.cat(sample_indices, dim = 1)
            else:
                samples_dist=samples_dist.squeeze(1)
                sample_indices = self.gaussian_to_indices(samples_dist)
        return sample_indices
    

if __name__ == '__main__':
    diffusion = VQGaussianDiffusion3DWrapper(compute_indices_recon_loss = True)
    indices = torch.randint(0, 1024, (16, 256), dtype = torch.long)

    out = diffusion(indices)
    print(out['loss'])