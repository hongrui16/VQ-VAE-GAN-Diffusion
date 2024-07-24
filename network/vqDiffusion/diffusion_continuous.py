import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
import os, sys


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    

from network.vqDiffusion.unet2d import Unet2D

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# class DiffusionUnet2d(nn.Module):
#     def __init__(self, model, seq_length = 16, num_timesteps=500, sampling_timesteps = 500, 
#                  vocab_size = 1024, schedule='linear', device='cpu'):
#         super(DiffusionUnet2d, self).__init__()
#         self.model = model
#         self.num_timesteps = num_timesteps
#         self.sampling_timesteps = sampling_timesteps
#         self.vocab_size = vocab_size
#         self.seq_length = seq_length
#         self.device = device


#         if schedule == 'linear':
#             self.betas = linear_beta_schedule(num_timesteps)
#         elif schedule == 'cosine':
#             self.betas = cosine_beta_schedule(num_timesteps)
#         else:
#             raise ValueError("Unsupported schedule type")

#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)



#     def random_mask(self, indices, mask_ratio):
#         mask = torch.rand(indices.shape, device=self.device) < mask_ratio
#         masked_indices = indices.clone()
#         masked_indices[mask] = 0  # 用0值代替被mask掉的indices
#         return masked_indices

#     # 替换 indices 的函数
#     def random_replace(self, indices, replace_ratio):
#         mask = torch.rand(indices.shape, device = self.device) < replace_ratio
#         random_indices = torch.randint_like(indices, high=self.vocab_size, device=self.device)
#         replaced_indices = torch.where(mask, random_indices, indices)
#         return replaced_indices

    

#     def add_noise(self, x_start, t):
#         alpha_t = self.alphas_cumprod[t].item()
#         mask_ratio = 1 - alpha_t
#         replace_ratio = 1 - alpha_t
#         if torch.rand(1).item() < 0.5:
#             xt = self.random_mask(x_start, mask_ratio=mask_ratio)
#         else:
#             xt = self.random_replace(x_start, replace_ratio=replace_ratio)
#         noise = x_start - xt
#         return xt, noise


#     def predict_indices(self, xt, t):
#         # xt shape: (batch_size, 1, seq_len)
#         hidden_representations = self.model(xt)
#         predicted_indices = hidden_representations.argmax(dim=-1)  # 假设模型输出的是 logits
#         return predicted_indices

#     def denoise_step(self, xt, t):
#         predicted_indices = self.predict_indices(xt, t)
#         denoised_xt = self.codebook[predicted_indices]
#         return denoised_xt



#     def sample(self, batch_size = 16):
#         xt = torch.randn(0, self.vocab_size, (batch_size, 1, self.seq_length), device=self.device)
#         for t in reversed(range(self.sampling_timesteps)):
#             xt = self.denoise_step(xt, t)
#         return xt ## x0


    
#     def forward(self, x_start):
#         b, c, n = x_start.shape
#         device = x_start.device

#         t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

#         # noise sample
#         xt, noise = self.add_noise(x_start = x_start, t = t, noise = noise)

#         # predict and take gradient step
#         pred_noise = self.model(xt, t)

        
#         return pred_noise, noise
    
#     def compute_loss(self, pred, target):
#         loss = F.mse_loss(pred, target, reduction = 'mean')
#         return loss
        

class GaussianDiffusion2D(Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        vocab_size = 1024,
        distribute_dim = -1, ## the dim is the distribution of indices probability
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective
        self.codebook_size = vocab_size
        self.distruibute_dim = distribute_dim


        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def indices_to_smooth_onehot(self, x0, smoothing=0.1):
        onehot = F.one_hot(x0, num_classes=self.codebook_size).float()
        if self.distruibute_dim == 1:
            onehot = onehot.permute(0, 2, 1)  # 将one-hot编码的维度调整到第1维

        return onehot * (1 - smoothing) + smoothing / self.codebook_size


    def predict_start_from_noise(self, x_t, t, noise):
        # alpha = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        # beta = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        # # print('alpha:', alpha.shape, alpha)
        # # print('beta:', beta.shape, beta)
        # x_temp= extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        # # print('x_temp:', x_temp.shape, x_temp[:,0])

        # noise_temp = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        # print('noise_temp:', noise_temp.shape, noise_temp[:,0])
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, x_self_cond, t)
        # print('model_output:', model_output.shape, model_output[:,0])
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            # print('xt:', x.shape, x[:,0])
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            # print('1 x_start:', x_start.shape, x_start[:,0])
            x_start = maybe_clip(x_start)
            # print('2 x_start:', x_start.shape, x_start[:,0])

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, batch_size, clip_denoised = True):
        device = self.betas.device
        batch_size = batch_size

        input_indices = torch.randint(0, self.codebook_size, (batch_size, self.seq_length), device= device)  # 示例离散表示，长度为16

        img = self.indices_to_smooth_onehot(input_indices)
        # print('img:', img.shape, img[:,0])
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'p_sample_loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        # img = self.unnormalize(img)
        if self.distruibute_dim == 1:
            img = img.permute(0, 2, 1)
        img = img.argmax(dim = -1).float()
        return img

    @torch.no_grad()
    def ddim_sample(self, batch_size, clip_denoised = True):
        
        device, total_timesteps, sampling_timesteps, eta, objective = self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        input_indices = torch.randint(0, self.codebook_size, (batch_size, self.seq_length), device= device)  # 示例离散表示，长度为16

        img = self.indices_to_smooth_onehot(input_indices)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'ddim_sample time step'):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)
            # print('pred_noise:', pred_noise.shape, pred_noise[:,0])
            # print('x_start:', x_start.shape, x_start[:,0])

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            # print('img:', img.shape, img[:,0])
        # img = self.unnormalize(img)
        if self.distruibute_dim == 1:
            img = img.permute(0, 2, 1)
        img = img.argmax(dim = -1).float()
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(batch_size)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        # print('x_self_cond:', x_self_cond)
        # print('t', t)
        model_out = self.model(x, x_self_cond, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x0, *args, **kwargs):
        b, c  = x0.shape
        device = x0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        onehot_smoothed_x0 = self.indices_to_smooth_onehot(x0)
        out = {}
        loss = self.p_losses(onehot_smoothed_x0, t, *args, **kwargs)
        out['loss'] = loss 
        return out


# trainer class

if __name__ == '__main__':

    seq_length = 8
    codebook_size = 16
    bs = 1
    timesteps = 10
    sampling_timesteps = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    distribute_dim = -1 # -1 or 1

    if distribute_dim == 1:
        model = Unet2D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = codebook_size,
            out_dim= codebook_size
        )
    else:
        model = Unet2D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = seq_length,
        out_dim= seq_length
        )


    model.to(device)
    diffusion = GaussianDiffusion2D(model, seq_length = seq_length, timesteps = timesteps, 
                                    sampling_timesteps = sampling_timesteps, vocab_size = codebook_size, distribute_dim = distribute_dim)

    diffusion.to(device)

    input_indices = torch.randint(0, codebook_size, (bs, seq_length), device= device)  # 示例离散表示，长度为16

    print('input_indices:', input_indices)
    out = diffusion(input_indices)
    loss = out['loss']
    print('loss:', loss)    

    sampling_indices = diffusion.sample(batch_size = 1)
    print('sampling_indices:', sampling_indices)