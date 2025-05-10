import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple
from tqdm import tqdm


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x0'])

def extract(tensor, t, shape):
    """
    动态提取时间步索引对应的参数值，并调整形状以匹配目标张量
    Args:
        tensor (Tensor): 一维时间步参数 (timesteps,)
        t (Tensor): 批量时间步索引 (batch_size,)
        shape (torch.Size): 目标形状 (batch_size, ...)
    Returns:
        Tensor: 调整后形状的张量 (batch_size, 1, 1, ...)
    """
    batch_size = t.shape[0]
    out = torch.gather(tensor, 0, t)  # 从一维张量中提取每个时间步索引的值
    return out.view(batch_size, *([1] * (len(shape) - 1)))  # 调整形状以支持广播


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param

    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name])

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name])



class DiffusionModel(nn.Module):
    def __init__(self, model, num_timesteps = 1000, objective = 'pred_noise', beta_start=0.0001, beta_end=0.02, device='cpu', **kwargs):
        super(DiffusionModel, self).__init__()  # Add this line

        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        self.objective = objective
        assert self.objective in ['pred_noise', 'pred_x0', 'pred_v']

        self.betas = self.get_beta_schedule(num_timesteps, beta_start, beta_end).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod, (0, 1), value=1)


        # Precompute buffers for efficiency
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod = (1 / self.alphas_cumprod).sqrt()
        self.sqrt_recipm1_alphas_cumprod = (1 / self.alphas_cumprod - 1).sqrt()


    def get_beta_schedule(self, num_timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, num_timesteps)

    def add_noise(self, x_0, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        noisy_x = x_0 * sqrt_alphas_cumprod_t + noise * sqrt_one_minus_alphas_cumprod_t
        return noisy_x


    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x_0):
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip_alphas_cumprod_t * x_t - x_0) / sqrt_recipm1_alphas_cumprod_t


    def predict_v(self, x_0, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_0
    

    def predict_start_from_v(self, x_t, t, v):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v

    def model_predictions(self, x_t, condition, t, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x_t, condition, t)
        
        # Optional clipping function
        maybe_clip = (
            lambda x: torch.clamp(x, min=-1., max=1.) if clip_x_start else x
        )

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_0 = self.predict_start_from_noise(x_t, t, pred_noise)
            x_0 = maybe_clip(x_0)
            if rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)
        
        elif self.objective == 'pred_x0':
            x_0 = model_output
            x_0 = maybe_clip(x_0)
            pred_noise = self.predict_noise_from_start(x_t, t, x_0)
        
        elif self.objective == 'pred_v':
            v = model_output
            x_0 = self.predict_start_from_v(x_t, t, v)
            x_0 = maybe_clip(x_0)
            pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        return ModelPrediction(pred_noise=pred_noise, pred_x0=x_0)

    def forward(self, x_0, condition, t, noise):
        x_t = self.add_noise(x_0, t, noise)
        model_output = self.model(x_t, condition, t)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_0
        elif self.objective == 'pred_v':
            v = self.predict_v(x_0, t, noise)
            target = v
            
        loss = F.mse_loss(model_output, target)

        return loss, x_t


    @torch.no_grad()
    def ddim_sample(self, condition: torch.Tensor, x_t: torch.Tensor,
                     eta: float = 0.0, sampling_timesteps: int = None, 
                     disable_print: bool = True) -> torch.Tensor:

        batch_size = x_t.shape[0]
        device = self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = sampling_timesteps or total_timesteps

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))



        for time, time_next in tqdm(time_pairs, desc="DDIM step", disable=disable_print):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred = self.model_predictions(x_t, condition, time_cond)
            pred_noise, x_start = pred.pred_noise, pred.pred_x0

            if time_next == -1:
                x_t = x_start
                continue    

            hat_alpha_t = self.alphas_cumprod[time]
            hat_alpha_t_1 = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - hat_alpha_t / hat_alpha_t_1) * (1 - hat_alpha_t_1) / (1 - hat_alpha_t)).sqrt()
    
            c = (1 - hat_alpha_t_1 - sigma**2).sqrt()

            noise = torch.randn_like(x_t)

            x_t = x_start * hat_alpha_t_1.sqrt() + c * pred_noise + sigma * noise
        
        return x_t

    @torch.no_grad()
    def ddpm_sample(self, condition, x_t, disable_ddpm_print = True):

        batch_size = x_t.shape[0]
        for t in tqdm(reversed(range(self.num_timesteps)), desc="DDPM step", disable = disable_ddpm_print):
        
            time_cond = torch.full((batch_size,), t, device=self.device).long()
            pred = self.model_predictions(x_t, condition, time_cond)
            pred_noise, x_start = pred.pred_noise, pred.pred_x0
            
            alpha_t = self.alphas[t]
            hat_alpha_t = self.alphas_cumprod[t]


            beta_t = self.betas[t]

            sigma_t = beta_t.sqrt() if t > 0 else torch.zeros_like(beta_t)

            z = torch.randn_like(x_t, device=self.device) if t > 0 else torch.zeros_like(x_t)
        
            x_t = (1 / alpha_t.sqrt()) * (x_t - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * pred_noise) + sigma_t * z


        return x_t



            
if __name__ == "__main__":
    vector_dim = 106  # Dimension of the input vector (MANO pose + shape)
    condition_dim = 8  # Dimension of the condition vector
    num_timesteps = 10  # Number of diffusion timesteps
    bs = 2  # Batch size
    sampling_timesteps = 5
    eta = 0.01
    objective = 'pred_noise'
    # objective = 'pred_x0'

    # model = DiffusionModel(num_timesteps=num_timesteps, objective = objective, vector_dim=vector_dim, condition_dim=condition_dim)
    # x_t = torch.randn(bs, vector_dim)
    # condition = torch.randn(bs, condition_dim)
    # x_sample = model.ddim_sample(condition, x_t, eta, sampling_timesteps)
    # print(x_sample.shape)  # torch.Size([2, 106])

    # x_sample = model.ddpm_sample(condition, x_t)
    # print(x_sample.shape)  # torch.Size([2, 106])

    # t = torch.randint(0, num_timesteps, (bs,))

    # noise = torch.randn(bs, vector_dim)

    # loss, x_t = model(x_t, condition, t, noise)
    # print(loss)  # tensor(0.0001, grad_fn=<MseLossBackward>)
    # print(x_t.shape)  # torch.Size([2, 106])

    