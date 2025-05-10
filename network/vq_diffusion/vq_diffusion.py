import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel

# 设置设备

# 宏控制：是否启用文本条件
USE_TEXT_CONDITION = False  # 设置为 False 时，仅依赖图像 token



def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def index_to_log_onehot(x, num_classes):
    # 将索引转换为对数 one-hot，num_classes = K+1（包括 [MASK] token）
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

# 概率调度
def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.9):
    """
    计算扩散过程的概率调度，参考论文第4.1节。
    Args:
        time_step: 扩散步数 T。
        N: VQ-VAE codebook 大小 K（不包括 [MASK] token）。
        att_1: \bar{\alpha}_1 \approx 1。
        att_T: \bar{\alpha}_T \approx 0.000009。
        ctt_1: \bar{\gamma}_1 \approx 0。
        ctt_T: \bar{\gamma}_T \approx 0.9（论文第5节）。
    Returns:
        at: \alpha_t，单步保持不变概率。
        bt: \beta_t，单步均匀替换概率。
        ct: \gamma_t，单步替换为 [MASK] token 概率。
        att: \bar{\alpha}_t，累积保持不变概率。
        btt: \bar{\beta}_t，累积均匀替换概率。
        ctt: \bar{\gamma}_t，累积替换为 [MASK] token 概率。
    """

    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt

# Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ada_ln_scale = nn.Linear(embed_dim, embed_dim)
        self.ada_ln_bias = nn.Linear(embed_dim, embed_dim)

        if USE_TEXT_CONDITION:
            self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            self.norm_cross = nn.LayerNorm(embed_dim)

    def forward(self, x, t_emb, cond_emb=None):
        h = self.norm1(x)
        scale = self.ada_ln_scale(t_emb).unsqueeze(1)
        bias = self.ada_ln_bias(t_emb).unsqueeze(1)
        h = scale * h + bias

        attn_output, _ = self.self_attention(h, h, h)
        h = h + self.dropout(attn_output)

        if USE_TEXT_CONDITION:
            h = self.norm_cross(h)
            attn_output, _ = self.cross_attention(h, cond_emb, cond_emb)
            h = h + self.dropout(attn_output)

        h = self.norm2(h)
        ffn_output = self.ffn(h)
        h = h + self.dropout(ffn_output)
        return h

# Transformer 预测器
class TransformerPredictor(nn.Module):
    def __init__(self, num_tokens, embedding_dim, num_layers, num_heads, seq_len, diffusion_steps):
        """
        Args:
            num_tokens (int): 词汇表大小（K 个 vavae codebook token + 1 个 [MASK] token）
            embedding_dim (int): 嵌入维度。每个 token 的向量表示维度，也是 transformer 的 hidden size
            num_layers (int): Transformer 块的数量。
            num_heads (int): 注意力头的数量。
            seq_len (int): 序列长度。即 VQ-VAE encoder 输出的 token 数（通常是 H × W，例如 16×16 = 256）
            diffusion_steps (int): 扩散步数 T。
        """
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.time_embedding = nn.Embedding(diffusion_steps, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embedding_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, embedding_dim * 4) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embedding_dim, num_tokens - 1)  # 不包括 mask token
    
    def forward(self, indices, t, cond_emb=None):
        B, H, W = indices.shape
        indices = indices.view(B, -1)
        x = self.embedding(indices) + self.positional_encoding
        t_emb = self.time_embedding(t)

        for block in self.blocks:
            x = block(x, t_emb, cond_emb if USE_TEXT_CONDITION else None)

        logits = self.fc(x)  # [B, seq_len, num_tokens-1]
        return logits.view(B, H, W, -1)

# VQ-Diffusion 模型
class VQDiffusion(nn.Module):
    def __init__(self, vqvae, device = 'cpu'):
        super(VQDiffusion, self).__init__()

        self.vqvae = vqvae
        latent_dim = 64

        vqvae_codebook_size = self.vqvae.num_codebook_vectors
        content_seq_len = self.vqvae.encoder_output_hw ** 2 # 256x256 输入图像，vqvae encoder outputs 256*16*16, content_seq_len=16*16=256.
        self.num_classes = vqvae_codebook_size + 1  # 扩散状态数（K 个 vavae codebook token + 1 个 [MASK] token）

        diffusion_steps = 100 
        num_transformer_layers = 4
        num_heads = 4
        

        self.num_timesteps = diffusion_steps
        self.content_seq_len = content_seq_len
        self.auxiliary_loss_weight = 0.0005
        self.adaptive_auxiliary_loss = True
        self.mask_weight = [1.5, 1.0]
        self.truncation_rate = 0.86  # 论文中最佳 r=0.86
        self.device = device
        self.to(device)

        if USE_TEXT_CONDITION:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.predictor = TransformerPredictor(
            self.num_classes, latent_dim, num_transformer_layers, num_heads, content_seq_len, diffusion_steps
        )
        self.predictor.to(device)



        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1)
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)
        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        self.zero_vector = None

    def q_pred_one_timestep(self, log_x_t, t):
        log_at = extract(self.log_at, t, log_x_t.shape)
        log_bt = extract(self.log_bt, t, log_x_t.shape)
        log_ct = extract(self.log_ct, t, log_x_t.shape)
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )
        return log_probs

    def predict_start(self, log_x_t, t, cond_emb=None):
        x_t = log_onehot_to_index(log_x_t)
        out = self.predictor(x_t, t, cond_emb)
        log_pred = F.log_softmax(out.double(), dim=-1).float()
        batch_size = log_x_t.size(0)
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len, device=log_x_t.device) - 30
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        batch_size = log_x_start.size(0)
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1, device=log_x_t.device)
        log_zero_vector = torch.log(log_one_vector + 1e-30).expand(-1, -1, self.content_seq_len)

        log_qt = self.q_pred(log_x_t, t)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, t, cond_emb=None):
        log_x_recon = self.predict_start(log_x, t, cond_emb)
        log_model_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_prob

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def log_sample_categorical_truncated(self, logits):
        # 截断采样：保留 top-r token
        logits = logits.clone()
        batch_size, num_classes, seq_len = logits.shape
        logits = logits.permute(0, 2, 1)  # [batch_size, seq_len, num_classes]
        values, indices = torch.topk(logits, k=int(num_classes * self.truncation_rate), dim=-1)
        min_value = values[:, :, -1].unsqueeze(-1)
        logits[logits < min_value] = -float('inf')
        logits = logits.permute(0, 2, 1)  # 恢复形状
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def sample_time(self, b, device):
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def _train_loss(self, x, cond_emb=None, is_train=True):
        b, device = x.size(0), x.device
        log_x_start = index_to_log_onehot(x, self.num_classes)
        t, pt = self.sample_time(b, device)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        log_x0_recon = self.predict_start(log_xt, t, cond_emb if USE_TEXT_CONDITION else None)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)

        kl = (log_true_prob.exp() * (log_true_prob - log_model_prob)).sum(dim=1)
        mask_region = (xt == self.num_classes-1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = kl.sum(dim=1)

        decoder_nll = -(log_x_start.exp() * log_model_prob).sum(dim=1)
        decoder_nll = decoder_nll.sum(dim=1)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        vb_loss = kl_loss / pt

        if self.auxiliary_loss_weight != 0 and is_train:
            kl_aux = (log_x_start[:, :-1, :].exp() * (log_x_start[:, :-1, :] - log_x0_recon[:, :-1, :])).sum(dim=1)
            kl_aux = kl_aux * mask_weight
            kl_aux = kl_aux.sum(dim=1)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            addition_loss_weight = 1.0 if not self.adaptive_auxiliary_loss else (1 - t / self.num_timesteps) + 1.0
            vb_loss += addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt

        return log_model_prob, vb_loss

    def sample(self, num_samples, cond_text=None):
        """
        标准采样函数，从纯噪声 x_T 逐步去噪到 x_0，生成图像。
        
        Args:
            num_samples (int): 生成样本数量。
            cond_text (list of str, optional): 条件文本，仅在 USE_TEXT_CONDITION=True 时使用。
        
        Returns:
            torch.Tensor: 索引，形状为 [num_samples, sqrt(content_seq_len), sqrt(content_seq_len)]，
                        值在 [0, num_embeddings-1]，对应 VQ-VAE codebook。
        """
        device = next(self.parameters()).device
        cond_emb = None
        if USE_TEXT_CONDITION:
            if cond_text is None:
                cond_text = [""] * num_samples
            cond_inputs = self.tokenizer(cond_text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            cond_inputs = {k: v.to(device) for k, v in cond_inputs.items()}
            with torch.no_grad():
                cond_emb = self.text_encoder(**cond_inputs).last_hidden_state

        t = torch.full((num_samples,), self.num_timesteps - 1, device=device, dtype=torch.long)
        log_x_t = torch.zeros(num_samples, self.num_classes, self.content_seq_len, device=device)
        log_beta_T = self.log_cumprod_bt[-1]
        log_gamma_T = self.log_cumprod_ct[-1]
        log_x_t[:, :-1, :] = log_beta_T
        log_x_t[:, -1, :] = log_gamma_T
        log_x_t = torch.clamp(log_x_t, -70, 0)

        while t.max() >= 0:
            log_model_prob = self.p_pred(log_x_t, t, cond_emb)
            log_x_t = self.log_sample_categorical(log_model_prob)
            t = t - 1

        x_t = log_onehot_to_index(log_x_t)
        x_t = torch.clamp(x_t, max=self.num_classes-2)  # 确保索引 <= K-1，排除 [MASK] token
        x_t = x_t.view(num_samples, int(self.content_seq_len**0.5), int(self.content_seq_len**0.5))
        return x_t
    

    def fast_sample(self, num_samples, cond_text=None, skip_step=4):
        """
        快速采样函数，使用跳步策略和截断采样生成图像。
        
        Args:
            num_samples (int): 生成样本数量。
            cond_text (list of str, optional): 条件文本，仅在 USE_TEXT_CONDITION=True 时使用。
            skip_step (int): 跳步大小 \Delta_t，默认为 4（论文第4.2节）。
        
        Returns:
            torch.Tensor: 索引，形状为 [num_samples, sqrt(content_seq_len), sqrt(content_seq_len)]，
                        值在 [0, num_embeddings-1]，对应 VQ-VAE codebook。
        """
        device = next(self.parameters()).device
        cond_emb = None
        if USE_TEXT_CONDITION:
            if cond_text is None:
                cond_text = [""] * num_samples
            cond_inputs = self.tokenizer(cond_text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            cond_inputs = {k: v.to(device) for k, v in cond_inputs.items()}
            with torch.no_grad():
                cond_emb = self.text_encoder(**cond_inputs).last_hidden_state

        log_x_t = torch.zeros(num_samples, self.num_classes, self.content_seq_len, device=device)
        log_beta_T = self.log_cumprod_bt[-1]
        log_gamma_T = self.log_cumprod_ct[-1]
        log_x_t[:, :-1, :] = log_beta_T
        log_x_t[:, -1, :] = log_gamma_T
        log_x_t = torch.clamp(log_x_t, -70, 0)

        
        steps = torch.arange(self.num_timesteps-1, -1, -skip_step, device=device)
        for t_val in steps:
            t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
            log_model_prob = self.p_pred(log_x_t, t, cond_emb)
            log_x_t = self.log_sample_categorical_truncated(log_model_prob)
        

        x_t = log_onehot_to_index(log_x_t)
        x_t = torch.clamp(x_t, max=self.num_classes-2)  # 确保索引 <= K-1，排除 [MASK] token
        x_t = x_t.view(num_samples, int(self.content_seq_len**0.5), int(self.content_seq_len**0.5))
        return x_t


if __name__ == "__main__":
    pass