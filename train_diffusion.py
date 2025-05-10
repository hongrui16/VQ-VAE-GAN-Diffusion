import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid


# Assume Unet3D, DiffusionModel, EMA, ModelPrediction, and helper functions are imported
from network.diffusion.unet_2d import Unet2D
from network.diffusion.unet_3d import Unet3D
# from network.diffusion.unet_3d_v2 import UNet_3D
from network.diffusion.gaussian_diffusion import DiffusionModel, EMA, ModelPrediction, extract
from dataloader.build_dataloader import load_dataloader
# Data loading



def show_and_save_mnist_subset(dataset, n=8, save_path='mnist_samples.png'):
    fig, axs = plt.subplots(1, n, figsize=(12, 2))
    for i in range(n):
        img_tensor, label = dataset[i]
        img = img_tensor[0].numpy() * 0.5 + 0.5  # 反归一化

        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")


def save_batch_from_loader(dataloader, save_path='mnist_batch.png', nrow=8):
    # 获取第一个 batch
    batch = next(iter(dataloader))
    print("Batch shape:", batch.shape)

    if isinstance(batch, (list, tuple)):
        images = batch[0]  # 只取图像
    else:
        images = batch  # 如果没有 label
    
    # 反归一化 [-1,1] → [0,1]
    images = images * 0.5 + 0.5

    # 拼接为图像网格
    grid = make_grid(images[:nrow], nrow=nrow, padding=2)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()

    # 可视化并保存
    plt.figure(figsize=(nrow, 2))
    plt.imshow(ndarr.squeeze(), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved batch image to {save_path}")

# 用法：
# save_batch_from_loader(train_loader)

def save_images(images, path, nrow=8):
    images = images.detach().cpu()

    # 如果是 [B, H, W]，自动 unsqueeze 成 [B, 1, H, W]
    if images.ndim == 3:
        images = images.unsqueeze(1)  # [B, 1, H, W]

    # 动态归一化到 [0, 1]
    min_val = images.min()
    max_val = images.max()
    images = (images - min_val) / (max_val - min_val + 1e-8)

    # 拼接为网格 [C, H, W]
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid = grid.permute(1, 2, 0).numpy()  # [H, W, C]

    # 若为单通道，去掉冗余通道维度
    if grid.shape[2] == 1:
        grid = grid[:, :, 0]

    # 转为图像并保存
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)
    print(f"Saved image to {path} (min={min_val:.2f}, max={max_val:.2f})")

# Training function
def train(model, dataloader, optimizer, ema, num_epochs, device, 
          input_channel, img_size, save_dir='checkpoints', debug = False, squeeze_c = False):
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, images in enumerate(progress_bar):
            if squeeze_c:
                images = images.squeeze(1)
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample timesteps and noise
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(images)
            
            # Compute loss
            loss, _ = model(images, condition=None, t=t, noise=noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            if debug and batch_idx >= 10:  # Debug mode: limit to 10 batches
                break
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow,
            'epoch': epoch
        }, os.path.join(save_dir, f'checkpoint.pth'))
        
        # Generate samples every 5 epochs
        if epoch  % 5 == 0:
            generate_samples(model, ema, device, input_channel, img_size, num_samples=64, save_path=os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
        

        if debug:
            break

# Generation function
def generate_samples(model, ema, device, input_channel, img_size, num_samples=64,
                      save_path='samples.png', sampling_timesteps=500, eta=0.0, debug = False, squeeze_c = False):
    model.eval()
    ema.apply_shadow()  # Use EMA parameters for sampling
    
    # Initialize noise
    batch_size = min(num_samples, 64)  # Limit batch size to avoid memory issues
    if squeeze_c:
        # print("squeeze_c is True")
        x_t = torch.randn(batch_size, img_size, img_size, device=device)
    else:
        # print("squeeze_c is False")
        x_t = torch.randn(batch_size, input_channel, img_size, img_size, device=device)
    condition = None  # No conditioning for CIFAR-10
    
    # Generate samples using DDIM
    samples = model.ddim_sample(
        condition=condition,
        x_t=x_t,
        eta=eta,
        sampling_timesteps=sampling_timesteps,
        disable_print=False
    )
    # samples = model.ddpm_sample(
    #     condition=condition,
    #     x_t=x_t,
    #     disable_ddpm_print = debug
    # )
    print("Sample mean:", samples.mean().item(), "std:", samples.std().item(), "min:", samples.min().item(), "max:", samples.max().item())
    print("Sample shape:", samples.shape)

    # Save generated images
    save_images(samples, save_path)
    
    ema.restore()  # Restore original parameters
    model.train()

# Main execution
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train a diffusion model on CIFAR-10')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    if args.debug:
        print("Debug mode enabled. Reducing dataset size for faster training.")

    # Hyperparameters
    batch_size = 500
    num_epochs = 50
    num_timesteps = 1000
    sampling_timesteps = 500
    eta = 0.0
    learning_rate = 2e-4
    ema_decay = 0.999
    objective = 'pred_noise'
    log_dir = 'zlog'

    debug = args.debug
    if debug:
        batch_size = 2


    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slurm_job_id = os.environ.get('SLURM_JOB_ID', '0')

    dataset_name = 'cifar10'
    # dataset_name = 'mnist'
    dataset_name = 'Oxford102Flower'
    save_dir = os.path.join(log_dir, f'{dataset_name}/{time_stamp}_JID{slurm_job_id}')
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == 'mnist':
        input_channels = 1
        img_size = 32
    elif dataset_name == 'cifar10':
        input_channels = 3
        img_size = 32
    else:
        input_channels = 3
        img_size = 256
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    unet = Unet3D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=input_channels,
        dropout=0.1
    ).to(device)
    squeeze_c = False
    
    # unet = UNet_3D(
    #     in_channels=input_channels,
    #     out_channels=input_channels,
    # ).to(device)

    # unet = Unet2D(
    #     dim = 64,
    #     dim_mults = (1, 2, 4, 8),
    #     # dim_mults = (1, 2, 4),
    #     channels = img_size
    # ).to(device)
    # squeeze_c = True



    model = DiffusionModel(
        model=unet,
        num_timesteps=num_timesteps,
        objective=objective,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    ).to(device)
    
    # Optimizer and EMA
    optimizer = Adam(model.parameters(), lr=learning_rate)
    ema = EMA(model, decay=ema_decay)
    
    # Data loader
    train_loader, data_reduced = load_dataloader(
        split='train',
        batch_size=batch_size,
        logger=None,
        config=None,
        img_size =32,
        num_workers=5,
        in_channels=input_channels,
        dataset_name=dataset_name
    )

    # ## visualize an example
    # img = data_reduced[0][0]
    # img = img.permute(1, 2, 0).cpu().numpy()
    # img = (img * 255).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save(os.path.join(save_dir, 'example.png'))
    # print("Example image saved at:", os.path.join(save_dir, 'example.png'))
    
    save_batch_from_loader(train_loader, save_path=os.path.join(save_dir, 'mnist_batch.png'), nrow=8)
    print("Batch image saved at:", os.path.join(save_dir, 'mnist_batch.png'))

    # # Show and save a subset of MNIST dataset
    # if dataset_name == 'mnist':
    #     show_and_save_mnist_subset(data_reduced, n=8, save_path=os.path.join(save_dir, 'mnist_subset.png'))
    #     print("MNIST subset saved at:", os.path.join(save_dir, 'mnist_subset.png'))
    
    # Train
    train(model, train_loader, optimizer, ema, num_epochs, device, input_channels, 
        img_size, save_dir, debug = debug, squeeze_c = squeeze_c)
    
    # Generate final samples
    generate_samples(
        model,
        ema,
        device,
        input_channels, 
        img_size,
        num_samples=64,
        save_path=os.path.join(save_dir, 'final_samples.png'),
        sampling_timesteps=sampling_timesteps,
        eta=eta,
        debug = debug,
        squeeze_c = squeeze_c
    )