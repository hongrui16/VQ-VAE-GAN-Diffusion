# Importing Libraries
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Run, Image
import os
import time
from utils.utils import print_gpu_memory_usage

class VAGANTransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        run: Run,
        experiment_dir: str = "experiments",
        device: str = "cuda",
        learning_rate: float = 4.5e-06,
        beta1: float = 0.9,
        beta2: float = 0.95,
        model_name: str = None,
        logger = None,
        save_img_dir = None,
        args = None,
        val_dataloader = None,

    ):
        self.vqganTransModel = model
        self.run = run
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.model_name = model_name
        self.save_img_dir = save_img_dir
        self.args = args
        self.val_dataloader = val_dataloader

        if not args.resume_ckpt_dir is None:
            weight_path = os.path.join(args.resume_ckpt_dir, 'transformer.pt')
            if os.path.exists(weight_path):
                self.vqganTransModel.transformer.load_state_dict(torch.load(weight_path))
                self.logger.info(f"transformer loaded from {args.resume_ckpt_dir}")

        self.vqganTransModel.to(device)
        self.device = device

        self.optim = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

    def configure_optimizers(
        self, learning_rate: float = 4.5e-06, beta1: float = 0.9, beta2: float = 0.95
    ):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        # Enabling weight decay to only certain layers
        for mn, m in self.vqganTransModel.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.vqganTransModel.transformer.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(beta1, beta2)
        )
        return optimizer

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int):
        self.vqganTransModel.train()
        for epoch in range(epochs):
            start_time = time.time()
            for index, imgs in enumerate(dataloader):
                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                logits, targets = self.vqganTransModel(imgs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                loss.backward()
                self.optim.step()

                self.run.track(
                    loss,
                    name="Cross Entropy Loss",
                    step=index,
                    context={"stage": "transformer"},
                )

                if index % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Cross Entropy Loss : {loss:.4f}"
                    )

                    _, sampled_imgs = self.vqganTransModel.log_images(imgs[0][None])

                    self.run.track(
                        Image(
                            torchvision.utils.make_grid(sampled_imgs)
                            .mul(255)
                            .add_(0.5)
                            .clamp_(0, 255)
                            .permute(1, 2, 0)
                            .to("cpu", torch.uint8)
                            .numpy()
                        ),
                        name="Transformer Images",
                        step=index,
                        context={"stage": "transformer"},
                    )

                if self.args.debug:
                    break

            if epoch == 0:
                print_gpu_memory_usage(self.logger)
            self.save_checkpoint(self.experiment_dir)

            self.vqganTrans_generate_images(epoch=epoch)
            torch.cuda.empty_cache()

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break
            
    
    def vqganTrans_generate_images(self, n_images: int = 5, epoch = -1):

        self.logger.info(f"{self.model_name} Transformer Generating {n_images} images...")
        
        self.vqganTransModel.eval()

        self.vqganTransModel = self.vqganTransModel.to(self.device)
        with torch.no_grad():
            for i in range(n_images):
                start_indices = torch.zeros((4, 0)).long().to(self.device)
                sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
                sos_tokens = sos_tokens.long().to(self.device)
                sample_indices = self.vqganTransModel.sample(
                    start_indices, sos_tokens, steps=256
                )
                sampled_imgs = self.vqganTransModel.z_to_image(sample_indices)
                torchvision.utils.save_image(
                    sampled_imgs,
                    os.path.join(self.save_img_dir, f"{self.model_name}Trans_epoch{epoch:03d}_{i}.jpg"),
                    nrow=4,
                )
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Saves the vqgan model checkpoints"""
        # filepath = os.path.join(checkpoint_dir, f"{self.model_name}Trans.pt")
        # if os.path.exists(filepath):
        #     os.remove(filepath)
        # torch.save(self.vqganTransModel.state_dict(), filepath)
        # self.logger.info(f"Checkpoint saved at {checkpoint_dir}")

        # save transformer model only
        transformer_path = os.path.join(checkpoint_dir, 'transformer.pt')
        if os.path.exists(transformer_path):
            os.remove(transformer_path)
        torch.save(self.vqganTransModel.transformer.state_dict(), transformer_path)
        self.logger.info(f"Transformer model saved at {checkpoint_dir}")