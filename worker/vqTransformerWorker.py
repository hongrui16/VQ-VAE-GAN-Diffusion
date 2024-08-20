# Importing Libraries
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Run, Image
import os
import time
import tqdm

from utils.utils import print_gpu_memory_usage

class VQTransformerWorker:
    def __init__(
        self,
        model: nn.Module,
        run: Run = None,
        experiment_dir: str = "experiments",
        device: str = "cpu",
        logger = None,
        train_dataset = None,
        save_img_dir = None,
        args = None,
        val_dataloader = None,
        config = None,

    ):
        model_name = config['architecture']['model_name']
        dataset_name = config['dataset']['dataset_name']
        img_size = config["dataset"]["img_size"][dataset_name]
        img_channels = config['dataset']['img_channels'][dataset_name]
        batch_size = config['dataset']["batch_size"][model_name][dataset_name]

        self.img_size = img_size
        self.batch_size = batch_size
        self.num_codebook_vectors = config['architecture']['vqvae']['num_codebook_vectors']
        self.seq_len = config['architecture']['vqvae']['latent_channels']

        learning_rate = config['trainer'][model_name]['learning_rate']
        beta1 = config['trainer'][model_name]['beta1']
        beta2 = config['trainer'][model_name]['beta2']

        self.vqTransModel = model
        self.run = run
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.model_name = model_name
        self.save_img_dir = save_img_dir
        self.args = args
        self.val_dataloader = val_dataloader
        self.global_step = 0


        self.vqTransModel.to(device)
        self.device = device

        train_model = config['architecture'][model_name]['train_model']
        if train_model:
            self.optim = self.configure_optimizers(
                learning_rate=learning_rate, beta1=beta1, beta2=beta2
            )


            num_iters_per_epoch = len(train_dataset)//self.batch_size
            self.save_step = 100
            if num_iters_per_epoch < 0.1*self.save_step:
                self.save_step = 1
            elif num_iters_per_epoch < 0.5*self.save_step:
                self.save_step = 5
            elif num_iters_per_epoch < 1.5*self.save_step:
                self.save_step = 10
            elif num_iters_per_epoch < 10*self.save_step:
                self.save_step = 50
            elif num_iters_per_epoch < 50*self.save_step:
                self.save_step = 100
            else:
                self.save_step = 200

            self.logger.info(f"Save step set to {self.save_step}")             


    def configure_optimizers(
        self, learning_rate: float = 4.5e-06, beta1: float = 0.9, beta2: float = 0.95
    ):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        # Enabling weight decay to only certain layers
        for mn, m in self.vqTransModel.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.vqTransModel.transformer.named_parameters()}

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
        self.vqTransModel.train()
        for epoch in range(epochs):
            start_time = time.time()
            tqdm_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for index, imgs in enumerate(tqdm_bar):
                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                logits, targets = self.vqTransModel(imgs)
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

                if self.global_step % self.save_step == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Cross Entropy Loss : {loss:.4f}"
                    )

                    _, sampled_imgs = self.vqTransModel.log_images(imgs[0][None])

                    if self.run is not None:
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
                self.global_step += 1

            if epoch == 0:
                print_gpu_memory_usage(self.logger)
            self.save_checkpoint(self.experiment_dir)

            self.generate_images(epoch=epoch)
            torch.cuda.empty_cache()

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {end_time - start_time:.2f} seconds")

            if self.args.debug:
                break
            
    
    def generate_images(self, n_images: int = 16, epoch = -1, random_indices = False, flag = 'generate'):

        self.logger.info(f"{self.model_name} Generating {n_images} images...")
        
        self.vqTransModel.eval()

        self.vqTransModel = self.vqTransModel.to(self.device)
        with torch.no_grad():
            start_indices = torch.zeros((4, 0)).long().to(self.device)
            sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
            sos_tokens = sos_tokens.long().to(self.device)
            sample_indices = self.vqTransModel.sample(
                start_indices, sos_tokens, steps=256
            )
            sampled_imgs = self.vqTransModel.z_to_image(sample_indices)
            torchvision.utils.save_image(
                sampled_imgs,
                os.path.join(self.save_img_dir, f"{self.model_name}_{flag}_epoch{epoch:03d}.jpg"),
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
        torch.save(self.vqTransModel.transformer.state_dict(), transformer_path)
        self.logger.info(f"Transformer model saved at {checkpoint_dir}")