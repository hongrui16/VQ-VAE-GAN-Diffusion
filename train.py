# Importing Libraries
import argparse

import yaml
from aim import Run
from datetime import datetime
import os
import torch
import sys
import logging
import shutil

from dataloader.build_dataloader import load_dataloader


from network.vqgan.vqvae import VQVAE
from network.vqTransformer.vqTransformer import VQTransformer
from network.vqDiffusion.vqDiffusion import VQDiffusion

from trainer.vqganVqvaeTrainer import VQGANVQVAETrainer
from trainer.vqTransformerTrainer import VQTransformerTrainer
from trainer.vqdiffusionTrainer import VQDiffusionTrainer




def main(args, config):

    model_name = config['architecture']["model_name"]
    log_dir = config['trainer']["log_dir"]
    dataset_name = config['trainer']["dataset_name"]
    num_epochs = config['trainer']["num_epochs"]
    batch_size = config['trainer']["batch_size"] if not args.debug else 2

    current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    exp_dir = os.path.join(log_dir, dataset_name, model_name, f'run_{current_timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    # checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    # os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = os.path.join(exp_dir, "generated_images")
    os.makedirs(save_dir, exist_ok=True)

    
    log_path = os.path.join(exp_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger(f'{model_name}-LOG')
    logging.info(f"Logging to {exp_dir}")

    ## print all the args into log file
    # logging.info(f"<<<<<<<<<<<<<<<<***************hyperparameters***********************************************")
    # kwargs = vars(args)
    # for key, value in kwargs.items():
    #     logger.info(f"{key}: {value}")
    # logging.info(f"<<<<<<<<<<<<<<<<***************hyperparameters***********************************************")
    shutil.copy(args.config_path, os.path.join(exp_dir, "config_3channel.yml"))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        
        logging.info("Using CPU")

    vqvae = VQVAE(logger= logger, config = config)
    logging.info(f"VQVAE model created")


    if 'transformer' in model_name.lower():
        vqgan_transformer = VQTransformer(
            vqvae, config = config, device=device
        )
        logging.info(f"{model_name} Transformer models created")

    if 'diffusion' in model_name.lower():
        vqdiffusion = VQDiffusion(vqvae, device=device, 
                                    logger=logger, config = config)
        logging.info(f"{model_name} Diffusion models created")

    train_dataloader, train_dataset = load_dataloader(name=dataset_name, batch_size = batch_size, split='train', logger=logger)
    val_dataloader, val_dataset = load_dataloader(name=dataset_name, batch_size = batch_size, split='val', logger=logger)
    logging.info(f"Data loaded")

    run = Run(experiment=dataset_name)
    run["hparams"] = config



    train_vqvae = config['architecture']['vqvae']['train_vqvae']
    if train_vqvae:
        vqgan_vqvae_trainer = VQGANVQVAETrainer(
            model=vqvae,
            run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset = train_dataset,
            save_img_dir = save_dir,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
        )

        logger.info(f"Training {model_name} on {device} for {num_epochs} epoch(s).")
        vqgan_vqvae_trainer.train(
            dataloader=train_dataloader,
            epochs=num_epochs,
        )

    train_transformer = config['architecture']['transformer']['train_transformer']
    if train_transformer and 'transformer' in model_name.lower():
        vqTransformer_trainer = VQTransformerTrainer(
            model=vqgan_transformer,
            run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            save_img_dir = save_dir,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
        )

        logger.info(f"Training {model_name} Transformer on {device} for {num_epochs} epoch(s).")
        vqTransformer_trainer.train(dataloader=train_dataloader, epochs=num_epochs)

    train_diffusion = config['architecture']['diffusion']['train_diffusion']
    if train_diffusion and 'diffusion' in model_name.lower():
        vqdiffusion_trainer = VQDiffusionTrainer(
            model=vqdiffusion,
            run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            save_img_dir = save_dir,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
        )

        logger.info(f"Training {model_name} Diffusion on {device} for {num_epochs} epoch(s).")
        vqdiffusion_trainer.train(dataloader=train_dataloader, epochs=num_epochs)


    ### get the gpu memory usage and print it out

    logging.info(f'Memory Usage:{torch.cuda.memory_allocated()}')  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config_3channel.yml",
        help="path to config file",
    )
    
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Seed for Reproducibility",
    )

    parser.add_argument(
    '--debug', 
    action='store_true', 
    help='Enable debug mode')


    # parser.add_argument(
    #     "--log_dir",
    #     type=str,
    #     default="log",
    #     help="path to save log files",
    # )



    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=2,
    #     help="batch size",
    #     )
    
    # parser.add_argument(
    #     '--num_epochs',
    #     type=int,
    #     default=1,
    #     help='number of epochs to train'
    # )



    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default="mnist",
    #     help="Dataset for the model",
    # )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cuda",
    #     choices=["cpu", "cuda"],
    #     help="Device to train the model on",
    # )
    

    

    # parser.add_argument(
    #     '--model_name',
    #     type=str,
    #     help='input model name, vqgan, vqvae, vqdiffusion'
    # )
    
    

    # parser.add_argument(
    #     '--resume_ckpt_dir',
    #     type=str,
    #     default=None,
    #     help='path to the checkpoint to resume training')
    
    # parser.add_argument(
    #     '--no_train_transformer',
    #     action='store_false',
    #     dest='train_transformer',
    #     help='Do not train the transformer (default is to train)'
    # )


    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)




'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:A100.40gb:1 --mem=40gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-12:00:00

'''