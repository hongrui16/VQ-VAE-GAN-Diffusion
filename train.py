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

from worker.vqganVqvaeWorker import VQGANVQVAEWorker
from worker.vqTransformerWorker import VQTransformerWorker
from worker.vqdiffusionWorker import VQDiffusionWorker




def main(args, config):
    if args.debug:
        config['dataset']["batch_size"] = 1
        train_split = 'val'
    else:
        train_split = config['dataset']["train_split"]

    model_name = config['architecture']["model_name"]
    log_dir = config['trainer']["log_dir"]
    num_epochs = config['trainer']["num_epochs"]


    batch_size = config['dataset']["batch_size"] 
    dataset_name = config['dataset']["dataset_name"]
    num_workers = config['dataset']["num_workers"]

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
    shutil.copy(args.config, os.path.join(exp_dir, "config_3channel.yml"))

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

    train_dataloader, train_dataset = load_dataloader(name=dataset_name, batch_size = batch_size, 
                                                      num_workers = num_workers, split=train_split, 
                                                      logger=logger, config = config)
    val_dataloader, val_dataset = load_dataloader(name=dataset_name, batch_size = batch_size, 
                                                  num_workers = num_workers, split='val', 
                                                    logger=logger, config = config)    
    logging.info(f"Data loaded")

    run = Run(experiment=dataset_name)
    run["hparams"] = config



    
    vqgan_vqvae_worker = VQGANVQVAEWorker(
        model=vqvae,
        # run=run,
        device=device,
        experiment_dir=exp_dir,
        logger = logger,
        train_dataset = train_dataset,
        save_img_dir = save_dir,
        args = args,
        val_dataloader=val_dataloader,
        config = config,
    )
    logger.info(f'Initializing {model_name.lower} Worker')
    train_vqvae = config['architecture']['vqvae']['train_vqvae']
    if train_vqvae:
        logger.info(f"Training {model_name} on {device} for {num_epochs} epoch(s).")
        vqgan_vqvae_worker.train(
            dataloader=train_dataloader,
            epochs=num_epochs,
        )

    
    if 'transformer' in model_name.lower():
        vqTransformer_worker = VQTransformerWorker(
            model=vqgan_transformer,
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset= train_dataset,
            save_img_dir = save_dir,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
        )
        logger.info('Initializing Transformer Worker')
        train_transformer = config['architecture']['transformer']['train_transformer']
        if train_transformer:
            logger.info(f"Training {model_name} Transformer on {device} for {num_epochs} epoch(s).")
            vqTransformer_worker.train(dataloader=train_dataloader, epochs=num_epochs)

    
    if 'diffusion' in model_name.lower():
        vqdiffusion_worker = VQDiffusionWorker(
            model=vqdiffusion,
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset= train_dataset,
            save_img_dir = save_dir,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
        )
        logger.info('Initializing Diffusion Worker')
        train_diffusion = config['architecture']['diffusion']['train_diffusion']
        if train_diffusion:
            logger.info(f"Training {model_name} Diffusion on {device} for {num_epochs} epoch(s).")
            vqdiffusion_worker.train(dataloader=train_dataloader, epochs=num_epochs)


    ### get the gpu memory usage and print it out

    logging.info(f'Memory Usage:{torch.cuda.memory_allocated()}')  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
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
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)




'''
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:A100.40gb:1 --mem=40gb -t 0-24:00:00
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=6 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-12:00:00

'''