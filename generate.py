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


from network.vqvae.vqvae import VQVAE
from network.vqTransformer.vqTransformer import VQTransformer
from network.vqDiffusion.vqDiffusion import VQDiffusion

from worker.vqganVqvaeWorker import VQGANVQVAEWorker
from worker.vqTransformerWorker import VQTransformerWorker
from worker.vqdiffusionWorker import VQDiffusionWorker
from worker.gaussianDiffusion2DWorker import GaussianDiffusion2DWorker
from worker.gaussianDiffusion3DWorker import GaussianDiffusion3DWorker

from network.vqDiffusion.submodule.diffusion_gaussian2d import GaussianDiffusion2D
from network.vqDiffusion.submodule.unet2d import Unet2D
from network.vqDiffusion.submodule.diffusion_gaussian3d import GaussianDiffusion3D


def main(args, config):
    model_name = config['architecture']["model_name"]
    dataset_name = config['dataset']["dataset_name"]

    if args.debug:
        config['dataset']["batch_size"][model_name][dataset_name] = 1
        config['trainer']["num_workers"] = 1

    img_size = config["dataset"]["img_size"][dataset_name]
    in_channels = config['dataset']['img_channels'][dataset_name]
    batch_size = config['dataset']["batch_size"][model_name][dataset_name]

    resume_path = config['architecture'][model_name]['resume_path']
    base_name = os.path.basename(resume_path)
    log_dir = resume_path[:resume_path.find(base_name)]

    current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    exp_dir = os.path.join(log_dir, 'inference', f'run_{current_timestamp}')
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

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        
        logging.info("Using CPU")


    if model_name.lower() in ['vqgan', 'vqvae', 'vqvae_transformer', 'vqgan_transformer', 'vqdiffusion']:
        vqvae = VQVAE(logger= logger, config = config)
        logging.info(f"VQVAE model created")
            
        vqgan_vqvae_worker = VQGANVQVAEWorker(
            model=vqvae,
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            save_img_dir = save_dir,
            args = args,
            config = config,
        )
        logger.info(f'Initializing {model_name.lower} Worker')
        if model_name.lower() in ['vqgan', 'vqvae']:
            logger.info(f"{model_name} generates images.")
            vqgan_vqvae_worker.generate_images()

    if 'transformer' in model_name.lower():
        vqgan_transformer = VQTransformer(
            vqvae, 
            config = config, 
            device=device,
            logger = logger,
        )
        logging.info(f"{model_name} Transformer models created")

        vqTransformer_worker = VQTransformerWorker(
            model=vqgan_transformer,
            # run=run,
            device=device,
            logger = logger,
            save_img_dir = save_dir,
            args = args,
            config = config,
        )
        logger.info('Initializing Transformer Worker')

        logger.info(f"{model_name} generates images.")
        vqTransformer_worker.generate_images()



    if 'vqdiffusion' in model_name.lower():
        vqdiffusion = VQDiffusion(vqvae, device=device, 
                                    logger=logger, config = config)
        logging.info(f"{model_name} models created")

        vqdiffusion_worker = VQDiffusionWorker(
            model=vqdiffusion,
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            save_img_dir = save_dir,
            args = args,
            config = config,
        )
        logger.info('Initializing Diffusion Worker')
        logger.info(f"{model_name} generates images.")
        vqdiffusion_worker.generate_images()


    if 'gaussiandiffusion2d' == model_name.lower():
        unet2d = Unet2D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = img_size,
            out_dim= img_size )
        time_steps = config['architecture'][model_name]['diffusion_steps']
        sampling_timesteps = config['architecture'][model_name]['sampling_steps']
        gaussian_diffusion_2d = GaussianDiffusion2D(unet2d, timesteps = time_steps,
                                                sampling_timesteps = sampling_timesteps,
                                                diffusion_type=model_name)
        
        logging.info(f"{model_name} models created")

        gaussian_diffusion_2d_worker = GaussianDiffusion2DWorker(gaussian_diffusion_2d, 
                                                                 device=device,
                                                                experiment_dir  = exp_dir, 
                                                                args = args,
                                                                save_img_dir = save_dir,
                                                                logger=logger, 
                                                                config = config)
        
        logger.info('Initializing Gaussian Diffusion Worker')
        logger.info(f"{model_name} generates images.")
        gaussian_diffusion_2d_worker.generate_images()

    
    if 'gaussiandiffusion3d' == model_name.lower():

        timesteps = config['architecture'][model_name]['diffusion_steps']
        sampling_timesteps = config['architecture'][model_name]['sampling_steps']
        model_base_dim = config['architecture'][model_name]['model_base_dim']
        gaussian_diffusion_3d = GaussianDiffusion3D(
                                                image_sizes = [img_size,img_size],
                                                timesteps = timesteps,
                                                in_channels = in_channels,
                                                sampling_timesteps = sampling_timesteps,
                                                base_dim = model_base_dim,
                                                dim_mults = [2,4],
                                                device = device,
                                                )
        
        logging.info(f"{model_name} models created")

        gaussian_diffusion_3d_worker = GaussianDiffusion3DWorker(gaussian_diffusion_3d, 
                                                                 device=device, 
                                                                experiment_dir  = exp_dir, 
                                                                args = args,
                                                                save_img_dir = save_dir,
                                                                logger = logger, 
                                                                config = config)
        
        logger.info(f'Initializing {model_name} Worker')
        logger.info(f"{model_name} generates images.")
        gaussian_diffusion_3d_worker.generate_images()

    '''
    run = Run(experiment=dataset_name)
    run["hparams"] = config
    '''

    
    ### get the gpu memory usage and print it out

    logging.info(f'Memory Usage:{torch.cuda.memory_allocated()}')  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config_small.yml",
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