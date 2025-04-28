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





from worker.vaeWorker import VAEWorker
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
        config['dataset']["batch_size"][model_name][dataset_name] = 2
        train_split = 'val'
        config['trainer']["num_workers"] = 1        
    else:
        train_split = config['dataset']["train_split"]

    log_dir = config['trainer']["log_dir"]
    ckpt_dir = config['trainer']["ckpt_dir"] if not args.debug else log_dir

    num_epochs = config['trainer']["num_epochs"]

    current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    exp_dir = os.path.join(log_dir, dataset_name, model_name, f'run_{current_timestamp}')
    save_ckpt_dir = os.path.join(ckpt_dir, dataset_name, model_name, f'run_{current_timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(save_ckpt_dir, exist_ok=True)
    # checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    # os.makedirs(checkpoint_dir, exist_ok=True)



    
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
    config_filename = os.path.basename(args.config)
    shutil.copy(args.config, os.path.join(exp_dir, config_filename))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        
        logging.info("Using CPU")


    train_dataloader, train_dataset = load_dataloader(name=dataset_name, split=train_split, 
                                                      logger=logger, config = config)
    val_dataloader, val_dataset = load_dataloader(name=dataset_name, split='val', 
                                                    logger=logger, config = config)    
    logging.info(f"Data loaded")
    
    img_size = config["dataset"]["img_size"][dataset_name]
    in_channels = config['dataset']['img_channels'][dataset_name]

    if model_name in ['vae']:
        # model_name = 'vae'
        # dataset_name = config['dataset']['dataset_name']
        # img_size = config["dataset"]["img_size"][dataset_name]
        # img_channels = config['dataset']['img_channels'][dataset_name]
        # batch_size = config['dataset']["batch_size"][model_name][dataset_name]

        vae_worker = VAEWorker(
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset=train_dataset,

            args=args,
            val_dataloader=val_dataloader,
            config=config,
            save_ckpt_dir=save_ckpt_dir,
        )
        logger.info('Initializing VAE Worker')
        train_model = config['architecture']['vae']['train_model']
        if train_model:
            logger.info(f"Training {model_name} on {device} for {num_epochs} epoch(s).")
            vae_worker.train(
                dataloader=train_dataloader,
                epochs=num_epochs,
            )
        
        if args.debug:
            vae_worker.generate_images(
                n_images=10,
                sample_from_normal_dist = True,

            )

    if model_name.lower() in ['vqgan', 'vqvae']:
        
        vqgan_vqvae_worker = VQGANVQVAEWorker(
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset = train_dataset,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
            save_ckpt_dir=save_ckpt_dir,

        )
        logger.info(f'Initializing {model_name.lower} Worker')
        train_model = config['architecture']['vqvae']['train_model']
        if train_model:
            logger.info(f"Training {model_name} on {device} for {num_epochs} epoch(s).")
            vqgan_vqvae_worker.train(
                dataloader=train_dataloader,
                epochs=num_epochs,
            )

    if model_name.lower() in ['vqvae_transformer', 'vqgan_transformer']:
        
        logging.info(f"{model_name} Transformer models created")

        vqTransformer_worker = VQTransformerWorker(
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset= train_dataset,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
            save_ckpt_dir=save_ckpt_dir,

        )
        logger.info('Initializing Transformer Worker')
        train_model = config['architecture'][model_name]['train_model']
        if train_model:
            logger.info(f"Training {model_name} Transformer on {device} for {num_epochs} epoch(s).")
            vqTransformer_worker.train(dataloader=train_dataloader, epochs=num_epochs)



    if model_name.lower() in ['vqdiffusion']:

        vqdiffusion_worker = VQDiffusionWorker(
            # run=run,
            device=device,
            experiment_dir=exp_dir,
            logger = logger,
            train_dataset= train_dataset,
            args = args,
            val_dataloader=val_dataloader,
            config = config,
            save_ckpt_dir=save_ckpt_dir,

        )
        logger.info('Initializing Diffusion Worker')
        train_model = config['architecture'][model_name]['train_model']
        if train_model:
            logger.info(f"Training {model_name} Diffusion on {device} for {num_epochs} epoch(s).")
            vqdiffusion_worker.train(dataloader=train_dataloader, epochs=num_epochs)


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

        gaussian_diffusion_2d_worker = GaussianDiffusion2DWorker(gaussian_diffusion_2d, device=device, train_dataset=train_dataset,
                                                            experiment_dir  = exp_dir, args = args,
                                                            logger=logger, config = config)
        
        logger.info('Initializing Gaussian Diffusion Worker')
        train_model = config['architecture'][model_name]['train_model']
        if train_model:
            logger.info(f"Training {model_name} on {device} for {num_epochs} epoch(s).")
            gaussian_diffusion_2d_worker.train(dataloader=train_dataloader, epochs=num_epochs)

    
    if 'gaussiandiffusion3d' == model_name.lower():


        timesteps = config['architecture'][model_name]['diffusion_steps']
        

        sampling_timesteps = config['architecture'][model_name]['sampling_steps']
        model_base_dim = config['architecture'][model_name]['model_base_dim']
        gaussian_diffusion_3d = GaussianDiffusion3D(
                                                image_sizes = [img_size,img_size],
                                                timesteps = timesteps,
                                                in_channels = in_channels,
                                                sampling_timesteps = sampling_timesteps,
                                                base_dim= model_base_dim,
                                                dim_mults=[2,4],
                                                device = device,
                                                )
        
        logging.info(f"{model_name} models created")

        gaussian_diffusion_3d_worker = GaussianDiffusion3DWorker(gaussian_diffusion_3d, device=device, train_dataset=train_dataset,
                                                            experiment_dir  = exp_dir, args = args,
                                                            save_img_dir = save_dir,
                                                            logger=logger, config = config)
        
        logger.info(f'Initializing {model_name} Worker')
        train_model = config['architecture'][model_name]['train_model']
        if train_model:
            logger.info(f"Training {model_name} on {device} for {num_epochs} epoch(s).")
            gaussian_diffusion_3d_worker.train(dataloader=train_dataloader, epochs=num_epochs)

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
        default="configs/training_config_large.yml",
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