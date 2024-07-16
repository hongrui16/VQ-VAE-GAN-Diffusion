# Importing Libraries
import argparse

import yaml
from aim import Run
from datetime import datetime
import os
import torch
import sys
import logging

from dataloader.build_dataloader import load_dataloader

from network.vqganTransformer.vqganTransformer import VQGANTransformer
from trainer.vqganTrainer import VQGANTrainer
from trainer.vqganTransformerTrainer import VAGANTransformerTrainer

from network.vqgan.vqgan import VQGAN


def main(args, config):

    current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    exp_dir = os.path.join(args.log_dir, args.dataset_name, args.model_name, f'run_{current_timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    # checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    # os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = os.path.join(exp_dir, "generated_images")
    os.makedirs(save_dir, exist_ok=True)

    
    log_path = os.path.join(exp_dir, "info.log")
    
    # Log to file & tensorboard writer
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger(f'{args.model_name}-LOG')
    logging.info(f"Logging to {exp_dir}")

    ## print all the args into log file
    logging.info(f"<<<<<<<<<<<<<<<<***************hyperparameters***********************************************")
    kwargs = vars(args)
    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")
    logging.info(f"<<<<<<<<<<<<<<<<***************hyperparameters***********************************************")

    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        device = args.device
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device("cpu")
        device = args.device
        logging.info("Using CPU")

    vqgan = VQGAN(**config["architecture"]["vqgan"])
    logging.info(f"{args.model_name} model created")

    if args.train_transformer:
        vqgan_transformer = VQGANTransformer(
            vqgan, **config["architecture"]["transformer"], device=device
        )
        logging.info(f"{args.model_name} Transformer models created")

    train_dataloader, train_dataset = load_dataloader(name=args.dataset_name, batch_size = args.batch_size, split='train', logger=logger)
    val_dataloader, val_dataset = load_dataloader(name=args.dataset_name,  batch_size = args.batch_size, split='val', logger=logger)
    logging.info(f"Data loaded")

    run = Run(experiment=args.dataset_name)
    run["hparams"] = config


    # trainer = Trainer(
    #     vqgan,
    #     vqgan_transformer,
    #     run=run,
    #     config=config["trainer"],
    #     seed=args.seed,
    #     device=args.device,
    #     experiment_dir=exp_dir,
    #     logger = logger,
    #     train_dataset = train_dataset,
    #     model_name = args.model_name
    # )
    # logging.info(f"Trainer created")

    # logging.info(f"Training VQGAN")
    # trainer.train_vqgan(train_dataloader, epochs=args.num_epochs)

    # logging.info(f"Generating images using VQGAN")
    # trainer.vqgan_generate_images(dataloader = val_dataloader, num_images = 10)



    vqgan_trainer = VQGANTrainer(
        model=vqgan,
        run=run,
        device=device,
        experiment_dir=exp_dir,
        logger = logger,
        model_name = args.model_name,
        train_dataset = train_dataset,
        save_img_dir = save_dir,
        args = args,
        val_dataloader=val_dataloader,
        **config['trainer']["vqgan"],
    )

    logger.info(f"Training {args.model_name} on {device} for {args.num_epochs} epoch(s).")
    vqgan_trainer.train(
        dataloader=train_dataloader,
        epochs=args.num_epochs,
    )


    # logging.info(f"Generating images using {args.model_name}")
    # vqgan_trainer.vqgan_generate_images(dataloader = val_dataloader)

    if args.train_transformer:
        vqganTransformer_trainer = VAGANTransformerTrainer(
            model=vqgan_transformer,
            run=run,
            device=device,
            experiment_dir=exp_dir,
            model_name = args.model_name,
            logger = logger,
            save_img_dir = save_dir,
            args = args,
            val_dataloader=val_dataloader,
            **config['trainer']["transformer"],
        )

        logger.info(f"Training {args.model_name} Transformer on {device} for {args.num_epochs} epoch(s).")
        vqganTransformer_trainer.train(dataloader=train_dataloader, epochs=args.num_epochs)


    # logging.info(f"Generating images using {args.model_name} Transformer")
    # vqganTransformer_trainer.vqganTrans_generate_images()

    ### get the gpu memory usage and print it out

    logging.info(f'Memory Usage:{torch.cuda.memory_allocated()}')  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.yml",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mnist",
        help="Dataset for the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to train the model on",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Seed for Reproducibility",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="path to save log files",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size",
        )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='number of epochs to train'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='vqgan',
        help='input model name, vqgan, vqvae, vqdiffusion'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode')

    parser.add_argument(
        '--resume_ckpt_dir',
        type=str,
        default=None,
        help='path to the checkpoint to resume training')
    
    parser.add_argument(
        '--no_train_transformer',
        action='store_false',
        dest='train_transformer',
        help='Do not train the transformer (default is to train)'
    )


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