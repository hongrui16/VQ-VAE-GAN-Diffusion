#!/bin/bash
#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=vqvaeInterHand26M                   # vqgan, vqvae, vqdiffusion
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/pytorch-vqgan/log/%u/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/pytorch-vqgan/log/%u/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6                # number of cores needed
#SBATCH --gres=gpu:A100.80gb:1                # up to 8; only request what you need # gpu:A100.80gb:1 ; gpu:3g.40gb:1 
#SBATCH --mem=80gb                # 100gb for InterHand26M, memory per CORE; total memory is 1 TB (1,000,000 MB), 
#SBATCH --export=ALL 
#SBATCH --time=2-24:00:00                   # set to 24hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10                           
module load python

source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/pytorch-vqgan

# python train.py --dataset_name InterHand26M --batch_size 16 --num_epochs 10 --config_path configs/config_3channel.yml
# python train.py --dataset_name Oxford102Flower --batch_size 24 --num_epochs 50 --config_path configs/config_3channel.yml --model_name vqvae --resume_ckpt_dir log/Oxford102Flower/vqvae/run_2024-07-15-14-47-33
# python train.py --dataset_name Oxford102Flower --batch_size 24 --num_epochs 50 --config_path configs/config_3channel.yml --model_name vqgan --resume_ckpt_dir log/Oxford102Flower/vqgan/run_2024-07-15-14-51-46

python train.py --dataset_name InterHand26M --batch_size 20 --num_epochs 50 --config_path configs/config_3channel.yml --model_name vqvae --no_train_transformer

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=15 --gres=gpu:3g.40gb:1 --mem=50gb -t 0-24:00:00


# cd /home/rhong5/research_pro/hand_modeling_pro/pytorch-vqgan/dataloader
# python InterHand26M_loader.py