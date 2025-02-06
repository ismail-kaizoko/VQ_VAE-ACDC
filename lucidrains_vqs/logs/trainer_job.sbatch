#!/usr/bin/bash
#SBATCH --job-name=500
#SBATCH --gres=gpu  # Request 1 GPU
#SBATCH --partition=P100            # Use the GPU partition
#SBATCH --time=01:00:00            # Time limit
#SBATCH --output=logs/%x.out    # Stdout logs
#SBATCH --error=logs/%x.err     # Stderr logs



# source /home/ids/ihamdaoui-21/anaconda3/etc/profile.d/conda.h
# conda activate torch

# module load conda
source /home/ids/ihamdaoui-21/anaconda3/etc/profile.d/conda.sh
conda activate torch


# Run the training script with specific arguments
srun python /home/ids/ihamdaoui-21/vq_vae_project/VQ_VAE-ACDC/lucidrains_vqs/train_launcher.py \
    --K 512\
    --D 64\
    --downsampling_factor 8\
    --epochs 100\
    --model_name saved_models/seg_kwargs/500.pth\
    --data_mod 'SEG'\
    --kwargs kmeans_init=True\


