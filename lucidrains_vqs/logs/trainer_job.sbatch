#!/usr/bin/bash
#SBATCH --job-name=random
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
    --epochs 2\
    --model_name saved_models/seg/random.pth\
    --data_modality 'SEG'\
    --kwargs kmeans_init=True\


