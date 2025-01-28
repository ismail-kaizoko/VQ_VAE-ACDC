#!/bin/bash

#SBATCH --job-name= (name)
#SBATCH --partition=gpu            # Use the GPU partition

#SBATCH --time=01:00:00            # Time limit
#SBATCH --output=logs/%x_%j.out    # Stdout logs
#SBATCH --error=logs/%x_%j.err     # Stderr logs



# source /home/ids/ihamdaoui-21/anaconda3/etc/profile.d/conda.h
# conda activate torch

module load conda
conda activate torch


# Run the training script with specific arguments
srun python train.py 
