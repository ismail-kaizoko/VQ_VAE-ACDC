
########## import necessary libraries ################

import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

import torch.optim as optim


# Data preprocessing utils : 
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import DataLoader

# Visuals utils
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# my defined model
from utils.acdc_dataset import *
from utils.funcs import *
from utils.vqvae import *





######################## Parameters ######################

# dataset params :
L = 128  # length of input images
data_modality = 'SEG'  # either 'SEG' to use segmentation dataset or 'MRI' for gray-scale MRIs


# training params
BATCH_SIZE = 16
lr = 5e-4
epochs = 100
model_name = 'saved_models/seg_model_300.pth'


# model hyper-params :
K =  512 # num_embeddings
D =  64  # embedding_dim 
downsampling_factor = 8

use_residual = False # swicth to True if wants to use RQ-VAE
num_quantizers = 2
shared_codebook = False

beta = .25
decay = .8





#################### dataset init ######################
dataset_path = "/home/ids/ihamdaoui-21/ACDC/database"

train_set_path = os.path.join(dataset_path, "training")
test_set_path  = os.path.join(dataset_path, "testing")



# instanciate model :
VQ_VAE =  VQVAE(embedding_dim= D,
                num_embeddings= K,
                downsampling_factor= downsampling_factor,
                residual = use_residual,
                num_quantizers = num_quantizers,
                shared_codebook = shared_codebook,
                beta = beta,
                decay = decay,
                data_mod = data_mod
                    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ACDC_RQVAE.to(device)

