
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

import argparse





######################## Parameters ######################




# # dataset params :
# L = 128  # length of input images
# data_modality = 'SEG'  # either 'SEG' to use segmentation dataset or 'MRI' for gray-scale MRIs


# # training params
# BATCH_SIZE = 16
# lr = 5e-4
# epochs = 100
# model_name = 'saved_models/seg_model_300.pth'


# # model hyper-params :
# K =  512 # num_embeddings
# D =  64  # embedding_dim 
# downsampling_factor = 8

# use_residual = False # swicth to True if wants to use RQ-VAE
# num_quantizers = 2
# shared_codebook = False

# beta = .25
# decay = .8


def train(): 
    ######################    Parse command-line ################################.

    parser = argparse.ArgumentParser(description="Train a model with specific configurations.")

    # Dataset parameters
    parser.add_argument("--L", type=int, default=128, help="Length of input images")
    parser.add_argument("--data_modality", type=str, choices=['SEG', 'MRI'], default='SEG', help="Data modality: 'SEG' for segmentation dataset, 'MRI' for gray-scale MRIs")

    # Training parameters
    parser.add_argument("--BATCH_SIZE", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default='random.pth', help="Path to save or load the model")

    # Model hyper-parameters
    parser.add_argument("--K", type=int, default=512, help="Number of embeddings")
    parser.add_argument("--D", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--downsampling_factor", type=int, default=8, help="Downsampling factor")

    parser.add_argument("--use_residual", action='store_true', help="Use RQ-VAE if set")
    parser.add_argument("--num_quantizers", type=int, default=2, help="Number of quantizers")
    parser.add_argument("--shared_codebook", action='store_true', help="Use shared codebook if set")

    parser.add_argument("--beta", type=float, default=0.25, help="Beta parameter")
    parser.add_argument("--decay", type=float, default=0.8, help="Decay parameter")
    ## add loss parser , (new)

    # **kwargd arguments
    parser.add_argument("--kwargs", nargs='*', help="Additional key-value pairs (e.g., --kwargs key1=value1 key2=value2)")

    args = parser.parse_args()









    #################### dataset init ######################
    dataset_path = "/home/ids/ihamdaoui-21/ACDC/database"

    train_set_path = os.path.join(dataset_path, "training")
    test_set_path  = os.path.join(dataset_path, "testing")


    train_dataset = load_dataset(train_set_path, modality= args.data_modality)
    test_dataset  = load_dataset(test_set_path, modality= args.data_modality)


    if args.data_modality == 'SEG':
        input_transforms = Compose([
            transforms.Resize(size=(args.L,args.L), interpolation=transforms.InterpolationMode.NEAREST),
            One_hot_Transform(num_classes=4)
            ])
    else : 
        input_transforms = Compose([
            transforms.Resize(size=(args.L,args.L), interpolation=transforms.InterpolationMode.NEAREST),
            PercentileClip(lower_percentile=1, upper_percentile=99),
            MinMaxNormalize(min_value=0.0, max_value=1.0),
            ])


    TrainDataset = ACDC_Dataset(data = train_dataset, transforms= input_transforms) 
    TestDataset  = ACDC_Dataset(data = test_dataset, transforms= input_transforms)

    TrainLoader  = DataLoader(TrainDataset, batch_size = args.BATCH_SIZE, shuffle = True)
    TestLoader   = DataLoader(TestDataset , batch_size = args.BATCH_SIZE, shuffle = False)





    # instanciate model :
    VQ_VAE =  VQVAE(embedding_dim= args.D,
                    num_embeddings= args.K,
                    downsampling_factor= args.downsampling_factor,
                    residual = args.use_residual,
                    num_quantizers = args.num_quantizers,
                    shared_codebook = args.shared_codebook,
                    beta = args.beta,
                    decay = args.decay,
                    data_mod = args.data_modality
                        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VQ_VAE.to(device)

    optimizer = optim.AdamW(model.parameters(), lr= args.lr, weight_decay=1e-4)




    ###################### training loop ######################


    model.train()

    train_loss_values    = []
    commit_loss_values   = []
    val_loss_values      = []


    best_val_loss = float('inf')

    for epoch in range(args.epochs):

        train_loss  = []
        commit_loss = []

        with tqdm(enumerate(TrainLoader), unit="batch", total=len(TrainLoader)) as tepoch:
            for batch_idx, (inputs) in tepoch:
                inputs = inputs.float().to(device)  # Move data to the appropriate device (GPU/CPU)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass // args is a list containing : [output, input, vq_loss]
                output, inputs, indices, commitement_Loss = model(inputs)
                
                # Loss and backward
                all_loss = model.loss_function(output, inputs, indices, commitement_Loss)
                loss = all_loss['loss']  # Use the loss function defined in the model
                recons_loss = all_loss['Reconstruction_Loss']
                commitement_Loss = all_loss['commitement_Loss']

                loss.backward()
                optimizer.step()
                            
                # Track running loss
                train_loss.append( recons_loss.item() )
                commit_loss.append( commitement_Loss.item() )

                # tqdm bar displays the loss
                tepoch.set_postfix(loss=loss.item())

        train_loss_values.append( np.mean(train_loss))
        commit_loss_values.append( np.mean(commit_loss))

        # Validation after each epoch
        val_loss = evaluate_model(model, TestLoader, device)
        val_loss_values.append(val_loss)


        #saving model if Loss values decreases
        if val_loss < best_val_loss :
            save_model(model_name, model, epoch, train_loss_values, val_loss_values, commit_loss_values)
            best_val_loss = val_loss

        print('Epoch {}: '.format(epoch))


    print("Training complete.")



    ################ Evaluae the model #################

    if (args.data_modality == 'SEG'):
        dataset = "Segmentations dataset"
        score = " The DiceScore "
    else : 
        dataset = "MRI images dataset"
        score = " The MSE score "


    print(" -------------------------------------------------------------")
    print("\n This model is trained on the {}".format(dataset))



    print("The model score is : " , score_model(model, TestLoader, device))


    ################ CodeBook usage ###################

    print("\n\n\n -------------------------------------------------------")
    print("codebook_investigation : ")
    hist = codebook_hist_testset(model, TestLoader, device)
    hist = hist/np.sum(hist)
    print(hist)







# if __name__ == "__main__":
#     args = parse_args()
#     print(args)


if __name__ == "__main__":
    train()