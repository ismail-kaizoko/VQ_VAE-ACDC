############################################################################
#######################  import necessary libraries  #######################
############################################################################


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
import json
from datetime import datetime




############################################################################
######################## Parameters ######################
############################################################################


def print_arguments(args, parser):
    """
    Prints all arguments used in the script and highlights whether default values were used.
    
    Args:
        args: The parsed arguments (from `parser.parse_args()`).
        parser: The `argparse.ArgumentParser` object.
    """
    print("Arguments used in this script:")
    print("\n" )
    
    # Get the default values from the parser
    defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
    
    # Iterate through the arguments and print their values
    for arg_name, arg_value in vars(args).items():
        if arg_name in defaults:
            default_value = defaults[arg_name]
            if arg_value == default_value:
                # Highlight if the default value was used
                print(f"{arg_name}: {arg_value} (default)")
            else:
                # Highlight if the user provided a value
                print(f"{arg_name}: {arg_value} (user-provided)")
        else:
            # Handle cases where the argument doesn't have a default (e.g., --kwargs)
            print(f"{arg_name}: {arg_value}")

    print("-" * 50)



def save_training_metadata(args, best_epoch, score, percentage):
    """
    Save all the parameters, hyper-parameters, and codebook usage in a JSON file.

    Args:
        args: The parsed command-line arguments (from argparse).
        model: The deep learning model.
        codebook_usage: A dictionary or list containing codebook usage information (optional).
    """


    # Get the current timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = (args.model_name).replace('.pth', '.json')

        # Parse kwargs arguments into a dictionary
    kwargs_dict = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            if '=' in kwarg:
                key, value = kwarg.split('=', 1)
                kwargs_dict[key] = value


    # Prepare the metadata dictionary
    metadata = {
        "model_parameters": {
            "embedding_dim": args.D,
            "num_embeddings": args.K,
            "downsampling_factor": args.downsampling_factor,
            "residual": args.use_residual,
            "num_quantizers": args.num_quantizers,
            "shared_codebook": args.shared_codebook,
            "beta": args.beta,
            "decay": args.decay,
            "data_modality": args.data_modality,
        },
        "kwargs_arguments": kwargs_dict,  # Store additional kwargs arguments
        "training_parameters": {
            "batch_size": args.BATCH_SIZE,
            "epochs":args.epochs,
            "loss_function": args.loss_func,
        },
        "evaluation": {
            "best_epoch" : best_epoch, 
            "score" : score,
            "codebook_usage" : percentage,
        },
        "timestamp": timestamp,
    }

    # Save the metadata to a JSON file
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"( Training metadata saved to {filename} )")


def train(): 
    ######################    Parse command-line ################################.

    parser = argparse.ArgumentParser(description="Train a model with specific configurations.")

    # Dataset parameters
    parser.add_argument("--L", type=int, default=128, help="Length of input images")
    parser.add_argument("--data_modality", type=str, choices=['SEG', 'MRI'], required = True, help="Data modality: 'SEG' for segmentation dataset, 'MRI' for gray-scale MRIs")

    # Training parameters
    parser.add_argument("--BATCH_SIZE", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, required = True, help="Path to save or load the model")
    parser.add_argument("--loss_func",type= str, default=None, help="Loss function to use")

    
    # Model hyper-parameters
    parser.add_argument("--K", type=int, default=512, help="Number of embeddings")
    parser.add_argument("--D", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--downsampling_factor", type=int, default=8, help="Downsampling factor")

    parser.add_argument("--use_residual", action='store_true', help="Use RQ-VAE if set to true")
    parser.add_argument("--num_quantizers", type=int, default=2, help="Number of quantizers")
    parser.add_argument("--shared_codebook", action='store_true', help="Use shared codebook if set")

    parser.add_argument("--beta", type=float, default=0.25, help="Beta parameter")
    parser.add_argument("--decay", type=float, default=0.8, help="Decay parameter")
    # **kwargd arguments
    parser.add_argument("--kwargs", nargs='*', help="Additional key-value pairs (e.g., --kwargs key1=value1 key2=value2)")

    args = parser.parse_args()



    # Print arguments with default highlighting
    print_arguments(args, parser)




    ############################################################################
    #################### dataset init ######################
    ############################################################################
    
    
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



    ############################################################################
    ###################### training loop ######################
    ############################################################################

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
            save_model(args.model_name, model, epoch, train_loss_values, val_loss_values, commit_loss_values, val_loss)
            best_val_loss = val_loss


    print("\n\n")
    print("Training is complete with no errors")
    print("\n\n")



    ############################################################################
    ################ Evaluae the model #################
    ############################################################################



    if (args.data_modality == 'SEG'):
        dataset = "Segmentations dataset"
        type_score = " % in The DiceScore "
    else : 
        dataset = "MRI images dataset"
        type_score = " in The MSE score "


    print("-" * 50)
    print("\n\n")

    print("This model is trained on the {}".format(dataset))

    # loading the best model weights : 

    model.load_state_dict(torch.load(args.model_name)['model_state_dict'])
    model = model.to(device)

    best_epoch = torch.load(args.model_name)['epoch']


    score = score_model(model, TestLoader, device)
    print("It has scored  : " , score, type_score)
    print("\n\n.")
    print("-" * 50)


    ############################################################################
    ################ CodeBook usage ###################
    ############################################################################

    print("\n\n")

    print("codebook_investigation : ")


    hist, percentage = codebook_hist_testset(model, TestLoader, device)
    # print(hist)
    print("\n\n")
    print("-" * 50)


    save_training_metadata(args, best_epoch, score, percentage)





if __name__ == "__main__":
    train()