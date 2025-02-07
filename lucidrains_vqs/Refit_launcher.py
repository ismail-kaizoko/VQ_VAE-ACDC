############### THIS FILE LAUNCHES THE Refi-FINUTUNING ##############

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
from utils.launcher_utils import *

import argparse
import json



def Refit(): 
    ######################    Parse command-line Args    ######################

    parser = argparse.ArgumentParser(description="Train a model with specific configurations.")

    # Dataset parameters
    parser.add_argument("--L", type=int, default=128, help="Length of input images")

    # Model hyper-parameters
    parser.add_argument("--new_K", type=int, default=128, help="Number of embeddings")
 
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--baseline_model_name", type=str, required = True, help="Path to save or load the model")
    parser.add_argument("--model_name", type=str, required = True, help="Path to save or load the model")

    parser.add_argument("--kwargs", nargs='*', help="Additional key-value pairs (e.g., --kwargs key1=value1 key2=value2)")

    args = parser.parse_args()



    print(f"     THis is a Refit-Finutned version of the model {args.baseline_model_name} with new K= {args.new_K} ")
    print("-" * 100)
    print("\n\n")

    # print_params("not implemented yed TODO")

    ####################### Load previous model infos ####################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    baseline_model_path = args.baseline_model_name
    baseline_model_params = load_model_metadata(baseline_model_path)
    baseline_model = VQVAE(**baseline_model_params).to(device)


    # fetch the previous model encoder and decoder : 

    # Load the saved model checkpoint
    checkpoint = torch.load(baseline_model_path)
    # Filter the encoder parameters
    encoder_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
    # Filter the decoder parameters
    decoder_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('decoder.')}



    ############################################################################
    #################### dataset init ######################
    ############################################################################
    
    data_mod = baseline_model_params['data_mod']
    dataset_path = "/home/ids/ihamdaoui-21/ACDC/database"

    train_set_path = os.path.join(dataset_path, "training")
    test_set_path  = os.path.join(dataset_path, "testing")


    train_dataset = load_dataset(train_set_path, modality= data_mod)
    test_dataset  = load_dataset(test_set_path, modality= data_mod)


    if data_mod == 'SEG':
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

    TrainLoader  = DataLoader(TrainDataset, batch_size = args.batch_size, shuffle = True)
    TestLoader   = DataLoader(TestDataset , batch_size = args.batch_size, shuffle = False)






    #######################  Proceed the The Refit : Finding data centroids ###############


    latent_vectors = []

    # Process the dataset
    with torch.no_grad():  # No need to track gradients
        for batch in TrainLoader:
            # Pass the batch through the encoder
            encoded = baseline_model.encode(batch.float().to(device))[0]  # Output shape: (batch_size, 32, 32, 32)
            
            # Flatten the encoded output to (batch_size, 32*32)
            encoded_flat = encoded.view(encoded.size(0), 64, -1).permute(0, 2, 1)  # Shape: (batch_size, 1024, 64)
            
            # Now flatten across the batch and spatial dimensions to (batch_size * 1024, 64)
            encoded_flat = encoded_flat.reshape(-1, 64)
            
            # Convert the tensor to NumPy and store it
            latent_vectors.append(encoded_flat.cpu().numpy())

    # Concatenate all the latent vectors into a single NumPy array
    latent_vectors = np.concatenate(latent_vectors, axis=0)  # Shape: (size_of_dataset, 32*32)

    # # Optionally, save the latent vectors to disk
    # np.save('latent_vectors.npy', latent_vectors)
    from sklearn.cluster import kmeans_plusplus

    # Calculate seeds from k-means++
    centers_init, indices = kmeans_plusplus(latent_vectors, n_clusters= args.new_K)

    new_codebook = torch.from_numpy(centers_init)






    ################# instanciate the new model (shrinked) ##################
    model_params = baseline_model_params.copy()
    model_params['num_embeddings'] = args.new_K


    model = VQVAE(**model_params).to(device)
    model.vq_layer.codebook = new_codebook


    # Load the encoder and decoder weights into the new model
    # Remove the 'encoder.' prefix from all keys in encoder_state_dict
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items()}
    model.encoder.load_state_dict(encoder_state_dict)

    # Remove the 'encoder.' prefix from all keys in encoder_state_dict
    decoder_state_dict = {k.replace('decoder.', ''): v for k, v in decoder_state_dict.items()}
    model.decoder.load_state_dict(decoder_state_dict)

    #model ready to train !!

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


    if (data_mod == 'SEG'):
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


    save_training_metadata_Refit(args, model_params, best_epoch, score, percentage)



if __name__ == "__main__":
    Refit()