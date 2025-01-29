import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os
import numpy as np
from matplotlib.colors import ListedColormap





def reconstruct_logits(batch, model):
    output, _, _, _ = model(batch.float())
    return output



def visualize_batch(batch, title):
    batch_size = batch.shape[0]
    samples = 8


    fig, axes = plt.subplots(samples, 4, figsize=(5, 10))  # Adjust figsize to accommodate more rows
    fig.suptitle(title, fontsize=16)

    for ax in axes.flat:
        ax.set_axis_off()

    for i in range(samples):

        img = batch[i]
        axes[i,0].imshow(img[0,:,:], cmap = 'gray', vmin=0, vmax=1)
        axes[i,1].imshow(img[1,:,:], cmap = 'gray', vmin=0, vmax=1)
        axes[i,2].imshow(img[2,:,:], cmap = 'gray', vmin=0, vmax=1)
        axes[i,3].imshow(img[3,:,:], cmap = 'gray', vmin=0, vmax=1)
        # axes[i].axis('off')

    
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




def visualize_batch_logits(batch, title):
    batch_size = batch.shape[0]
    samples = 8


    fig, axes = plt.subplots(samples, 4, figsize=(5, 10))  # Adjust figsize to accommodate more rows
    fig.suptitle(title, fontsize=16)

    for ax in axes.flat:
        ax.set_axis_off()

    for i in range(samples):

        img = batch[i]
        axes[i,0].imshow(img[0,:,:], cmap = 'gray')
        axes[i,1].imshow(img[1,:,:], cmap = 'gray')
        axes[i,2].imshow(img[2,:,:], cmap = 'gray')
        axes[i,3].imshow(img[3,:,:], cmap = 'gray')
        # axes[i].axis('off')

    
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def visualize_errors(true_seg, pred_seg, title):
    # batch_size = batch.shape[0]
    samples = 8

    custom_colors = [
    '#000000', '#ff0000', '#00ff00', '#0000ff']
    cmap = ListedColormap(custom_colors)
    # error_cmap = LinearSegmentedColormap.from_list('black_red', ['black', 'red'], N=256)

    error_mask = torch.where(true_seg != pred_seg, 1, 0)

    fig, axes = plt.subplots(samples, 3, figsize=(10, 20))  # Adjust figsize to accommodate more rows
    fig.suptitle(title, fontsize=16)


    for i in range(samples):
        axes[i,0].imshow(true_seg[i], cmap = cmap)
        axes[i,0].axis('off')

        axes[i,1].imshow(pred_seg[i], cmap = cmap)
        axes[i,1].axis('off')

        axes[i,2].imshow(error_mask[i], cmap = 'magma')
        axes[i,2].axis('off')

    row_titles = ['Ground truth', 'Vq-Vae predictions', 'Pixel Errors']
    for i in range(3):
        axes[0, i].set_title(row_titles[i], fontsize=14, fontweight='bold')
    
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




###################### loss and score function ###############################


def dice_score(targets, preds, smooth=1e-6, logits = True):
    """
    Calculate Dice Loss across the 4 segmentation channels using binary masks.
    :param preds: output tensor of shape [batch_size, 4, height, width] (logits or binary)
    :param targets: Ground truth one-hot tensor of shape [batch_size, 4, height, width]
    :param smooth: A small value to avoid division by zero
    :param Score : return the Dice Score if True, DiceLoss otherwise
    """
    # Apply softmax over channel dimension (4 channels) to convert logits to probabilities
    

    # Convert probabilities to binary one-hot predictions by using argmax and one-hot encoding
    if logits : 
        preds = F.softmax(preds, dim=1)
        preds_bin = torch.argmax(preds, dim=1)  # Shape: [batch_size, height, width] (class index for each pixel)
        preds_onehot = F.one_hot(preds_bin, num_classes=4).permute(0, 3, 1, 2).float()  # Shape: [batch_size, 4, height, width]
        preds = preds_onehot

    # Flatten predictions and targets for dice coefficient calculation
    preds_flat = preds_onehot.contiguous().view(preds_onehot.shape[0], preds_onehot.shape[1], -1)  # [batch_size, 4, height*width]
    targets_flat = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)  # [batch_size, 4, height*width]

    # Calculate intersection and union
    intersection = (preds_flat * targets_flat).sum(dim=2)  # Summing over height and width dimensions
    union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)  # Sum of both sets

    # Calculate Dice coefficient and Dice loss
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)  # Dice coefficient per channel
    dice_loss = dice_coeff.mean()  # Average over the batch and channels

    return dice_loss

def dice_loss(targets, preds, smooth=1e-6, logits = True):
    score = dice_score(targets, preds, smooth , logits)
    return 1 - score




###################### evaluation function (depending on model and data used) ##############

def evaluate_model(model, val_loader, device):
    data_mod = model.data_mod

    if data_mod == 'SEG' :
        return evaluate_model_with_DiceLoss(model, val_loader, device)
    elif data_mod == 'MRI':
        return evaluate_model_with_mse(model, val_loader, device)
    else : 
        raise Exception('wtf')

def score_model(model, val_loader, device):
    data_mod = model.data_mod

    if data_mod == 'SEG' :
        return evaluate_model_with_DiceScore(model, val_loader, device)
    elif data_mod == 'MRI':
        return evaluate_model_with_mse(model, val_loader, device)
    else : 
        raise Exception('wtf')




def evaluate_model_with_mse(model, val_loader, device):
    model.eval()

    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch.float().to(device)
           
            outputs, _, _, _ = model(inputs)
            
            # Loss and backward
            loss = F.mse_loss(inputs, outputs)
            
            val_loss.append(loss.item() )

    avg_val_loss = np.mean(np.array(val_loss))

    return avg_val_loss


def evaluate_model_with_DiceScore(model, val_loader, device):
    model.eval()

    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch.float().to(device)
           
            outputs, _, _, _ = model(inputs)
            
            # Loss and backward
            loss = dice_score(inputs, outputs)
            
            val_loss.append(loss.item() )

    avg_val_loss = np.mean(np.array(val_loss))

    return avg_val_loss

def evaluate_model_with_DiceLoss(model, val_loader, device):
    model.eval()

    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch.float().to(device)
           
            outputs, _, _, _ = model(inputs)
            
            # Loss and backward
            loss = dice_loss(inputs, outputs)
            
            val_loss.append(loss.item() )

    avg_val_loss = np.mean(np.array(val_loss))

    return avg_val_loss


######################## save funcs #########################


def save_model(model_name, model, epoch, train_loss_values, val_loss_values, codebook_loss_values):
    if model.residual:
        save_RQ_model(model_name, model, epoch, train_loss_values, val_loss_values, codebook_loss_values)
    else :
        save_model_standard(model_name, model, epoch, train_loss_values, val_loss_values, codebook_loss_values)


def save_model_standard(model_name, model, epoch, train_loss_values, val_loss_values, codebook_loss_values):
    checkpoint_path = os.path.join( os.getcwd() , model_name )
    torch.save({'epoch' : epoch,
                'K' : model.vq_layer.codebook_size,
                'D' :  model.vq_layer.dim,
                'model_state_dict' : model.state_dict(),
                'train_loss_values' : train_loss_values, 
                'val_loss_values' : val_loss_values, 
                'codebook_loss_values' : codebook_loss_values,
                'codebook' : model.vq_layer.codebook }, checkpoint_path)


def save_RQ_model(model_name, model, epoch, train_loss_values, val_loss_values, codebook_loss_values):
    checkpoint_path = os.path.join( os.getcwd() , model_name )
    torch.save({'epoch' : epoch,
                'K' : model.vq_layer.codebook_sizes[0],
                'D' :  model.vq_layer.layers[0].dim,
                'model_state_dict' : model.state_dict(),
                'train_loss_values' : train_loss_values, 
                'val_loss_values' : val_loss_values, 
                'codebook_loss_values' : codebook_loss_values,
                'codebook' : model.vq_layer.codebooks }, checkpoint_path)





###################### plot funcs ########################

def plot_train_val_loss(train_loss_values, val_loss_values ):
    # Plot the training and validation losses
    plt.figure(figsize=(15, 10))
    plt.plot(train_loss_values, label='Train Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Evolution of Loss')
    plt.legend()
    plt.grid()
    plt.show()


def plot_rc_loss(train_loss_values, codebook_loss_values, beta):
    recons_loss_values = np.array(train_loss_values) - ( (1+beta)*np.array(codebook_loss_values))
    # Plot the training and validation losses
    plt.figure(figsize=(20, 10))
    # plt.plot(train_loss_values, label='Train Loss')
    # plt.plot(val_loss_values, label='Validation Loss')
    plt.plot(codebook_loss_values, label = "CodeBook Loss")
    # plt.plot(commit_loss_values, label = "Committement Loss")
    plt.plot(recons_loss_values, label = "recons Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Evolution of ELoss')
    plt.legend()
    plt.grid()
    plt.show()



###########################   Printing Functions #################

def print_arguments(args, defaults):
    """
    Print all arguments with their values and indicate if they were set by the user or by default.
    """
    print("\nModel Arguments:")
    for arg in vars(args):
        value = getattr(args, arg)
        if arg in defaults:
            default_value = defaults[arg]
            if value == default_value:
                print(f"{arg}: {value} (default)")
            else:
                print(f"{arg}: {value} (set by user)")
        else:
            # For arguments without explicit defaults (e.g., flags like --use_residual)
            if isinstance(value, bool):
                if value:
                    print(f"{arg}: {value} (set by user)")
            else:
                print(f"{arg}: {value}")



####################### codebook funcs #######################

def codebook_hist_testset(model, val_loader, device):
    model.eval() 
    
    if model.residual:
        num_codebooks = len(model.vq_layer.codebook_sizes)
        size_codebook = model.vq_layer.codebook_sizes[0]
        hist = torch.zeros(num_codebooks,size_codebook ).to(device)

        for i in range(num_codebooks):
        
            with torch.no_grad():
                for batch in val_loader:
                    hist += model.codebook_usage(batch.float().to(device)).to(device)
        
        hist = hist.detach().cpu().numpy()
        
        for i, hist_i in enumerate(hist):
            unused_codes = len(np.where(hist_i == 0.0)[0])
            
            
            percentage = (size_codebook - unused_codes) * 100 / size_codebook
            print(f"Codebook {i+1}: ONLY {size_codebook - unused_codes} OF CODES WERE USED FROM {size_codebook}, WHICH MAKES {percentage:.2f}% OF CODES FROM THE CODEBOOK")

    else : 

        hist = torch.zeros(model.vq_layer.codebook_size).to(device)
        with torch.no_grad():
            for batch in val_loader:
                hist += model.codebook_usage(batch.float().to(device))
        
        hist = hist.detach().cpu().numpy()
        unused_codes = len(np.where(hist == 0.0)[0])

        percentage = (model.vq_layer.codebook_size - unused_codes)*100/model.vq_layer.codebook_size

        print(f" ONLY {model.vq_layer.codebook_size - unused_codes} OF CODES WERE USED FROM {model.vq_layer.codebook_size}, WHICH MAKE {percentage} % OF CODES FROM THE CODE-BOOK")
        
    return hist


