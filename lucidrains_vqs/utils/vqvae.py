from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import ResidualVQ




###################################################### Details ######################################################## 
# this is an implementation of a Coder-Decoder VQ-VAE based on a Vactor-Quantizer module of lucid-frains implementation




class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)




class VQVAE(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 downsampling_factor :int = 4,
                 residual = False,
                 num_quantizers: int = 2,
                 shared_codebook: bool = False,
                 beta: float = 0.25,
                #  embedding: Tensor = None,
                 decay : float = 0.8,
                 data_mod = 'SEG' ,
                 kmeans_init = False,   # set to True
                 kmeans_iters = 10,
                 in_channels = None, 
                 loss_func = None,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.data_mod = data_mod
        self.residual = residual
        self.shared_codebook = shared_codebook
        self.num_quantizers = num_quantizers
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters


        if(self.data_mod not in ['SEG', 'MRI']):
            assert "data modalitie must be ether 'SEG' for heart segmentations or 'MRI' for mri scans"
        
        # set in_channels
        if (in_channels == None):
            #if None adapt to dataset type
            if self.data_mod == 'SEG':
                in_channels = 4  # one hot encoding input for the 4 classes
            else :
                in_channels = 1  # gray scale image

        self.in_channels = in_channels


        # set loss_function

        self.loss_functions = {
            'MSE' : F.mse_loss,
            'L1'  : F.l1_loss,
            'CrossEntropy' : F.cross_entropy
        }

        if (loss_func == None):
            #if None, adapt to dataset type (regression Vs Classification)
            if self.data_mod == 'SEG':
                self.loss_func = F.cross_entropy  # one hot encoding input for the 4 classes
            else :
                self.loss_func = F.mse_loss   # gray scale image

        else : 
            if loss_func not in self.loss_functions:
                raise ValueError(f"Unsupported loss function: {loss}. Supported losses are: {list(self.loss_functions.keys())}")
            else : 
                self.loss_func = self.loss_functions[loss_func]
        
    



        modules = []
        
        if downsampling_factor < 2 :
            raise Warning("VQVAE can't have a donwsampling factor less than 2")
        elif downsampling_factor ==2 :
            hidden_dims = [64]
        elif downsampling_factor == 4 :
            hidden_dims = [64, 128]
        elif downsampling_factor == 8 :
            hidden_dims = [64, 128, 256]
        else:
            assert("donwsamlping factor must be one of the following values : {2,4,8}")



        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(2):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)




        ############## define the quantization-operator ##################

        if self.residual : 
            
            self.vq_layer = ResidualVQ(dim = embedding_dim,
                                        codebook_size = num_embeddings,
                                        commitment_weight = self.beta,
                                        decay = self.decay,
                                        num_quantizers = self.num_quantizers,
                                        shared_codebook = self.shared_codebook,
                                        accept_image_fmap = True,
                                        kmeans_init = self.kmeans_init,
                                        kmeans_iters = self.kmeans_iters,
                                        **kwargs
                                        )
        else :
            self.vq_layer = VectorQuantize(dim = embedding_dim,
                                            codebook_size = num_embeddings,
                                            commitment_weight = self.beta,
                                            decay = self.decay,
                                            accept_image_fmap = True,
                                            kmeans_init = self.kmeans_init,
                                            kmeans_iters = self.kmeans_iters,
                                            **kwargs )



        ####################### Build The Decoder  #########################  
        #  !! Decoder customizable to data modality.

        if self.data_mod == 'SEG':
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim,
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=1,
                            padding=1),
                    nn.LeakyReLU())
            )

            for _ in range(2):
                modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

            modules.append(nn.LeakyReLU())

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=4,
                                        stride=2,
                                        padding=1),
                        nn.LeakyReLU())
                )

            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                    out_channels= self.in_channels,
                                    kernel_size=4,
                                    stride=2, padding=1),
                    nn.ReLU()
                    ))
        
        else:
            modules = []
            
            # Initial Conv layer
            modules.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim,
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=1,
                            padding=1),
                    nn.LeakyReLU())
            )

            # Residual layers
            for _ in range(2):
                modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

            modules.append(nn.LeakyReLU())

            hidden_dims.reverse()

            # Replace ConvTranspose2d with Interpolation + Conv2d
            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        # Interpolation to replace upsampling via ConvTranspose2d
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU())
                )

            # Final upsampling and convolution layer
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(inputs)
        quantized_inputs, indices, commitment_loss_beta = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), inputs, indices, commitment_loss_beta]


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inputs = args[1]
        indices = args[2]
        commitment_loss_beta = args[3]

        # if self.data_mod == 'SEG':
        #     recons_loss = F.cross_entropy(recons,inputs)
        # else : 
        #     recons_loss = F.mse_loss(recons,inputs)

        recons_loss = self.loss_func(recons,inputs)


        if self.residual :
            commitment_loss_beta = torch.sum(commitment_loss_beta) # sum over all commitement losses of all codebooks

        loss = recons_loss + commitment_loss_beta


        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'commitement_Loss':commitment_loss_beta}



    def generate_from_indices(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """ 
        pass


    def reconstruct(self, x):
        logits, _, _, _ = model(x.float())

        if self.data_mod == 'SEG':
            return F.softmax(logits, dim=1)
        else : 
            return logits 



    def codebook_usage(self, inputs):
        encoding = self.encode(inputs)
        _, indices, _ = self.vq_layer(encoding)

        if self.residual :

            num_codebooks = indices.shape[-1]
            embedding_histogram = torch.zeros(num_codebooks, self.vq_layer.codebook_sizes[0] )

            for i in range(num_codebooks):
                encoding_inds_flat_i = indices[... , i].view(-1)   # [B,H,W] --> [B,H,W]
                embedding_histogram_i = torch.bincount(encoding_inds_flat_i, minlength=self.vq_layer.codebook_size)  # Count occurrences of each embedding
                embedding_histogram[i] = embedding_histogram_i
        else :     
            encoding_inds_flat = indices.view(-1)   # [B,H,W] --> [B,H,W]
            embedding_histogram = torch.bincount(encoding_inds_flat, minlength=self.vq_layer.codebook_size)  # Count occurrences of each embedding
        
        
        return embedding_histogram


