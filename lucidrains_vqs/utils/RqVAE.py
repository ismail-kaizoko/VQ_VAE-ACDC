from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

# from vector_quantize_pytorch import VectorQuantize
# from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import ResidualVQ


###### Hyper Parameters of the Model ######
in_channels = 4 



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




class RQVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 num_quantizers: int,
                 shared_codebook: bool = False,
                #hidden_dims: List = None,
                 downsampling_factor :int = 4,
                 decay : float = 0.8,
                 beta: float = 0.25,
                #  embedding: Tensor = None,
                 **kwargs) -> None:
        super(RQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.shared_codebook = shared_codebook
        self.num_quantizers = num_quantizers
        self.decay = decay

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

        self.vq_layer = ResidualVQ(dim = embedding_dim,
                                    codebook_size = num_embeddings,
                                    commitment_weight = self.beta,
                                    decay = self.decay,
                                    num_quantizers = self.num_quantizers,
                                    shared_codebook = self.shared_codebook,
                                    )

        # Build Decoder
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
                                   out_channels=4,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.ReLU()
                ))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

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
        encoding = self.encode(inputs)[0]
        encoding = encoding.permute(0, 2, 3, 1)
        quantized_inputs, indices, commitment_loss_beta = self.vq_layer(encoding)
        quantized_inputs = quantized_inputs.permute(0, 3, 1, 2)
        return [self.decode(quantized_inputs), inputs, indices, commitment_loss_beta]

    ## !! update codebook_usage

    # def codebook_usage(self, inputs: Tensor, **kwargs) -> List[Tensor]:
    #     encoding = self.encode(inputs)[0]
    #     quantized_hist = self.vq_layer.quantized_latents_hist(encoding)
    #     return quantized_hist



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

        recons_loss = F.cross_entropy(recons,inputs)

        loss = recons_loss + torch.sum(commitment_loss_beta) # sum over all commitement losses of all codebooks
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'commitement Loss':commitment_loss_beta}

    # def sample(self,
    #            num_samples: int,
    #            current_device: Union[int, str], **kwargs) -> Tensor:
    #     raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """


    def codebook_usage(self, inputs):
        encoding = self.encode(inputs)[0]
        encoding = encoding.permute(0, 2, 3, 1)
        _, indices, _ = self.vq_layer(encoding)

        num_codebooks = indices.shape[-1]
        embedding_histogram = torch.zeros(num_codebooks,self.vq_layer.codebook_size )

        for i in range(num_codebooks):
            encoding_inds_flat_i = indices[... , i].view(-1)   # [B,H,W] --> [B,H,W]
            embedding_histogram_i = torch.bincount(encoding_inds_flat_i, minlength=self.vq_layer.codebook_size)  # Count occurrences of each embedding
            embedding_histogram[i] = embedding_histogram_i
            
        return embedding_histogram

