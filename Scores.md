# VQ-VAE performances 

This table contains the performance and parameters of various tests on the vq-vae model on various versions :
* version 1 : raw model prosed on the original paper : 
* version 2 : model finutuned using the Re-Fit method proposed in the article : , to increase the codeBook usage, this have shown very promissing results since we could attaind 100% usage.
* version 3 : using ExponentialMocingAverage to update the codebook instead of the MSE loss
* version 4 : using the RQ-VAE model (link to paper) instead of classic VQ-VAE


|Model version     | Model Name       | Parameters                        | Dice Score (%) | N° Codes used | Codebook usage (%) |
|------------------|------------------|-----------------------------------|:--------------:|:-------------:|:------------------:|
|||||||
|V.1 : raw architecture |
|| Model 100            | epochs=100, D= 64    , K= 512    , downsample = 4 , beta = 0.25         | 99.58           |  89   | 17.38      |
|| Model 101            | epochs=200, D= 64    , K= 512    , downsample = 4 , beta = 0.25         | 99.52           |  70   | 13.67      |
|| Model 102            | epochs=100, D= 32    , K= 256    , downsample = 4 , beta = 0.25         | 98.86           |  53   | 20.70      |
|| Model 103            | epochs=100, D= 64    , K= 512    , downsample = 4 , beta = 1            | 91.23           |  40   | 7.8        |
|| Model 104            | epochs=100, D= 64    , K= 256    , downsample = 4 , beta = 0.25         | 98.75           |  22   | 8.59       |         
|| Model 105            | epochs=100, D= 64    , K= 512    , downsample = 8 , beta = 0.25         | 95.99           |  49   | 9.57       |
|||||||
|V.2 : Re-Fit finutuning |
|| Model 200            | base_model = v1.model100, epochs=100, D= 64    , K= 256    , downsample = 4     | 99.66   | 256    | 100       |
|| Model 201            | base_model = v1.model100, epochs=100, D= 64    , K= 128    , downsample = 4     | 99.67   | 128    | 100       |
|||||||
|V3. : EMA updates



## Notes :
