# VQ-VAE performances

This table contains the performance and parameters of various tests on the vq-vae model on various versions :

- **version 1** : raw model prosed on the original paper :
- **version 2** : model finutuned using the Re-Fit method proposed in the article : , to increase the codeBook usage, this have shown very promissing results since we could attaind 100% usage.
- **version 3** : using ExponentialMocingAverage to update the codebook instead of the MSE loss
- **version 4** : using the RQ-VAE model (link to paper) instead of classic VQ-VAE

<!-- | Model version           | Model Name | Parameters                                                            | Dice Score (%) | N° Codes used | Codebook usage (%) |
| ----------------------- | ---------- | --------------------------------------------------------------------- | :------------: | :-----------: | :----------------: |
|                         |            |                                                                       |                |               |                    |
| V.1 : raw architecture  |
|                         | Model 100  | epochs=100, D= 64 , K= 512 , downsample = 4 , beta = 0.25             |     99.58      |      89       |       17.38        |
|                         | Model 101  | epochs=200, D= 64 , K= 512 , downsample = 4 , beta = 0.25             |     99.52      |      70       |       13.67        |
|                         | Model 102  | epochs=100, D= 32 , K= 256 , downsample = 4 , beta = 0.25             |     98.86      |      53       |       20.70        |
|                         | Model 103  | epochs=100, D= 64 , K= 512 , downsample = 4 , beta = 1                |     91.23      |      40       |        7.8         |
|                         | Model 104  | epochs=100, D= 64 , K= 256 , downsample = 4 , beta = 0.25             |     98.75      |      22       |        8.59        |
|                         | Model 105  | epochs=100, D= 64 , K= 512 , downsample = 8 , beta = 0.25             |     95.99      |      49       |        9.57        |
|                         |            |                                                                       |                |               |                    |
| V.2 : Re-Fit finutuning |
|                         | Model 200  | base_model = v1.model100, epochs=100, D= 64 , K= 256 , downsample = 4 |     99.66      |      256      |        100         |
|                         | Model 201  | base_model = v1.model100, epochs=100, D= 64 , K= 128 , downsample = 4 |     99.67      |      128      |        100         |
|                         |            |                                                                       |                |               |                    | -->

| Model version | Model Name | Parameters | Dice Score (%) | N° Codes used | Codebook usage (%) |
| ------------- | ---------- | ---------- | :------------: | :-----------: | :----------------: |

|V3. : EMA updates
|| Model 300 | epochs=100, D= 64 , K= 512 , downsample = 4 , beta = 0.25 | 93.57 | x | 3.51 |
|| Model 301 | epochs=100, D= 64 , K= 512 , downsample = 8 , beta = 0.25 | 94.45 | x | 3.32 |
|| Model 302 | epochs=100, D= 64 , K= 256 , downsample = 8 , beta = 0.25 | 95.21 | x | 12.89 |
|| Model 303 | epochs=100, D= 64 , K= 128 , downsample = 8 , beta = 0.25 | 91.46 | x | 21.875 |
|V4. : Refit
|| Model 400 from 300 | epochs=50, D= 64 , K= 128 , downsample = 4 , beta = 0.25 | 94 | x | 56.25 |
|| Model 401 from 300 | epochs=50, D= 64 , K= 256 , downsample = 4 , beta = 0.25 | 94 | x | 50.78 |
|| Model 402 from 301 | epochs=50, D= 64 , K= 128 , downsample = 8 , beta = 0.25 | 96.4 | x | 61.7 |

<!-- ### Here are the performances of the RQ-VAE model : (paper : .link)

| Model Name | Parameters                                                                       | Dice Score (%) |
| ---------- | -------------------------------------------------------------------------------- | :------------: |
| Model 400  | D= 64 , K= 512 , downsample =8 ,num_cb = 2, shared_cb = false,kmeans_init = true |     97.10      |
| Model 401  | D= 64 , K= 128 , downsample =8 ,num_cb = 2, shared_cb = false,kmeans_init = true |     94.94      | -->
