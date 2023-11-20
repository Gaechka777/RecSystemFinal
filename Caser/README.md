# Caser-PyTorch-Lighting

A PyTorch implementation of Convolutional Sequence Embedding Recommendation Model (Caser) from the paper in Lighting pipeline:

*Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18*

# Requirements
* Python 2 or 3
* [PyTorch v0.4+](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* lightning

# Usage
1. Install required packages.
2. Download dataset 'train_ver2.csv' to data folder from (https://www.kaggle.com/c/santander-product-recommendation).
2. run <code>python train.py</code>

# Configurations

#### Data

The data can be downloaded from this link (https://drive.google.com/file/d/1te9dc86cXWvpyy12vbbaNqbSDpkibPri/view?usp=sharing).

- In this experiment we use Santander bank`s dataset from (https://www.kaggle.com/c/santander-product-recommendation)

- After special preprocessing we obtain train, val and test datasets, each file contains a collection of quadrets:

  > user item rating timestamp

  The quadrets are organized in *time order*.

- As the problem is Sequential Recommendation, the rating doesn't matter, so we convert them to all 1.

#### Lighting

- It is possible to collect all training information in https://www.comet.com/ with useful graphics.
 You only need to change credentials in <code>configs/logger/comet.yaml</code>.
 
- You can change crucial training parameters in <code>configs/train.yaml</code>.
 Also it is possible to change model`s init parameters in <code>configs/model/caser.yaml</code>.
 
#### Model Args (in configs)

- <code>L</code>: length of sequence
- <code>T</code>: number of targets
- <code>d</code>: number of latent dimensions
- <code>nv</code>: number of vertical filters
- <code>nh</code>: number of horizontal filters
- <code>ac_conv</code>: activation function for convolution layer (i.e., phi_c in paper)
- <code>ac_fc</code>: activation function for fully-connected layer (i.e., phi_a in paper)
- <code>drop_rate</code>: drop ratio when performing dropout

# Acknowledgment

This project (utils.py, interactions.py, etc.) is  built on [Caser_pytorch](https://github.com/graytowne/caser_pytorch/tree/master).
