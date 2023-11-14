# Bert4Rec-PyTorch-Lighting

A PyTorch implementation Bert4Rec from the paper in Lighting pipeline:

*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM 2019*

# Requirements
* Python 2 or 3
* [PyTorch v0.4+](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* lightning

# Usage
Download dataset 'train_ver2.csv' to data folder from (https://www.kaggle.com/c/santander-product-recommendation).
Install dependencies

```bash
# clone project
git clone https://github.com/Gaechka777/RecSystem/tree/main/

# install requirements
pip install -r requirements.txt

#train model
python train.py --config-name=train_bert.yaml trainer.gpus=[0]
```

# Configurations

#### Data

- In this experiment we use Santander bank`s dataset from (https://www.kaggle.com/c/santander-product-recommendation)

- After special preprocessing we obtain train, val and test datasets, each file contains a collection of quadrets:

  > user item rating timestamp

  The quadrets are organized in *time order*.

- As the problem is Sequential Recommendation, the rating doesn't matter, so we convert them to all 1.

#### Lighting

- It is possible to collect all training information in https://www.comet.com/ with useful graphics.
 You only need to change credentials in <code>configs/logger/comet.yaml</code>.
 
- You can change crucial training parameters in <code>configs/train.yaml</code>.
 Also it is possible to change model`s init parameters in <code>configs/model/bert.yaml</code>.

# Acknowledgment

This project (utils.py, etc.) is  built on [Bert4Rec_pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch).
