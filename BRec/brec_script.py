import math
import copy
import os
import argparse
import pickle
import time
import numpy as np

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_warmup as warmup

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Norm(nn.Module):
    """Normalisation layer"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    """Attention function"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    scores_ = scores
    if dropout is not None:
        scores_ = dropout(scores_)
    output = torch.matmul(scores_, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer"""
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * N_heads * seq_len * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention
        output, scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        att = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(att)

        return output, [att, scores]


class FeedForward(nn.Module):
    """Feed-Forward layer"""
    def __init__(self, d_model, hidden_size=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder layer"""
    def __init__(self, d_model, heads, dropout=0.1, hidden_size=2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, hidden_size, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        output, scores = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(output)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, scores


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    """Transformer encoder"""
    def __init__(self, d_model, N, heads, dropout=0.1, hidden_size=2048):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, hidden_size), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask=None):
        scores = [None] * self.N
        for i in range(self.N):
            x, scores[i] = self.layers[i](x, mask)
        return self.norm(x), scores


class Transformer(nn.Module):
    """Transformer"""
    def __init__(self, n_items, d_model, N, heads, dropout=0.1, hidden_size=2048):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout, hidden_size)
        self.out = nn.Linear(d_model, n_items)

    def forward(self, x, mask=None, get_embedding=False, get_scores=False):
        assert get_embedding + get_scores < 2
        x_embedding, scores = self.encoder(x, mask)
        x = torch.mean(x_embedding, dim=-2)
        output = self.out(x)
        if get_embedding:
            return output, x_embedding
        elif get_scores:
            return output, scores
        else:
            return output


def get_model(n_items: int,
              d_model: int,
              heads: int=5,
              dropout: float=0.5,
              n_layers: int=6,
              hidden_size: int=2048,
              weights_path: str=None,
              device: str="cpu") -> Transformer:
    """
    A function for create model with given params.
        :param n_items: int, number of items in customer set
        :param d_model: int, dimensionality of the model
        :param heads: int, number of heads in transformer
        :param dropout: float, dropout rate
        :param n_layers: int, number of layers in model
        :param hidden_size: int, hidden embeddings size
        :param weights_path: str, path to download weights pretrained
        :param device: str, device
        Returns:
            Transformer model
    """
    assert d_model % heads == 0
    assert dropout < 1

    model = Transformer(n_items, d_model, n_layers, heads, dropout, hidden_size)

    if weights_path is not None:
        if not weights_path.endswith('.pth'):
            weights_path = os.path.join(weights_path, "weights.pth")
        print("loading pretrained", weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    else:  # init weights using xavier
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model = model.to(device)

    return model






def precision_k(k, gt, preds):
    """
    A function for prec1, prec2 ... precK calculation.
        :param k: int, scope of metric
        :param gt: list[int], index of ground truth recommendations
        :param preds: list[int], index of predicted recommendations
    Returns:
        precision_k
    """
    c = 0
    for p in preds[:k]:
        if p in gt:
            c += 1
    return c / k


def precision_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    Returns:
        precision_k all over the batch
    """
    precs = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        precs.append(precision_k(k, g, p))
    return sum(precs) / len(gt)


def recall_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    Returns:
        recall_k
    """
    c = 0
    for p in preds[:k]:
        if p in gt:
            c += 1
    return c / len(gt)


def recall_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    Returns:
        recall_k all over the batch
    """
    recalls = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        recalls.append(recall_k(k, g, p))
    return sum(recalls) / len(gt)


def mrr_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    Returns:
        mrr_k
    """
    for i, p in enumerate(preds[:k]):
        if p in gt:
            return 1 / (i + 1)
    return 0.


def mrr_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    Returns:
        mrr_k all over the batch
    """
    mrr = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        mrr.append(mrr_k(k, g, p))
    return sum(mrr) / len(gt)


def ndcg_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    Returns:
        ndcg_k
    """
    c = 0
    j = 0
    for i, p in enumerate(preds[:k]):
        if p in gt:
            c += 1 / np.log(1 + (i + 1))
            j += 1
    d = 0
    for i in range(j):
        d += 1 / np.log(1 + (i + 1))
    if d == 0:
        return 0
    return c / d


def ndcg_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    Returns:
        ndcg_k all over the batch
    """
    ndcg = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        ndcg.append(ndcg_k(k, g, p))
    return sum(ndcg) / len(gt)


def compute_metrics(l, p, verbose=True):
    """
        :param l: list[list[int]], list of index of ground truth recommendations
        :param p: list[list[int]], list of index of predicted recommendations
        Returns:
            Dictionary of metrics
    """
    tot_prec1 = precision_k_batch(1, l, p)
    tot_prec3 = precision_k_batch(3, l, p)
    tot_prec5 = precision_k_batch(5, l, p)
    tot_prec10 = precision_k_batch(10, l, p)
    tot_recall1 = recall_k_batch(1, l, p)
    tot_recall3 = recall_k_batch(3, l, p)
    tot_recall5 = recall_k_batch(5, l, p)
    tot_recall10 = recall_k_batch(10, l, p)
    mrr20 = mrr_k_batch(20, l, p)
    ndcg20 = ndcg_k_batch(20, l, p)
    metrics_dict = {"prec1": tot_prec1, "prec3": tot_prec3, "prec5": tot_prec5, "prec10": tot_prec10,
                    "recall1": tot_recall1, "recall3": tot_recall3, "recall5": tot_recall5, "recall10": tot_recall10,
                    "mrr20": mrr20, "ndcg20": ndcg20}
    if verbose:
        print_metrics_dict(metrics_dict)
    return metrics_dict


def print_metrics_dict(metrics):
    keys = metrics.keys()
    if "prec1" in keys:
        print("Precision 1:", metrics["prec1"])
    if "prec3" in keys:
        print("Precision 3:", metrics["prec3"])
    if "prec5" in keys:
        print("Precision 5:", metrics["prec5"])
    if "prec10" in keys:
        print("Precision 10:", metrics["prec10"])
    if "recall1" in keys:
        print("Recall 1:", metrics["recall1"])
    if "recall3" in keys:
        print("Recall 3:", metrics["recall3"])
    if "recall5" in keys:
        print("Recall 5:", metrics["recall5"])
    if "recall10" in keys:
        print("Recall 10:", metrics["recall10"])
    if "mrr20" in keys:
        print("Mean Reciprocal Rank 20:", metrics["mrr20"])
    if "ndgc20" in keys:
        print("Normalized Discount Cumulative Gain 20:", metrics["ndcg20"])


class CustomDataset(Dataset):
    """Dataset for banking data containig"""
    def __init__(self, train_x, train_y, nrows=None):
        if nrows is None:
            self.data = [(x, y) for x, y in zip(train_x, train_y)]
        else:
            self.data = [(x, y) for x, y in zip(train_x[:nrows], train_y[:nrows])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y


def logits_to_recs(logits):
    """Transforms logits to recomendations

    :param logits: logits from model
    :type logits: np.array
    :return: recomendations
    :rtype: np.array
    """
    logits = np.squeeze(logits)
    recs = np.argsort(logits)[::-1]
    return recs


def train_one_epoch(model, optimizer, criterion, dataset,
                    lr_scheduler, warmup_scheduler, epoch, batch_size=32, device="cpu"):
    """Function to train model in one epoch

    :param model: model to train
    :type model: Transformer
    :param optimizer: optimizer for model
    :type optimizer: torch.optim
    :param criterion: loss function
    :type criterion: callable
    :param dataset: train dataset
    :type dataset: torch.utils.data.Dataset
    :param lr_scheduler: learning rate scheduler
    :type lr_scheduler: torch.optim.lr_scheduler
    :param warmup_scheduler: warmup scheduler
    :type warmup_scheduler: warmup
    :param epoch: number of epoch
    :type epoch: int
    :param batch_size: batch size, defaults to 32
    :type batch_size: int, optional
    :param device: device for training, defaults to "cpu"
    :type device: str, optional
    :return: loss while epoch
    :rtype: float
    """
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    tot_loss = 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        logits = model(batch)
        warmup_scheduler.dampen()
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        tot_loss += loss.item()
    tot_loss /= len(dataset) // batch_size
    return tot_loss


def evaluate_one_epoch(model, criterion, dataset, device="cpu", owned_items=None):
    """Function to evaluate model in one epoch

    :param model: model to estimate
    :type model: Transformer
    :param criterion: loss function
    :type criterion: callable
    :param dataset: val dataset
    :type dataset: torch.utils.data.Dataset
    :param device: device for training, defaults to "cpu"
    :type device: str, optional
    :param owned_items: items which user really owns, defaults to None
    :type owned_items: np.array, optional
    :return: loss and metrics
    """
    batch_size = 1
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )
    model.eval()
    tot_loss = 0.0
    tot_prec1, tot_prec3, tot_prec5, tot_prec10 = 0.0, 0.0, 0.0, 0.0
    mrr20, ndcg20 = 0.0, 0.0
    n_users = 0
    j = 0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)
            loss = criterion(logits, labels)
            tot_loss += loss.item()
            recommendations = logits_to_recs(logits.detach().cpu().numpy())
            real_recommendations = [i for i, p in enumerate(labels[0].detach().cpu().numpy()) if int(float(p)) == 1]
            if owned_items is not None:
                old_items = [i for i, p in enumerate(owned_items[j]) if int(float(p)) == 1]
                real_recommendations = [i for i in real_recommendations if i not in old_items]
                recommendations = [i for i in recommendations if i not in old_items]
            if len(real_recommendations) > 0:
                n_users += 1
            else:
                continue
            tot_prec1 += precision_k(1, real_recommendations, recommendations)
            tot_prec3 += precision_k(3, real_recommendations, recommendations)
            tot_prec5 += precision_k(5, real_recommendations, recommendations)
            tot_prec10 += precision_k(10, real_recommendations, recommendations)
            mrr20 += mrr_k(20, real_recommendations, recommendations)
            ndcg20 += ndcg_k(20, real_recommendations, recommendations)
        tot_loss /= len(dataset) // batch_size
        tot_prec1 /= n_users
        tot_prec3 /= n_users
        tot_prec5 /= n_users
        tot_prec10 /= n_users
        mrr20 /= n_users
        ndcg20 /= n_users
        metrics_dict = {"prec1": tot_prec1, "prec3": tot_prec3, "prec5": tot_prec5,
                        "prec10": tot_prec10, "mrr20": mrr20, "ndcg20": ndcg20}
    return tot_loss, metrics_dict

        
def train_pipeline(train_dataset, val_dataset, model, args):
    """Training pipeline

    :param train_dataset: train dataset
    :type train_dataset:  torch.utils.data.Dataset
    :param val_dataset: val dataset
    :type val_dataset: torch.utils.data.Dataset
    :param model: model to train
    :type model: Transformer
    :param args: arguments for training
    :type args: argparse.ArgumentParser
    :return: trained model
    :rtype: Transformer
    """
    
    # Set criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-9)
    
    # warmup lr
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.warmup_epochs], gamma=0.1)
    if args.warmup_type == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif args.warmup_type == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif args.warmup_type == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)

    # initialize the step counter
    warmup_scheduler.last_step = -1
    
    # Set initials for model savings
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    
    # Training cycle
    train_losses = {}
    val_losses = {}
    start = time.time()
    print("training...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_dataset, lr_scheduler, warmup_scheduler, epoch, args.batch_size, args.device)
        print(f"epoch {epoch + 1} | train loss: {train_loss}")
        train_losses[epoch] = train_loss
        
        # Validation
        if not args.no_val and epoch % args.val_every_n == 0:
                val_loss, val_metrics = evaluate_one_epoch(model, criterion, val_dataset, args.device)
                print(f"epoch {epoch + 1} | val loss: {val_loss}")
                val_losses[epoch] = val_losses
                for metric in val_metrics.keys():
                    print(f"{metric}: {val_metrics[metric]}")
                
                # Best score updating
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model.state_dict())
                    torch.save(best_model, 
                               os.path.join(args.weights_path, f"best_model_weights.pth"))
                    print(f"Best model with new best validation score {best_val_loss} saved at ", 
                          os.path.join(args.weights_path, "best_model_weights.pth"))

        # Model saving
        if args.save_weights_epoch is not None and epoch % args.save_weights_epoch == 0:
                torch.save(model.state_dict(), os.path.join(args.weights_path, f"weights_{epoch}.pth"))
                print("model saved at", os.path.join(args.weights_path, f"weights_{epoch}.pth"))
        print("finished training in", time.time() - start)

    # Logs saving
    if args.log_dir is not None:
        with open(os.path.join(args.log_dir, 'train_logs.pkl'),'wb') as handle:
            pickle.dump(train_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.log_dir, 'val_logs.pkl'),'wb') as handle:
            pickle.dump(val_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model

def __main__():
    parser = argparse.ArgumentParser()

    # Dataset location
    # Datasets are supposed to be written into files
    # TRAIN
    #___________________
    # ~/train_dataset/x.npy
    # ~/train_dataset/y.npy
    #
    # VAL
    #___________________
    # ~/val_dataset/x.npy
    # ~/val_dataset/y.npy
    #
    # TEST
    #___________________
    # ~/test_dataset/x.npy
    # ~/test_dataset/y.npy
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--val_dataset', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)

    # Configuration of usage
    parser.add_argument('--no_train', type=bool, default=False)
    parser.add_argument('--no_val', type=bool, default=False)
    parser.add_argument('--no_test', type=bool, default=False)
    parser.add_argument('--load_weights', type=bool, default=False)
    parser.add_argument('--save_weights_epoch', type=int, default=None)
    parser.add_argument('--weights_path', type=str, default='./model')
    parser.add_argument('--predict_only', type=bool, default=False)
    parser.add_argument('--predictions_save_file', type=str, default='./prediction.npy')
    parser.add_argument('--prediction_type', type=str, default='logits')

    # Info about data
    parser.add_argument('--n_items', type=int, default=22)
    parser.add_argument('--limit_rows', type=int, default=1_000_000)

    # Model configuration
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--heads', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Training params
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_type', type=str, default='linear')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--val_every_n', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()

    assert args.prediction_type in ['logits', 'recommendations'], "Specify prediction_type one of 'logits' or 'recommendations'"
    
    if args.predict_only:
        if not args.no_test:
            test_x = np.load(os.path.join(args.test_dataset, "x.npy"))
            test_y = np.load(os.path.join(args.test_dataset, "y.npy"))
            test_set = CustomDataset(test_x, test_y, nrows=args.limit_rows)


        # model load or creation
        if args.load_weights:
            model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                              args.n_layers, args.hidden_size, args.weights_path, args.device)
            print("model loaded from", weights_path)
        else:
            model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                              args.n_layers, args.hidden_size, None, args.device)
        if args.prediction_type == 'logits':
            criterion = nn.BCEWithLogitsLoss()
            preds = evaluate_one_epoch(model, criterion, test_set, args.device, return_logits=True)
        else:
            criterion = nn.BCEWithLogitsLoss()
            preds = evaluate_one_epoch(model, criterion, test_set, args.device, return_recs=True)
        
        preds = np.stack(preds)
        np.save(args.predictions_save_file, preds)
    else:
    
        assert args.log_dir is None or os.path.isdir(args.log_dir), 'log_dir is not None and does not exist. Specify existing path or do not call --log_dir not to save learning logs'
        assert args.val_every_n>0, 'Val_every_n must be positive. If you do not want to estimate validation score just call flag --no_val'

        # Data loading
        assert args.train_dataset is not None or args.no_train, 'Set directory with train dataset with --train_dataset "directory"'
        assert args.val_dataset is not None or args.no_val, 'Set directory with val dataset with --val_dataset "directory"'
        assert args.test_dataset is not None or args.no_test, 'Set directory with test dataset with --test_dataset "directory"'

        if not args.no_train:
            train_x = np.load(os.path.join(args.train_dataset, "x.npy"))
            train_y = np.load(os.path.join(args.train_dataset, "y.npy"))
            train_set = CustomDataset(train_x, train_y, nrows=args.limit_rows)

        if not args.no_val:
            val_x = np.load(os.path.join(args.val_dataset, "x.npy"))
            val_y = np.load(os.path.join(args.val_dataset, "y.npy"))
            val_set = CustomDataset(val_x, val_y, nrows=args.limit_rows)

        if not args.no_test:
            test_x = np.load(os.path.join(args.test_dataset, "x.npy"))
            test_y = np.load(os.path.join(args.test_dataset, "y.npy"))
            test_set = CustomDataset(test_x, test_y, nrows=args.limit_rows)


        # model load or creation
        if args.load_weights:
            model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                              args.n_layers, args.hidden_size, args.weights_path, args.device)
            print("model loaded from", weights_path)
        else:
            model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                              args.n_layers, args.hidden_size, None, args.device)

        # Model training
        if not args.no_train:
            if not args.no_val:
                model = train_pipeline(train_set, val_set, model, args)
            else:
                model = train_pipeline(train_set, None, model, args)

        # Model test
        if not args.no_test:
            print("testing...")
            criterion = nn.BCEWithLogitsLoss()
            test_loss, test_metrics = evaluate_one_epoch(model, criterion, test_set, args.device)
            print("--Test results--")
            print("Test loss:", test_loss)
            print_metrics_dict(test_metrics)

if __name__ == '__main__':
    __main__()