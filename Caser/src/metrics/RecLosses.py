import numpy as np
import pandas as pd
import torch
import torch.nn as nn

"""
Module to provide popular loss functions for banking recommendation systems    
"""

class SimpleLikelihoodLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()

    def forward(self, predictions, targets):
        return -(torch.log(predictions*targets).sum(dim=1)).mean()

class CategoricalCrossEntropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()
    def forward(self, predictions, targets):
        return self.crossentropy(predictions, targets)
    
class Top1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Top1Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.tensor):
        """
            predictions: batch_size x n_items
            targets: batch_size x n_items

            Note, Warning: Видимо, не вполне правильная реализация. В оригинале, если я правильно понял,
                        берут сумму по всем парам positive-negative и считают сумму (neg - pos).
                        Но это очень долго, поэтому я беру  максимальный pos и считаю с ним.
        """
        neg_samples = 1-targets
        num_neg_samples = neg_samples.sum(dim=1)
        max_positive_score, _ = (predictions*targets).max(dim=1)

        r_neg = (neg_samples*(1-max_positive_score).view(-1, 1))*predictions
        part_1 = (self.sigmoid(r_neg)).sum(dim=1)/num_neg_samples
        part_2 = torch.pow(r_neg, 2).sum(dim=1)
        loss = part_1 + part_2
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'None':
            pass
        else:
            raise NotImplementedError
        return loss
    
class BPRLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Top1Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.tensor):
        """
            predictions: batch_size x n_items
            targets: batch_size x n_items

            Note, Warning: Видимо, не вполне правильная реализация. В оригинале, если я правильно понял,
                           берут сумму по всем парам positive-negative и считают сумму (neg - pos).
                           Но это очень долго, поэтому я беру  максимальный pos и считаю с ним.
        """
        neg_samples = 1-targets
        num_neg_samples, _ = neg_samples.sum(dim=1)
        max_positive_score = (predictions*targets).max(dim=1)

        r_neg = (neg_samples*(1-max_positive_score).view(-1, 1))*predictions
        loss = -(torch.log(self.sigmoid(r_neg))).sum(dim=1)/num_neg_samples

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'None':
            pass
        else:
            raise NotImplementedError
        return loss
    
class Top1MaxLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Top1MaxLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction
    def forward(self, predictions: torch.Tensor, targets: torch.tensor):
        """
            predictions: batch_size x n_items
            targets: batch_size x n_items

            Note, Warning: Видимо, не вполне правильная реализация. В оригинале, если я правильно понял,
                        берут сумму по всем парам positive-negative и считают сумму (neg - pos).
                        Но это очень долго, поэтому я беру  максимальный pos и считаю с ним.
        """
        neg_samples = 1-targets
        max_positive_score, _ = (predictions*targets).max(dim=1)
        max_negative_score, _ = (predictions*neg_samples).max(dim=1)

        loss = self.sigmoid(max_negative_score - max_positive_score)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'None':
            pass
        else:
            raise NotImplementedError
        return loss
    
class BPRMaxLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BPRMaxLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction
    def forward(self, predictions: torch.Tensor, targets: torch.tensor):
        """
            predictions: batch_size x n_items
            targets: batch_size x n_items

            Note, Warning: Видимо, не вполне правильная реализация. В оригинале, если я правильно понял,
                        берут сумму по всем парам positive-negative и считают сумму (neg - pos).
                        Но это очень долго, поэтому я беру  максимальный pos и считаю с ним.
        """
        neg_samples = 1-targets
        max_positive_score, _ = (predictions*targets).max(dim=1)
        max_negative_score, _ = (predictions*neg_samples).max(dim=1)

        loss = -torch.log(self.sigmoid(max_negative_score - max_positive_score))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'None':
            pass
        else:
            raise NotImplementedError
        return loss
    
class Top1Combined(nn.Module):
    def __init__(self, reduction='mean'):
        super(Top1Combined, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction
        self.softmax = nn.Softmax(dim=1)

    def forward(self, predictions: torch.Tensor, targets: torch.tensor):
        neg_samples = 1-targets
        num_neg_samples = neg_samples.sum(dim=1)
        max_positive_score, _ = (predictions*targets).max(dim=1)
        softmax_scores = self.softmax(predictions)

        r_neg = (neg_samples*(1-max_positive_score).view(-1, 1))*predictions
        part_1 = (self.sigmoid(r_neg)).sum(dim=1)/num_neg_samples
        part_2 = torch.pow(r_neg, 2).sum(dim=1)
        loss = part_1 + part_2
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'None':
            pass
        else:
            raise NotImplementedError
        return loss

class BPRCombined(nn.Module):
    def __init__(self, reduction='mean'):
        super(Top1Combined, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction
        self.softmax = nn.Softmax()

    def forward(self, predictions: torch.Tensor, targets: torch.tensor):
        pass
