import torch
import torch.nn as nn
import numpy as np


def calculate_loss(logits, labels):
    """

    Parameters
    ----------
    logits : model`s logits
    labels : real answers

    Returns
    -------
    calculated loss
    """
    logits = logits.view(-1, logits.size(-1))  # (B*T) x V
    labels = labels.view(-1)  # B*T
    loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
    return loss


def calculate_metrics(scores, candidates, labels):  # B x T x V
    """

    Parameters
    ----------
    scores : model`s scores
    candidates : subset of targets
    labels : real answers

    Returns
    -------
    calculated metrics
    """
    #     scores = scores[:, -1, :]  # B x V

    # scores = scores.gather(1, candidates)  # B x C
    labels_new = torch.zeros(scores.shape)
    for index, row in enumerate(labels_new):
        row[labels[index]] = 1
    labels = labels_new

    metrics = recalls_and_ndcgs_for_ks(scores, labels, [1, 2, 3])  # [1, 5, 10, 20, 50, 100]
    return metrics


def recall(scores, labels, k):
    """

    Parameters
    ----------
    scores : model`s scores
    labels : real answers
    k : number for calculating top-k metric

    Returns
    -------
    recall-k score
    """
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def recall_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    """
    c = 0
    for p in preds[:k]:
        if p in gt:
            c += 1
    return c / len(gt)


def ndcg(scores, labels, k):
    """

    Parameters
    ----------
    scores : model`s scores
    labels : real answers
    k : number for calculating top-k metric

    Returns
    -------
    ndcg-k metric
    """
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg_new = dcg / idcg
    return ndcg_new.mean()


def ndcg_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
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


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """

    Parameters
    ----------
    scores : model`s scores
    labels : real answers
    ks : numbers for calculating top-k metric
    """
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
        ndcg_new = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg_new.cpu().item()

        hits = labels_float.gather(1, cut)
        hits_temp = []
        for i in hits:
            temp = []
            for q in range(len(i)):
                if i[q] == 1:
                    temp.append(1 / (q + 1))
            if not temp:
                hits_temp.append(0)
            else:
                hits_temp.append(np.mean(temp))

        mrr = np.mean(hits_temp)
        metrics['MRR@%d' % k] = mrr

    return metrics
