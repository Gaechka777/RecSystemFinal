import torch
import torch.nn as nn
import numpy as np
def calculate_loss(logits, labels):
    """

    Args:
        logits: выход модели
        labels: истинные значения

    Returns:
    Посчитанная функция потерь
    """
    logits = logits.view(-1, logits.size(-1))  # (B*T) x V
    labels = labels.view(-1)  # B*T
    loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
    return loss

def calculate_metrics(scores, candidates, labels):
    """

    Args:
        scores: выход модели, оценка уверенности для каждой категории для определенного клиента
        candidates: кандидаты категорий, которые мы хотим рассмотреть
        labels: истинные значения

    Returns:
    Словарь метрик
    """
    scores = scores[:, -1, :]
    scores = scores.gather(1, candidates)
    metrics = recalls_and_ndcgs_for_ks(scores, labels, [1, 2, 3])
    return metrics

def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()

def recall_k(k, gt, preds):
    c = 0
    for p in preds[:k]:
        if p in gt:
            c += 1
    return c / len(gt)
def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def ndcg_k(k, gt, preds):
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

    Args:
        scores: выход нашей модели
        labels: что должны получить, ground-truth
        ks: топ-к

    Returns:
    Словарь метрик Recall@k, NDCG@k, MRR@k
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

       metrics['Precision@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(cut.device))).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

       hits = labels_float.gather(1, cut)
       hits_temp = []
       for i in hits:
           temp = []
           for q in range(len(i)):
               if i[q] == 1:
                   temp.append(1/(q+1))
           if temp == []:
               hits_temp.append(0)
           else:
               hits_temp.append(np.mean(temp))

       mrr = np.mean(hits_temp)
       metrics['MRR@%d' % k] = mrr
    return metrics