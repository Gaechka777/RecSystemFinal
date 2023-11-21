import torch
import torch.nn as nn
import numpy as np


def calculate_loss(logits, labels):
    """"
    Args:
        logits: model output
        labels: true values

    Returns:
    Calculated loss function
    """
    logits = logits.view(-1, logits.size(-1))  # (B*T) x V
    labels = labels.view(-1)  # B*T
    loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
    return loss


def write_to_file(answer, path_res, batch_idx):
    file = open(path_res, 'w')
    file.write('user label1 label2 label3')
    file.write("\n")
    for k in range(len(answer)):
        file.write(str(k + 64 * batch_idx) + ' ' + ' '.join(map(str, answer[k].tolist())))
        file.write("\n")
    file.close()

def calculate_metrics(scores,
                      candidates,
                      labels,
                      stage,
                      cand_need,
                      k_labels,
                      batch_idx,
                      path_to_res
    ):
    """

    Args:
        scores: model output, confidence score for each category for a specific customer
        candidates: candidates of the categories we want to consider
        labels: true values
        stage: train/val/test
        can_need: to output a recommendation for each user, there are candidates or not
        k_labels: how many categories are we predicting
        batch_idx: batch number
        path_to_res: path to save results

    Returns:
    Dict of metrics
    """
    scores = scores[:, -1, :]
    path_res = f'{path_to_res}results_{batch_idx}.txt'

    if stage == 'test':
        if cand_need:
            sc_cand = scores.gather(1, candidates)
            #print('candidates before -- ', candidates)
            r = (-sc_cand).argsort(dim=1)
            print('Candidates for each clients -- ', candidates.gather(1, r))
            answer = candidates.gather(1, r)
            write_to_file(answer, path_res, batch_idx)
        else:
            #print('scores', scores)
            answer = (-scores).argsort(dim=1)[:, :k_labels]
            print(f'{k_labels} candidates for each clients', answer)
            write_to_file(answer, path_res, batch_idx)

    scores = scores.gather(1, candidates)

    metrics = recalls_and_ndcgs_for_ks(scores, labels, list(range(1, k_labels + 1)))
    return metrics


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """

    Args:
        scores: output of our model
        labels: true values(ground-truth)
        ks: top-k

    Returns:
    Dictionary of Metrics Recall@k, NDCG@k, MR@k
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
        metrics['Recall@%d' % k] = (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                    labels.sum(1).float())).mean().cpu().item()

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

            if not temp:
                hits_temp.append(0)
            else:
                hits_temp.append(np.mean(temp))

        mrr = np.mean(hits_temp)
        metrics['MRR@%d' % k] = mrr
    return metrics
