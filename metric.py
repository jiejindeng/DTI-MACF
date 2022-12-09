import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score


def cal_aupr(scores_1d, labels_1d):
    """ scores_1d and labels_1d is 1 dimensional array """
    # prec, rec, thr = precision_recall_curve(labels_1d, scores_1d)
    # aupr_val =  auc(rec, prec)
    # This kind of calculation is not right
    aupr_val = average_precision_score(labels_1d, scores_1d)
    if np.isnan(aupr_val):
        aupr_val = 0.0
    return aupr_val

def cal_auc(scores_1d, labels_1d):
    """ scores_1d and labels_1d is 1 dimensional array """
    fpr, tpr, thr = roc_curve(labels_1d, scores_1d)
    auc_val = auc(fpr, tpr)
    # same with:    roc_auc_score(labels,scores)
    if np.isnan(auc_val):
        auc_val = 0.5
    return auc_val

def cal_metrics(scores, labels):
    """ scores and labels are 2 dimensional matrix, return aucpr, auc"""
    scores_1d = scores.flatten()
    labels_1d = labels.flatten()
    aupr_val = cal_aupr(scores_1d, labels_1d)
    auc_val = cal_auc(scores_1d, labels_1d)
    return aupr_val, auc_val

def accuracy(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    output = []
    for i in outputs:
        if i>=0.5:
            output.append(1)
        else:
            output.append(0)
    output = np.array(output, dtype=np.int)
    labels = np.array(labels, dtype=np.int)
    # outputs = outputs.ge(0.5).type(torch.int32)
    # labels = labels.type(torch.int32)
    corrects = (1 - (output ^ labels))
    # labels = labels.type(np.float)
    if labels.size == 0:
        return np.nan
    return corrects.sum().item() / labels.size


def precision(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    # labels = labels.detach().cpu().numpy()
    # outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    output = []
    for i in outputs:
        if i >= 0.5:
            output.append(1)
        else:
            output.append(0)
    return precision_score(labels, output)


def recall(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    # labels = labels.detach().cpu().numpy()
    # outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    output = []
    for i in outputs:
        if i >= 0.5:
            output.append(1)
        else:
            output.append(0)
    return recall_score(labels, output)


def specificity(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    # labels = labels.detach().cpu().numpy()
    # outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    output = []
    for i in outputs:
        if i >= 0.5:
            output.append(1)
        else:
            output.append(0)
    return recall_score(labels, output, pos_label=0)


def f1(outputs, labels):
    return (precision(outputs, labels) + recall(outputs, labels)) / 2


def mcc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.5).type(torch.int32)
    labels = labels.type(torch.int32)
    true_pos = (outputs * labels).sum()
    true_neg = ((1 - outputs) * (1 - labels)).sum()
    false_pos = (outputs * (1 - labels)).sum()
    false_neg = ((1 - outputs) * labels).sum()
    numerator = true_pos * true_neg - false_pos * false_neg
    deno_2 = outputs.sum() * (1 - outputs).sum() * labels.sum() * (1 - labels).sum()
    if deno_2 == 0:
        return np.nan
    return (numerator / (deno_2.type(torch.float32).sqrt())).item()


def getauc(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    # labels = labels.detach().cpu().numpy()
    # outputs = outputs.detach().cpu().numpy()
    # output = []
    # for i in outputs:
    #     if i >= 0.5:
    #         output.append(1)
    #     else:
    #         output.append(0)
    return roc_auc_score(labels, outputs)

def getaupr(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    # labels = labels.detach().cpu().numpy()
    # outputs = outputs.detach().cpu().numpy()
    return average_precision_score(labels, outputs)