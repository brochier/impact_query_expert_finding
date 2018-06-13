import numpy as np
import sklearn.metrics

"""
'Draw' distribution of precision and recall metrics given scoring/ranking threshold i.e.
best top x (= rank or score) get predicted as positive, the others as negatives.
"""
def get_precision_recall_curve(y_true, y_score):
  return sklearn.metrics.precision_recall_curve(y_true, y_score)

"""
Precision given two predictions like [0,1,0,0,1,0]
"""
def get_precision(y_true, y_score):
    inter_given_true_positives = np.logical_and(y_score, y_true).sum()
    return inter_given_true_positives / y_score.sum()

"""
Recall ...
"""
def get_recall(y_true, y_score):
    inter_given_true_positives = np.logical_and(y_score, y_true).sum()
    return inter_given_true_positives / y_true.sum()


"""
Precision at rank k, where the first k are assigned to positives
"""
def get_precision_at_k(y_true, y_score, k):
    if k > len(y_score):
        k = len(y_score)
    top_k_mask = y_score.argsort()[::-1][0:k]
    return y_true[top_k_mask].sum() / k

"""
Recall at rank k, where the first k are assigned to positives (then always equal to 1)
"""
def get_recall_at_k(y_true, y_score, k):
    top_k_mask = y_score.argsort()[::-1][0:k]
    return y_true[top_k_mask].sum() / y_true[top_k_mask].sum()

"""
Average precision
"""
def get_average_precision(y_true, y_score):
    return sklearn.metrics.average_precision_score(y_true, y_score)

"""
Reciprocal rank
"""
def get_reciprocal_rank(y_true, y_score):
    sorting_index = y_score.argsort()[::-1]
    for k, i in enumerate(sorting_index):
        if y_true[i] == 1:
            return k+1
    return len(y_true)

"""
ROC curve
"""
def get_roc_curve(y_true, y_score):
    return sklearn.metrics.roc_curve(y_true, y_score)


"""
ROC AUC score
"""
def get_roc_auc_score(y_true, y_score):
    return sklearn.metrics.roc_auc_score(y_true, y_score)

