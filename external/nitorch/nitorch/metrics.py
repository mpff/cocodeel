import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, roc_curve, auc


def specificity(y_true, y_pred):
    """Gets the specificity of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The specificity.

    """
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    """Gets the sensitivity of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The sensitivity.

    """
    return recall_score(y_true, y_pred, pos_label=1)


def auc_score(y_true, y_pred):
    """Gets the auc score of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The auc score.

    """

    y_true, y_pred = prepare_values(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)
