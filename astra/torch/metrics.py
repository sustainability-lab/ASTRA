import torch


def accuracy_score(y_pred, y):
    return (y_pred == y).float().mean()


def true_positives_score(y_pred, y, pos_label=1):
    return torch.logical_and(y_pred == pos_label, y == pos_label).sum()


def false_positives_score(y_pred, y, pos_label=1):
    return torch.logical_and(y_pred == pos_label, y != pos_label).sum()


def false_negatives_score(y_pred, y, pos_label=1):
    return torch.logical_and(y_pred != pos_label, y == pos_label).sum()


def precision_score(y_pred, y, pos_label=1):
    tp = true_positives_score(y_pred, y, pos_label)
    fp = false_positives_score(y_pred, y, pos_label)

    return tp / (tp + fp)


def recall_score(y_pred, y, pos_label=1):
    tp = true_positives_score(y_pred, y, pos_label)
    fn = false_negatives_score(y_pred, y, pos_label)

    return tp / (tp + fn)


def f1_score(y_pred, y, pos_label=1):
    tp = true_positives_score(y_pred, y, pos_label)
    fp = false_positives_score(y_pred, y, pos_label)
    fn = false_negatives_score(y_pred, y, pos_label)

    return tp / (tp + (fp + fn) / 2)
