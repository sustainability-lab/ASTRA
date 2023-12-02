import torch


def accuracy_score(y_pred, y):
    return (y_pred == y).float().mean()


def true_positives_score(y_pred, y, positive_label=1):
    return torch.logical_and(y_pred == positive_label, y == positive_label).sum()


def false_positives_score(y_pred, y, positive_label=1):
    return torch.logical_and(y_pred == positive_label, y != positive_label).sum()


def false_negatives_score(y_pred, y, positive_label=1):
    return torch.logical_and(y_pred != positive_label, y == positive_label).sum()


def precision_score(y_pred, y, positive_label=1):
    tp = true_positives_score(y_pred, y, positive_label)
    fp = false_positives_score(y_pred, y, positive_label)

    return tp / (tp + fp)


def recall_score(y_pred, y, positive_label=1):
    tp = true_positives_score(y_pred, y, positive_label)
    fn = false_negatives_score(y_pred, y, positive_label)

    return tp / (tp + fn)


def f1_score(y_pred, y, positive_label=1):
    tp = true_positives_score(y_pred, y, positive_label)
    fp = false_positives_score(y_pred, y, positive_label)
    fn = false_negatives_score(y_pred, y, positive_label)

    return tp / (tp + (fp + fn) / 2)


def classification_report(y_pred, y, positive_label=1):
    accuracy = accuracy_score(y_pred, y)
    tp = true_positives_score(y_pred, y, positive_label)
    fp = false_positives_score(y_pred, y, positive_label)
    fn = false_negatives_score(y_pred, y, positive_label)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = tp / (tp + (fp + fn) / 2)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
