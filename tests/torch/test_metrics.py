import pytest
import torch
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    recall_score as sk_recall,
    f1_score as sk_f1,
    precision_score as sk_precision,
)

from astra.torch.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

y = torch.randint(0, 2, (30,))
y_pred = torch.randint(0, 2, (30,))
target_accuracy = sk_accuracy(y.numpy(), y_pred.numpy())

target_metrics = {0: {}, 1: {}}

for positive_label in [0, 1]:
    target_metrics[positive_label]["precision"] = sk_precision(y.numpy(), y_pred.numpy(), pos_label=positive_label)
    target_metrics[positive_label]["recall"] = sk_recall(y.numpy(), y_pred.numpy(), pos_label=positive_label)
    target_metrics[positive_label]["f1"] = sk_f1(y.numpy(), y_pred.numpy(), pos_label=positive_label)


def test_accuracy():
    assert accuracy_score(y_pred, y) == target_accuracy


def test_precision():
    for positive_label in [0, 1]:
        assert precision_score(y_pred, y, positive_label=positive_label) == target_metrics[positive_label]["precision"]


def test_recall():
    for positive_label in [0, 1]:
        assert recall_score(y_pred, y, positive_label=positive_label) == target_metrics[positive_label]["recall"]


def test_f1():
    for positive_label in [0, 1]:
        assert f1_score(y_pred, y, positive_label=positive_label) == target_metrics[positive_label]["f1"]


def test_classification_report():
    for positive_label in [0, 1]:
        report = classification_report(y_pred, y, positive_label=positive_label)
        f1 = f1_score(y_pred, y, positive_label=positive_label)
        precision = precision_score(y_pred, y, positive_label=positive_label)
        recall = recall_score(y_pred, y, positive_label=positive_label)
        accuracy = accuracy_score(y_pred, y)
        assert report["accuracy"] == accuracy
        assert report["precision"] == precision
        assert report["recall"] == recall
        assert report["f1"] == f1
