import pytest
import torch
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    recall_score as sk_recall,
    f1_score as sk_f1,
    precision_score as sk_precision,
)

from astra.torch.metrics import accuracy_score, precision_score, recall_score, f1_score

y = torch.randint(0, 2, (30,))
y_pred = torch.randint(0, 2, (30,))
target_accuracy = sk_accuracy(y.numpy(), y_pred.numpy())
target_precision_1 = sk_precision(y.numpy(), y_pred.numpy(), pos_label=1)
target_precision_0 = sk_precision(y.numpy(), y_pred.numpy(), pos_label=0)
target_recal_1 = sk_recall(y.numpy(), y_pred.numpy(), pos_label=1)
target_recal_0 = sk_recall(y.numpy(), y_pred.numpy(), pos_label=0)
target_f1_1 = sk_f1(y.numpy(), y_pred.numpy(), pos_label=1)
target_f1_0 = sk_f1(y.numpy(), y_pred.numpy(), pos_label=0)


def test_accuracy():
    assert accuracy_score(y_pred, y) == target_accuracy


def test_precision():
    assert precision_score(y_pred, y, pos_label=1) == target_precision_1
    assert precision_score(y_pred, y, pos_label=0) == target_precision_0


def test_recall():
    assert recall_score(y_pred, y, pos_label=1) == target_recal_1
    assert recall_score(y_pred, y, pos_label=0) == target_recal_0


def test_f1():
    assert f1_score(y_pred, y, pos_label=1) == target_f1_1
    assert f1_score(y_pred, y, pos_label=0) == target_f1_0
