import numpy as np
import torch

from rri.metrics import accuracy, classification_metrics, mosi_metrics, regression_metrics


def test_classification_metrics_accept_logits():
    logits = torch.tensor([[4.0, 1.0], [0.1, 2.0], [3.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    assert accuracy(logits, targets) == 2 / 3
    metrics = classification_metrics(logits, targets)
    assert metrics["accuracy"] == 2 / 3
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_regression_metrics_are_finite_for_constant_values():
    metrics = regression_metrics([1.0, 1.0], [1.0, 2.0])
    assert metrics == {"mae": 0.5, "pearson": 0.0}


def test_mosi_metrics_follow_notebook_protocol():
    targets = np.array([-2.0, -0.2, 0.0, 1.1, 2.9])
    predictions = np.array([-1.0, 0.1, 2.0, 0.5, 3.0])
    metrics = mosi_metrics(predictions, targets)
    assert set(metrics) == {
        "mae",
        "pearson",
        "binary_accuracy",
        "weighted_f1",
        "accuracy_7",
    }
    assert metrics["binary_accuracy"] == 0.75

