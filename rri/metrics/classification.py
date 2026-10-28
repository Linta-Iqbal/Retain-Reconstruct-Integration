"""Classification metrics shared by Food101, MVSA, and NYUD."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score


def _labels(values: torch.Tensor | np.ndarray | Sequence[int]) -> np.ndarray:
    if torch.is_tensor(values):
        tensor = values.detach().cpu()
        if tensor.ndim > 1:
            tensor = tensor.argmax(dim=-1)
        return tensor.numpy().reshape(-1)
    array = np.asarray(values)
    if array.ndim > 1:
        array = array.argmax(axis=-1)
    return array.reshape(-1)


def accuracy(predictions, targets) -> float:
    predicted = _labels(predictions)
    expected = _labels(targets)
    if len(predicted) != len(expected) or len(expected) == 0:
        raise ValueError("Predictions and targets must have the same non-zero length")
    return float(np.mean(predicted == expected))


def classification_metrics(predictions, targets) -> dict[str, float]:
    predicted = _labels(predictions)
    expected = _labels(targets)
    return {
        "accuracy": accuracy(predicted, expected),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(expected, predicted, average="weighted", zero_division=0)
        ),
    }

