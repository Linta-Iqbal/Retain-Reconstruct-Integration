"""Regression and CMU-MOSI sentiment metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def _array(values: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def regression_metrics(predictions, targets) -> dict[str, float]:
    predicted, expected = _array(predictions), _array(targets)
    if len(predicted) != len(expected) or len(expected) == 0:
        raise ValueError("Predictions and targets must have the same non-zero length")
    if np.std(predicted) == 0 or np.std(expected) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(predicted, expected)[0, 1])
    return {
        "mae": float(mean_absolute_error(expected, predicted)),
        "pearson": correlation,
    }


def mosi_metrics(predictions, targets) -> dict[str, float]:
    """Notebook-compatible weighted F1/Acc-7 plus standard MAE/correlation."""
    predicted, expected = _array(predictions), _array(targets)
    nonzero = expected != 0
    if nonzero.any():
        expected_binary = (expected[nonzero] >= 0).astype(int)
        predicted_binary = (predicted[nonzero] >= 0).astype(int)
        binary_accuracy = float(accuracy_score(expected_binary, predicted_binary))
        weighted_f1 = float(
            f1_score(expected_binary, predicted_binary, average="weighted", zero_division=0)
        )
    else:
        binary_accuracy = weighted_f1 = 0.0
    expected_7 = np.rint(np.clip(expected, -3, 3)).astype(int)
    predicted_7 = np.rint(np.clip(predicted, -3, 3)).astype(int)
    result = regression_metrics(predicted, expected)
    result.update(
        {
            "binary_accuracy": binary_accuracy,
            "weighted_f1": weighted_f1,
            "accuracy_7": float(accuracy_score(expected_7, predicted_7)),
        }
    )
    return result

