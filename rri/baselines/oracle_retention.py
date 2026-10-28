"""Evaluation-label oracle for the global retain/rebuild coefficient."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from .learned_retention import retention_predictions


@dataclass(frozen=True)
class OracleRetentionResult:
    alpha: float
    score: float
    predictions: torch.Tensor


def oracle_retention_weight(
    observed: torch.Tensor,
    rebuilt: torch.Tensor,
    targets: torch.Tensor,
    predict_from_mixed: Callable[[torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor, torch.Tensor], float],
    *,
    alpha_grid: Iterable[float] = np.linspace(0.0, 1.0, 101),
    maximize: bool = True,
    normalize: bool = True,
) -> OracleRetentionResult:
    """Search evaluation labels; this is an upper bound, never a deployable method."""
    best: OracleRetentionResult | None = None
    for alpha_value in alpha_grid:
        alpha = float(alpha_value)
        predictions = retention_predictions(
            observed, rebuilt, alpha, predict_from_mixed, normalize=normalize
        )
        score = float(score_fn(predictions, targets))
        candidate = OracleRetentionResult(alpha, score, predictions.detach())
        if best is None or (score > best.score if maximize else score < best.score):
            best = candidate
    if best is None:
        raise ValueError("alpha_grid must contain at least one value")
    return best

