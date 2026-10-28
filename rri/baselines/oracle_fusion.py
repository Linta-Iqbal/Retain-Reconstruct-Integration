"""Evaluation-label oracle over modality fusion weights."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class OracleFusionResult:
    first_weight: float
    second_weight: float
    score: float
    predictions: torch.Tensor


@torch.no_grad()
def oracle_fusion_weights(
    fusion,
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    targets: torch.Tensor,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float],
    *,
    weight_grid: Iterable[float] = np.linspace(0.0, 2.0, 41),
    maximize: bool = True,
) -> OracleFusionResult:
    """Search evaluation labels; reported results must be marked as an oracle."""
    fusion.eval()
    first_projected, second_projected = fusion.project(first_features, second_features)
    weights = [float(value) for value in weight_grid]
    if not weights:
        raise ValueError("weight_grid must contain at least one value")
    best: OracleFusionResult | None = None
    for first_weight in weights:
        for second_weight in weights:
            predictions = fusion.fuse_projected(
                first_projected, second_projected, first_weight, second_weight
            )
            score = float(score_fn(predictions, targets))
            candidate = OracleFusionResult(
                first_weight, second_weight, score, predictions.detach()
            )
            if best is None or (score > best.score if maximize else score < best.score):
                best = candidate
    assert best is not None
    return best

