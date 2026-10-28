"""Feature-distribution matching objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distribution_moment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor | None = None,
    *,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match per-feature means and standard deviations using MSE."""
    if target is not None:
        target_mean = target.mean(dim=0)
        target_std = target.std(dim=0, unbiased=False)
    if target_mean is None or target_std is None:
        raise ValueError("Provide target or both target_mean and target_std")
    predicted_mean = predicted.mean(dim=0)
    predicted_std = predicted.std(dim=0, unbiased=False)
    return F.mse_loss(predicted_mean, target_mean) + F.mse_loss(
        predicted_std, target_std
    )

