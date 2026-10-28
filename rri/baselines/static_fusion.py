"""Fixed modality-weight fusion baseline from the 1.* notebooks."""

from __future__ import annotations

import torch


@torch.no_grad()
def static_fusion_predictions(
    fusion: torch.nn.Module,
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    *,
    first_weight: float = 1.0,
    second_weight: float = 1.0,
) -> torch.Tensor:
    fusion.eval()
    return fusion(first_features, second_features, first_weight, second_weight)

