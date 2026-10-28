"""Sample reconstruction losses and constrained loss weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sample_reconstruction_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(predicted, target)


class ReconstructionLossWeights(nn.Module):
    """Fixed weights or positive learned weights constrained to sum to two."""

    def __init__(
        self,
        *,
        fixed: bool = True,
        sample_weight: float = 1.0,
        distribution_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.fixed = fixed
        initial = torch.tensor([sample_weight, distribution_weight], dtype=torch.float32)
        if torch.any(initial <= 0):
            raise ValueError("Reconstruction loss weights must be positive")
        if fixed:
            self.register_buffer("fixed_weights", initial)
        else:
            self.raw_weights = nn.Parameter(torch.log(initial))

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.fixed_weights if self.fixed else 2.0 * torch.softmax(self.raw_weights, dim=0)
        return weights[0], weights[1]

    def forward(
        self, sample_loss: torch.Tensor, distribution_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_weight, distribution_weight = self.weights()
        total = sample_weight * sample_loss + distribution_weight * distribution_loss
        return total, sample_weight, distribution_weight

