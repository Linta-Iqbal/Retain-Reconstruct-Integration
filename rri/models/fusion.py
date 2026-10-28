"""Fusion heads shared across classification and regression experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ModalityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PairwiseFusionClassifier(nn.Module):
    """Project two modalities and classify their values plus interaction."""

    def __init__(
        self, first_dim: int, second_dim: int, num_classes: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.first_head = ModalityHead(first_dim, hidden_dim)
        self.second_head = ModalityHead(second_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def project(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.first_head(first), self.second_head(second)

    def fuse_projected(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        first_weight: float | torch.Tensor = 1.0,
        second_weight: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        weighted_first = first_weight * first
        weighted_second = second_weight * second
        joint = torch.cat(
            [weighted_first, weighted_second, weighted_first * weighted_second], dim=-1
        )
        return self.classifier(joint)

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        first_weight: float | torch.Tensor = 1.0,
        second_weight: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        first_projected, second_projected = self.project(first, second)
        return self.fuse_projected(
            first_projected, second_projected, first_weight, second_weight
        )


FusionNet = PairwiseFusionClassifier


class ConcatFusion(nn.Module):
    """Concatenation fusion for arbitrary modality counts and task outputs."""

    def __init__(
        self,
        input_dims: Sequence[int],
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        *,
        squeeze_output: bool = False,
    ) -> None:
        super().__init__()
        self.squeeze_output = squeeze_output
        self.net = nn.Sequential(
            nn.Linear(sum(input_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, output_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, output_dim), output_dim),
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        output = self.net(torch.cat(features, dim=-1))
        return output.squeeze(-1) if self.squeeze_output and output.shape[-1] == 1 else output

