"""Feature reconstructors conditioned on surviving modalities."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CorruptionAwareReconstructor(nn.Module):
    """Recover a clean target feature from its observation and survivors."""

    def __init__(
        self,
        target_dim: int,
        survivor_dims: int | Sequence[int],
        output_dim: int | None = None,
        hidden_dim: int = 256,
        *,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(survivor_dims, int):
            survivor_dims = [survivor_dims]
        output_dim = target_dim if output_dim is None else output_dim
        layers: list[nn.Module] = [
            nn.Linear(target_dim + sum(survivor_dims), hidden_dim),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)])
        self.net = nn.Sequential(*layers)
        self.normalize_output = normalize_output

    def forward(self, observed: torch.Tensor, *survivors: torch.Tensor) -> torch.Tensor:
        if not survivors:
            raise ValueError("At least one surviving modality is required")
        rebuilt = self.net(torch.cat((observed, *survivors), dim=-1))
        return F.normalize(rebuilt, dim=-1) if self.normalize_output else rebuilt


Reconstructor = CorruptionAwareReconstructor

