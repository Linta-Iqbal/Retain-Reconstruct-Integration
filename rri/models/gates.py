"""Sample-level observed-versus-reconstructed retention gates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SampleAlphaGate(nn.Module):
    """Predict one observed-feature retention weight for every sample."""

    def __init__(
        self,
        target_dim: int,
        survivor_dims: int | Sequence[int],
        hidden_dim: int = 128,
        *,
        include_disagreement: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(survivor_dims, int):
            survivor_dims = [survivor_dims]
        target_copies = 3 if include_disagreement else 2
        input_dim = target_copies * target_dim + sum(survivor_dims)
        second_hidden = max(hidden_dim // 2, 16)
        self.include_disagreement = include_disagreement
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, observed: torch.Tensor, rebuilt: torch.Tensor, *survivors: torch.Tensor
    ) -> torch.Tensor:
        values = [observed, rebuilt]
        if self.include_disagreement:
            values.append(torch.abs(observed - rebuilt))
        values.extend(survivors)
        return self.net(torch.cat(values, dim=-1))


Gate = SampleAlphaGate

