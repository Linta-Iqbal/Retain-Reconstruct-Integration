"""Visual/audio sequence corruptions for MOSI."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .base import stable_rng, validate_severity


def gaussian_sequence_noise(
    values: torch.Tensor,
    lengths: torch.Tensor,
    severity: float,
    *,
    seed: int | None = None,
    sample_indices: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale Gaussian noise by each sample's temporal feature standard deviation."""
    severity = validate_severity(severity)
    output = values.clone()
    for batch_index, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        if length <= 0:
            raise ValueError("Sequence lengths must be positive")
        clean = values[batch_index, :length]
        scale = clean.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        if seed is None:
            noise = torch.randn_like(clean)
        else:
            sample_index = (
                batch_index
                if sample_indices is None
                else int(sample_indices[batch_index])
            )
            rng = stable_rng(seed, sample_index)
            noise = torch.from_numpy(
                rng.normal(0.0, 1.0, size=tuple(clean.shape)).astype(np.float32)
            ).to(device=clean.device, dtype=clean.dtype)
        output[batch_index, :length] = clean + noise * scale * severity
    return output

