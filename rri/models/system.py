"""Composition of fusion, reconstructors, and sample-level gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .fusion import ConcatFusion, PairwiseFusionClassifier
from .gates import SampleAlphaGate
from .reconstructors import CorruptionAwareReconstructor


def mix_features(
    observed: torch.Tensor,
    rebuilt: torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    mixed = alpha * observed + (1.0 - alpha) * rebuilt
    return F.normalize(mixed, dim=-1) if normalize else mixed


class RetainRebuildSystem(nn.Module):
    """Dataset-independent retain/rebuild system over named modalities."""

    def __init__(
        self,
        modality_order: Sequence[str],
        fusion: nn.Module,
        reconstructors: Mapping[str, nn.Module],
        gates: Mapping[str, nn.Module],
        *,
        normalize_hybrid: bool = True,
    ) -> None:
        super().__init__()
        self.modality_order = tuple(modality_order)
        if set(reconstructors) != set(self.modality_order):
            raise ValueError("Every modality must have exactly one reconstructor")
        if set(gates) != set(self.modality_order):
            raise ValueError("Every modality must have exactly one gate")
        self.fusion = fusion
        self.reconstructors = nn.ModuleDict(dict(reconstructors))
        self.gates = nn.ModuleDict(dict(gates))
        self.normalize_hybrid = normalize_hybrid

    def predict(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.modality_order).difference(features)
        if missing:
            raise KeyError(f"Missing modality features: {sorted(missing)}")
        return self.fusion(*(features[name] for name in self.modality_order))

    def survivor_features(
        self, modality: str, features: Mapping[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        if modality not in self.modality_order:
            raise KeyError(f"Unknown modality: {modality}")
        return [features[name] for name in self.modality_order if name != modality]

    def rebuild(
        self,
        modality: str,
        observed: torch.Tensor,
        features: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        survivors = self.survivor_features(modality, features)
        return self.reconstructors[modality](observed, *survivors)

    def hybridize(
        self,
        modality: str,
        observed: torch.Tensor,
        features: Mapping[str, torch.Tensor],
        *,
        detach_reconstructor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_reconstructor:
            with torch.no_grad():
                rebuilt = self.rebuild(modality, observed, features)
            rebuilt = rebuilt.detach()
        else:
            rebuilt = self.rebuild(modality, observed, features)
        survivors = self.survivor_features(modality, features)
        alpha = self.gates[modality](observed, rebuilt, *survivors)
        hybrid = mix_features(
            observed, rebuilt, alpha, normalize=self.normalize_hybrid
        )
        return hybrid, alpha, rebuilt

    def forward_corrupted(
        self,
        features: Mapping[str, torch.Tensor],
        corrupted_modality: str,
        corrupted_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hybrid, alpha, _ = self.hybridize(
            corrupted_modality, corrupted_feature, features
        )
        fused_features = dict(features)
        fused_features[corrupted_modality] = hybrid
        return self.predict(fused_features), alpha

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        return self.predict(features)


def build_system(
    modality_order: Sequence[str],
    feature_dims: Mapping[str, int],
    *,
    task_type: str,
    output_dim: int,
    fusion_hidden_dim: int = 128,
    reconstructor_hidden_dim: int = 256,
    gate_hidden_dim: int = 128,
    dropout: float = 0.2,
    normalize_representations: bool = True,
) -> RetainRebuildSystem:
    """Build the notebook architecture from a compact model specification."""
    modalities = tuple(modality_order)
    missing = set(modalities).difference(feature_dims)
    if missing:
        raise KeyError(f"Missing feature dimensions: {sorted(missing)}")
    dimensions = [int(feature_dims[name]) for name in modalities]
    if task_type == "classification" and len(modalities) == 2:
        fusion: nn.Module = PairwiseFusionClassifier(
            dimensions[0], dimensions[1], output_dim, fusion_hidden_dim
        )
    else:
        fusion = ConcatFusion(
            dimensions,
            output_dim,
            fusion_hidden_dim,
            dropout,
            squeeze_output=task_type == "regression" and output_dim == 1,
        )
    reconstructors: dict[str, nn.Module] = {}
    gates: dict[str, nn.Module] = {}
    for modality in modalities:
        survivor_dims = [feature_dims[name] for name in modalities if name != modality]
        reconstructors[modality] = CorruptionAwareReconstructor(
            feature_dims[modality],
            survivor_dims,
            hidden_dim=reconstructor_hidden_dim,
            dropout=dropout if len(modalities) > 2 else 0.0,
            normalize_output=normalize_representations,
        )
        gates[modality] = SampleAlphaGate(
            feature_dims[modality], survivor_dims, gate_hidden_dim
        )
    return RetainRebuildSystem(
        modalities,
        fusion,
        reconstructors,
        gates,
        normalize_hybrid=normalize_representations,
    )

