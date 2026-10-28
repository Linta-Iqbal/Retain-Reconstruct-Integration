"""Task-aware clean and corrupted evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from rri.metrics import classification_metrics, mosi_metrics
from rri.models import RetainRebuildSystem


def evaluate_predictions(
    predictions: torch.Tensor, targets: torch.Tensor, task_type: str
) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(predictions, targets)
    if task_type == "regression":
        return mosi_metrics(predictions, targets)
    raise ValueError(f"Unsupported task type: {task_type}")


@torch.no_grad()
def evaluate_system(
    system: RetainRebuildSystem,
    features: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    task_type: str,
    corrupted_modality: str | None = None,
    corrupted_feature: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor | None]:
    system.eval()
    device = next(system.parameters()).device
    device_features = {name: value.to(device) for name, value in features.items()}
    if corrupted_modality is None:
        predictions = system.predict(device_features)
        alpha = None
    else:
        if corrupted_feature is None:
            raise ValueError("corrupted_feature is required with corrupted_modality")
        predictions, alpha = system.forward_corrupted(
            device_features, corrupted_modality, corrupted_feature.to(device)
        )
    metrics = evaluate_predictions(predictions, targets.to(device), task_type)
    return metrics, predictions.detach().cpu(), None if alpha is None else alpha.detach().cpu()

