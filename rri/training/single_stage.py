"""Alternating single-stage training ported from the final 5.* notebooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

from rri.losses import ReconstructionLossWeights, distribution_moment_loss
from rri.models import RetainRebuildSystem


class SingleStageTrainer:
    """Alternate reconstructor updates with fusion/sample-gate updates."""

    def __init__(
        self,
        system: RetainRebuildSystem,
        *,
        task_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reconstruction_learning_rate: float = 1e-3,
        classifier_learning_rate: float = 1e-3,
        fixed_reconstruction_weights: bool = True,
        sample_weight: float = 1.0,
        distribution_weight: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.system = system.to(device)
        self.device = torch.device(device)
        self.task_loss = task_loss
        self.loss_weights = ReconstructionLossWeights(
            fixed=fixed_reconstruction_weights,
            sample_weight=sample_weight,
            distribution_weight=distribution_weight,
        ).to(device)
        reconstruction_parameters = list(self.system.reconstructors.parameters())
        if not fixed_reconstruction_weights:
            reconstruction_parameters += list(self.loss_weights.parameters())
        self.reconstruction_optimizer = torch.optim.Adam(
            reconstruction_parameters, lr=reconstruction_learning_rate
        )
        self.classifier_optimizer = torch.optim.Adam(
            list(self.system.fusion.parameters()) + list(self.system.gates.parameters()),
            lr=classifier_learning_rate,
        )

    def _batch(self, values: Mapping[str, torch.Tensor], indices: torch.Tensor):
        return {name: tensor[indices].to(self.device) for name, tensor in values.items()}

    def train_epoch(
        self,
        clean_features: Mapping[str, torch.Tensor],
        corrupted_features: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        *,
        batch_size: int = 16,
    ) -> dict[str, float]:
        modalities = self.system.modality_order
        if set(clean_features) != set(modalities) or set(corrupted_features) != set(modalities):
            raise KeyError("Clean and corrupted feature maps must match the system modalities")
        sample_count = len(targets)
        if any(len(value) != sample_count for value in clean_features.values()):
            raise ValueError("All feature tensors and targets must have equal length")
        targets = targets.to(self.device)
        statistics = {
            name: (
                values.to(self.device).mean(0),
                values.to(self.device).std(0, unbiased=False).clamp_min(1e-6),
            )
            for name, values in clean_features.items()
        }
        self.system.train()
        totals = {key: 0.0 for key in ("reconstruction", "sample", "distribution", "task")}
        alpha_totals = {name: 0.0 for name in modalities}
        batches = 0
        order = torch.randperm(sample_count)

        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            clean = self._batch(clean_features, indices)
            corrupted = self._batch(corrupted_features, indices)
            batch_targets = targets[indices.to(targets.device)]

            self.reconstruction_optimizer.zero_grad()
            modality_sample_losses = []
            modality_distribution_losses = []
            for modality in modalities:
                clean_rebuilt = self.system.rebuild(modality, clean[modality], clean)
                corrupt_rebuilt = self.system.rebuild(modality, corrupted[modality], clean)
                target = clean[modality]
                modality_sample_losses.append(
                    0.5 * (F.mse_loss(clean_rebuilt, target) + F.mse_loss(corrupt_rebuilt, target))
                )
                target_mean, target_std = statistics[modality]
                modality_distribution_losses.append(
                    0.5
                    * (
                        distribution_moment_loss(
                            clean_rebuilt, target_mean=target_mean, target_std=target_std
                        )
                        + distribution_moment_loss(
                            corrupt_rebuilt, target_mean=target_mean, target_std=target_std
                        )
                    )
                )
            sample_loss = torch.stack(modality_sample_losses).mean()
            distribution_loss = torch.stack(modality_distribution_losses).mean()
            reconstruction_loss, _, _ = self.loss_weights(sample_loss, distribution_loss)
            reconstruction_loss.backward()
            self.reconstruction_optimizer.step()

            self.classifier_optimizer.zero_grad()
            task_terms = []
            for modality in modalities:
                for observed in (clean[modality], corrupted[modality]):
                    hybrid, alpha, _ = self.system.hybridize(
                        modality, observed, clean, detach_reconstructor=True
                    )
                    candidate_features = dict(clean)
                    candidate_features[modality] = hybrid
                    task_terms.append(self.task_loss(self.system.predict(candidate_features), batch_targets))
                alpha_totals[modality] += float(alpha.detach().mean())
            task_loss = torch.stack(task_terms).mean()
            task_loss.backward()
            self.classifier_optimizer.step()

            totals["reconstruction"] += float(reconstruction_loss.detach())
            totals["sample"] += float(sample_loss.detach())
            totals["distribution"] += float(distribution_loss.detach())
            totals["task"] += float(task_loss.detach())
            batches += 1

        metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        metrics.update(
            {f"alpha_{name}": value / max(batches, 1) for name, value in alpha_totals.items()}
        )
        return metrics

