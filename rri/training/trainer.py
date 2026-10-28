"""Small generic supervised trainer used for clean pretraining and smoke runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


class SupervisedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.model.to(self.device)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            features: Mapping[str, torch.Tensor] = batch["features"]
            targets = batch["targets"].to(self.device)
            features = {name: value.to(self.device) for name, value in features.items()}
            self.optimizer.zero_grad()
            predictions = self.model.predict(features)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total_samples += len(targets)
        return total_loss / max(total_samples, 1)

