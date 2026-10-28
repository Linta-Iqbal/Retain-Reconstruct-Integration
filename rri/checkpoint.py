"""Portable and atomic checkpoint utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_VERSION = 1


def make_checkpoint(
    *,
    model: torch.nn.Module,
    epoch: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    schedulers: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "config": dict(config),
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
    }
    if optimizers:
        payload["optimizer_state_dicts"] = {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }
    if schedulers:
        payload["scheduler_state_dicts"] = {
            name: scheduler.state_dict() for name, scheduler in schedulers.items()
        }
    return payload


def save_checkpoint(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a checkpoint so interrupted saves do not corrupt it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not an RRI checkpoint: {path}")
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    schedulers: Mapping[str, Any] | None = None,
) -> None:
    for name, optimizer in (optimizers or {}).items():
        state = checkpoint.get("optimizer_state_dicts", {}).get(name)
        if state is not None:
            optimizer.load_state_dict(state)
    for name, scheduler in (schedulers or {}).items():
        state = checkpoint.get("scheduler_state_dicts", {}).get(name)
        if state is not None:
            scheduler.load_state_dict(state)

