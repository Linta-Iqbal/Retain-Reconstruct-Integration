#!/usr/bin/env python3
"""Train an RRI system from a standardized precomputed feature bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rri.checkpoint import make_checkpoint, save_checkpoint
from rri.config import load_config, resolve_device
from rri.models import build_system
from rri.reproducibility import seed_everything
from rri.training import SingleStageTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def load_bundle(path: str | Path) -> dict:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Feature bundle not found: {bundle_path}. See README.md for its schema."
        )
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "train" not in bundle:
        raise ValueError("Feature bundle must contain a 'train' split")
    return bundle


def select_corruptions(split: dict, modalities: list[str], epoch: int) -> dict[str, torch.Tensor]:
    available = split.get("corruptions")
    if not isinstance(available, dict):
        raise KeyError("Training split requires a 'corruptions' mapping")
    selected: dict[str, torch.Tensor] = {}
    for offset, modality in enumerate(modalities):
        levels = available.get(modality)
        if not isinstance(levels, dict) or not levels:
            raise KeyError(f"No training corruptions found for modality {modality!r}")
        nonzero = sorted(
            ((float(level), tensor) for level, tensor in levels.items() if float(level) > 0),
            key=lambda item: item[0],
        )
        if not nonzero:
            raise ValueError(f"Modality {modality!r} has no non-zero corruption levels")
        selected[modality] = nonzero[(epoch + offset) % len(nonzero)][1]
    return selected


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.set)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    bundle = load_bundle(config["data"]["feature_bundle"])
    train_split = bundle["train"]
    features: dict[str, torch.Tensor] = train_split["features"]
    targets: torch.Tensor = train_split["targets"]
    modalities = list(config["dataset"]["modalities"])
    if set(features) != set(modalities):
        raise KeyError(f"Bundle modalities {sorted(features)} != configured modalities {modalities}")

    task_type = config["task"]["type"]
    if task_type == "classification":
        configured_classes = config["dataset"].get("num_classes")
        output_dim = int(configured_classes or (int(targets.max()) + 1))
        task_loss = F.cross_entropy
    elif task_type == "regression":
        output_dim = 1
        task_loss = F.mse_loss
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    feature_dims = {name: int(tensor.shape[-1]) for name, tensor in features.items()}
    model_config = config["model"]
    system = build_system(
        modalities,
        feature_dims,
        task_type=task_type,
        output_dim=output_dim,
        fusion_hidden_dim=int(model_config["fusion_hidden_dim"]),
        reconstructor_hidden_dim=int(model_config["reconstructor_hidden_dim"]),
        gate_hidden_dim=int(model_config["gate_hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.2)),
        normalize_representations=bool(model_config["normalize_representations"]),
    )
    training_config = config["training"]
    trainer = SingleStageTrainer(
        system,
        task_loss=task_loss,
        reconstruction_learning_rate=float(training_config["reconstruction_lr"]),
        classifier_learning_rate=float(training_config["classifier_lr"]),
        fixed_reconstruction_weights=bool(training_config["fixed_reconstruction_weights"]),
        sample_weight=float(training_config["sample_loss_weight"]),
        distribution_weight=float(training_config["distribution_loss_weight"]),
        device=device,
    )

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    results_dir = Path(config["output"]["results_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_task_loss = float("inf")
    epochs = int(training_config["epochs"])
    batch_size = int(config["data"]["train_batch_size"])
    model_spec = {
        "modalities": modalities,
        "feature_dims": feature_dims,
        "task_type": task_type,
        "output_dim": output_dim,
    }

    for epoch in range(1, epochs + 1):
        corrupted = select_corruptions(train_split, modalities, epoch - 1)
        metrics = trainer.train_epoch(features, corrupted, targets, batch_size=batch_size)
        row = {"epoch": epoch, **metrics}
        history.append(row)
        payload = make_checkpoint(
            model=system,
            epoch=epoch,
            config=config,
            metrics=metrics,
            optimizers={
                "reconstruction": trainer.reconstruction_optimizer,
                "classifier": trainer.classifier_optimizer,
            },
            metadata={"model_spec": model_spec},
        )
        if metrics["task"] < best_task_loss:
            best_task_loss = metrics["task"]
            save_checkpoint(payload, checkpoint_dir / "best.pt")
        if epoch == epochs:
            save_checkpoint(payload, checkpoint_dir / "last.pt")
        print(json.dumps(row, sort_keys=True))

    history_path = results_dir / "train_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()

