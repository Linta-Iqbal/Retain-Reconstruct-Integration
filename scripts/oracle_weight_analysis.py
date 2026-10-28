#!/usr/bin/env python3
"""Compare learned retention and oracle upper bounds from the 1.* notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rri.baselines import (
    fit_global_retention,
    oracle_fusion_weights,
    oracle_retention_weight,
    retention_predictions,
)
from rri.checkpoint import load_checkpoint
from rri.config import load_config, resolve_device
from rri.evaluation import ResultRecord, evaluate_predictions, write_results
from rri.metrics import accuracy, mosi_metrics
from rri.models import build_system


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def level_tensor(levels: dict, target_level: float) -> torch.Tensor:
    for raw_level, tensor in levels.items():
        if np.isclose(float(raw_level), target_level):
            return tensor
    raise KeyError(f"Training corruption level {target_level} is missing")


def main() -> None:
    args = arguments()
    config = load_config(args.config, overrides=args.set)
    device = resolve_device(config.get("device", "auto"))
    bundle = torch.load(
        config["data"]["feature_bundle"], map_location="cpu", weights_only=False
    )
    train_split, test_split = bundle["train"], bundle["test"]
    checkpoint_path = Path(
        args.checkpoint or Path(config["output"]["checkpoint_dir"]) / "best.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path)
    spec = checkpoint["metadata"]["model_spec"]
    model_config = config["model"]
    system = build_system(
        spec["modalities"],
        spec["feature_dims"],
        task_type=spec["task_type"],
        output_dim=spec["output_dim"],
        fusion_hidden_dim=int(model_config["fusion_hidden_dim"]),
        reconstructor_hidden_dim=int(model_config["reconstructor_hidden_dim"]),
        gate_hidden_dim=int(model_config["gate_hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.2)),
        normalize_representations=bool(model_config["normalize_representations"]),
    ).to(device)
    system.load_state_dict(checkpoint["model_state_dict"])
    system.eval()

    train_features = {
        name: tensor.to(device) for name, tensor in train_split["features"].items()
    }
    test_features = {
        name: tensor.to(device) for name, tensor in test_split["features"].items()
    }
    train_targets = train_split["targets"].to(device)
    test_targets = test_split["targets"].to(device)
    task_type = config["task"]["type"]
    task_loss = F.cross_entropy if task_type == "classification" else F.mse_loss
    score_fn = (
        accuracy
        if task_type == "classification"
        else lambda predictions, targets: mosi_metrics(predictions, targets)["weighted_f1"]
    )

    baseline_config = config["baselines"]
    alpha_grid = np.linspace(
        0.0, 1.0, int(baseline_config["oracle_retention_steps"])
    )
    fusion_grid = np.linspace(
        0.0,
        float(baseline_config["oracle_fusion_max_weight"]),
        int(baseline_config["oracle_fusion_steps"]),
    )
    dataset_name = config["dataset"]["name"]
    seed = int(config["seed"])
    records: list[ResultRecord] = []

    def append_metrics(
        method: str, modality: str, level: float, predictions: torch.Tensor
    ) -> None:
        metrics = evaluate_predictions(predictions, test_targets, task_type)
        for metric, value in metrics.items():
            records.append(
                ResultRecord(
                    dataset_name,
                    "test",
                    method,
                    modality,
                    config["corruption"].get(modality, "unknown"),
                    level,
                    seed,
                    metric,
                    float(value),
                    str(checkpoint_path),
                )
            )

    for modality, test_levels in test_split["corruptions"].items():
        for raw_level, raw_test_observed in sorted(
            test_levels.items(), key=lambda item: float(item[0])
        ):
            level = float(raw_level)
            train_observed = level_tensor(
                train_split["corruptions"][modality], level
            ).to(device)
            test_observed = raw_test_observed.to(device)
            with torch.no_grad():
                train_rebuilt = system.rebuild(
                    modality, train_observed, train_features
                )
                test_rebuilt = system.rebuild(modality, test_observed, test_features)

            def train_predict(
                mixed: torch.Tensor, indices: torch.Tensor
            ) -> torch.Tensor:
                candidate = {
                    name: tensor[indices] for name, tensor in train_features.items()
                }
                candidate[modality] = mixed
                return system.predict(candidate)

            def test_predict(mixed: torch.Tensor) -> torch.Tensor:
                candidate = dict(test_features)
                candidate[modality] = mixed
                return system.predict(candidate)

            learned_alpha = fit_global_retention(
                train_observed,
                train_rebuilt,
                train_targets,
                train_predict,
                task_loss,
                epochs=int(baseline_config["learned_retention_epochs"]),
                learning_rate=float(baseline_config["learned_retention_lr"]),
                modules_to_freeze=[system],
                normalize=system.normalize_hybrid,
            )
            learned_predictions = retention_predictions(
                test_observed,
                test_rebuilt,
                learned_alpha,
                test_predict,
                normalize=system.normalize_hybrid,
            )
            oracle_retention = oracle_retention_weight(
                test_observed,
                test_rebuilt,
                test_targets,
                test_predict,
                score_fn,
                alpha_grid=alpha_grid,
                normalize=system.normalize_hybrid,
            )
            with torch.no_grad():
                static_features = dict(test_features)
                static_features[modality] = test_observed
                static_predictions = system.predict(static_features)
                sample_predictions, _ = system.forward_corrupted(
                    test_features, modality, test_observed
                )

            append_metrics("static_fusion", modality, level, static_predictions)
            append_metrics(
                "learned_global_retention", modality, level, learned_predictions
            )
            append_metrics(
                "oracle_global_retention",
                modality,
                level,
                oracle_retention.predictions,
            )
            append_metrics("sample_gate", modality, level, sample_predictions)
            for method, metric, value in (
                ("learned_global_retention", "retention_alpha", learned_alpha),
                (
                    "oracle_global_retention",
                    "retention_alpha",
                    oracle_retention.alpha,
                ),
            ):
                records.append(
                    ResultRecord(
                        dataset_name,
                        "test",
                        method,
                        modality,
                        config["corruption"].get(modality, "unknown"),
                        level,
                        seed,
                        metric,
                        float(value),
                        str(checkpoint_path),
                    )
                )

            if len(spec["modalities"]) == 2 and hasattr(system.fusion, "project"):
                ordered = dict(test_features)
                ordered[modality] = test_observed
                first, second = (ordered[name] for name in spec["modalities"])
                oracle_fusion = oracle_fusion_weights(
                    system.fusion,
                    first,
                    second,
                    test_targets,
                    score_fn,
                    weight_grid=fusion_grid,
                )
                append_metrics(
                    "oracle_fusion_weights",
                    modality,
                    level,
                    oracle_fusion.predictions,
                )
                for metric, value in (
                    ("first_modality_weight", oracle_fusion.first_weight),
                    ("second_modality_weight", oracle_fusion.second_weight),
                ):
                    records.append(
                        ResultRecord(
                            dataset_name,
                            "test",
                            "oracle_fusion_weights",
                            modality,
                            config["corruption"].get(modality, "unknown"),
                            level,
                            seed,
                            metric,
                            float(value),
                            str(checkpoint_path),
                        )
                    )

    output = Path(
        args.output
        or Path(config["output"]["results_dir"]) / "oracle_weight_analysis.csv"
    )
    write_results(records, output)
    print(output)


if __name__ == "__main__":
    main()

