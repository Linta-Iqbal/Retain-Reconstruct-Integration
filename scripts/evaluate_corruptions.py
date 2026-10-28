#!/usr/bin/env python3
"""Evaluate static fusion and the sample gate over stored corruption grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rri.checkpoint import load_checkpoint
from rri.config import load_config, resolve_device
from rri.evaluation import ResultRecord, evaluate_system, write_results
from rri.models import build_system


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config, overrides=args.set)
    device = resolve_device(config.get("device", "auto"))
    bundle = torch.load(config["data"]["feature_bundle"], map_location="cpu", weights_only=False)
    split = bundle["test"]
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
    base = {name: value.to(device) for name, value in split["features"].items()}
    targets = split["targets"].to(device)
    records: list[ResultRecord] = []
    dataset_name = config["dataset"]["name"]
    seed = int(config["seed"])

    for modality, levels in split["corruptions"].items():
        for raw_level, raw_corrupted in sorted(levels.items(), key=lambda item: float(item[0])):
            level = float(raw_level)
            corrupted = raw_corrupted.to(device)
            static_features = dict(base)
            static_features[modality] = corrupted
            static_metrics, _, _ = evaluate_system(
                system, static_features, targets, task_type=config["task"]["type"]
            )
            hybrid_metrics, _, _ = evaluate_system(
                system,
                base,
                targets,
                task_type=config["task"]["type"],
                corrupted_modality=modality,
                corrupted_feature=corrupted,
            )
            corruption_name = config["corruption"].get(modality, "unknown")
            for method, metrics in (
                ("static_fusion", static_metrics),
                ("sample_gate", hybrid_metrics),
            ):
                for metric, value in metrics.items():
                    records.append(
                        ResultRecord(
                            dataset_name,
                            "test",
                            method,
                            modality,
                            corruption_name,
                            level,
                            seed,
                            metric,
                            float(value),
                            str(checkpoint_path),
                        )
                    )
    output = Path(
        args.output or Path(config["output"]["results_dir"]) / "results.csv"
    )
    write_results(records, output)
    print(output)


if __name__ == "__main__":
    main()

