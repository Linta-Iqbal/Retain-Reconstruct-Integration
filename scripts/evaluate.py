#!/usr/bin/env python3
"""Evaluate a clean feature split with a trained RRI checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rri.checkpoint import load_checkpoint
from rri.config import load_config, resolve_device
from rri.evaluation import evaluate_system
from rri.models import build_system


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config, overrides=args.set)
    device = resolve_device(config.get("device", "auto"))
    bundle = torch.load(config["data"]["feature_bundle"], map_location="cpu", weights_only=False)
    split = bundle[args.split]
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
    metrics, _, _ = evaluate_system(
        system,
        split["features"],
        split["targets"],
        task_type=config["task"]["type"],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

