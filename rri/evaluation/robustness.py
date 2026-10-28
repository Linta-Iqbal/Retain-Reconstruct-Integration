"""Long-format robustness evaluation records."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ResultRecord:
    dataset: str
    split: str
    method: str
    corrupted_modality: str
    corruption: str
    severity: float
    seed: int
    metric: str
    value: float
    checkpoint: str


def evaluate_corruption_grid(
    corruptions: Mapping[str, Mapping[float | str, torch.Tensor]],
    evaluate: Callable[[str, torch.Tensor], Mapping[str, float]],
    *,
    dataset: str,
    method: str,
    corruption_names: Mapping[str, str],
    seed: int,
    checkpoint: str,
    split: str = "test",
) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for modality, levels in corruptions.items():
        for raw_level, features in sorted(levels.items(), key=lambda pair: float(pair[0])):
            severity = float(raw_level)
            metrics = evaluate(modality, features)
            for metric, value in metrics.items():
                records.append(
                    ResultRecord(
                        dataset=dataset,
                        split=split,
                        method=method,
                        corrupted_modality=modality,
                        corruption=corruption_names.get(modality, "unknown"),
                        severity=severity,
                        seed=seed,
                        metric=metric,
                        value=float(value),
                        checkpoint=checkpoint,
                    )
                )
    return records


def write_results(records: list[ResultRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ResultRecord.__dataclass_fields__)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return destination

