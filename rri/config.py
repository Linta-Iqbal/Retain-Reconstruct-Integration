"""YAML configuration loading with defaults and command-line overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides using YAML value parsing."""
    result = copy.deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must have KEY=VALUE form: {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"Invalid override key: {dotted_key!r}")
        cursor = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"Cannot assign below non-mapping key: {dotted_key!r}")
            cursor = child
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return result


def load_config(
    dataset_config: str | Path,
    *,
    default_config: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Load a dataset config, merge defaults, expand environment variables."""
    dataset_path = Path(dataset_config).resolve()
    if default_config is None:
        candidate = dataset_path.with_name("default.yaml")
        default_path = candidate if candidate.exists() else None
    else:
        default_path = Path(default_config).resolve()

    config: dict[str, Any] = {}
    if default_path is not None:
        config = _read_yaml(default_path)
    config = deep_merge(config, _read_yaml(dataset_path))
    config = apply_overrides(config, overrides)
    config = _expand_environment(config)
    config["_meta"] = {
        "config_path": str(dataset_path),
        "default_config_path": str(default_path) if default_path else None,
    }
    return config


def resolve_device(requested: str = "auto"):
    """Resolve ``auto``/``cpu``/``cuda`` into a torch device."""
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device

