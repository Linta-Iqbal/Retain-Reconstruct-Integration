"""Common corruption validation and deterministic random-number helpers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def validate_severity(severity: float, *, maximum: float = 1.0) -> float:
    value = float(severity)
    if not 0.0 <= value <= maximum:
        raise ValueError(f"Severity must be in [0, {maximum}], got {value}")
    return value


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def stable_rng(base_seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(base_seed, *parts))

