"""Whitespace-word masking used in image/text and MOSI experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import stable_rng, validate_severity


def mask_words(
    text: str,
    rate: float,
    rng: np.random.Generator,
    replacement_token: str = "[MASK]",
    *,
    keep_one: bool = False,
) -> str:
    rate = validate_severity(rate)
    words = text.split()
    if rate == 0 or not words:
        return text
    count = max(1, min(int(round(rate * len(words))), len(words)))
    if keep_one and len(words) > 1:
        count = min(count, len(words) - 1)
    selected = set(int(index) for index in rng.choice(len(words), size=count, replace=False))
    return " ".join(
        replacement_token if index in selected else word
        for index, word in enumerate(words)
    )


def mask_word_batch(
    word_lists: Sequence[Sequence[str]],
    rate: float,
    replacement_token: str,
    *,
    seed: int | None = None,
    sample_indices: Sequence[int] | None = None,
) -> list[list[str]]:
    """Independently mask words while retaining at least one original word."""
    rate = validate_severity(rate)
    result: list[list[str]] = []
    for batch_index, words_value in enumerate(word_lists):
        words = list(words_value)
        if not words or rate == 0:
            result.append(words)
            continue
        sample_index = batch_index if sample_indices is None else int(sample_indices[batch_index])
        rng = stable_rng(seed or 0, sample_index) if seed is not None else np.random.default_rng()
        masked = [replacement_token if rng.random() < rate else word for word in words]
        if all(word == replacement_token for word in masked):
            retained = int(rng.integers(0, len(words)))
            masked[retained] = words[retained]
        result.append(masked)
    return result

