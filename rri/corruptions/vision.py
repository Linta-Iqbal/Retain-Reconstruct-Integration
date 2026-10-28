"""Raw RGB/depth-rendering image corruptions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import validate_severity


def gaussian_image_noise(
    image: Image.Image, sigma: float, rng: np.random.Generator
) -> Image.Image:
    sigma = validate_severity(sigma)
    if sigma == 0:
        return image.convert("RGB").copy()
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = rng.normal(0.0, sigma, size=values.shape).astype(np.float32)
    corrupted = np.rint(np.clip(values + noise, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(corrupted, mode="RGB")


def salt_and_pepper(
    image: Image.Image, amount: float, rng: np.random.Generator
) -> Image.Image:
    amount = validate_severity(amount)
    clean = image.convert("RGB")
    if amount == 0:
        return clean.copy()
    array = np.asarray(clean, dtype=np.uint8).copy()
    height, width = array.shape[:2]
    count = min(height * width, max(1, int(round(amount * height * width))))
    locations = rng.choice(height * width, size=count, replace=False)
    row_indices, column_indices = np.unravel_index(locations, (height, width))
    salt = rng.random(count) < 0.5
    array[row_indices[salt], column_indices[salt], :] = 255
    array[row_indices[~salt], column_indices[~salt], :] = 0
    return Image.fromarray(array, mode="RGB")

