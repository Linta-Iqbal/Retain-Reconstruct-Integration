"""NYU Depth RGB/depth row loading and deterministic splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"image_id", "rgb_path", "depth_png_path", "scene_label"}


def load_nyud_rows(
    labels_csv: str | Path,
    rgb_root: str | Path,
    depth_root: str | Path,
    label_to_id: Mapping[str, int] | None = None,
    *,
    validate_paths: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame = pd.read_csv(labels_csv)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing columns: {sorted(missing)}")

    mapping = dict(label_to_id or {})
    labels = frame["scene_label"].astype(str).str.strip()
    if not mapping:
        mapping = {label: index for index, label in enumerate(sorted(labels.unique()))}
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Labels absent from the training mapping: {unknown}")

    rgb_base, depth_base = Path(rgb_root), Path(depth_root)
    rows: list[dict[str, Any]] = []
    for values, label_name in zip(frame.itertuples(index=False), labels):
        rgb_path = Path(str(values.rgb_path))
        depth_path = Path(str(values.depth_png_path))
        if not rgb_path.is_absolute():
            rgb_path = rgb_base / rgb_path
        if not depth_path.is_absolute():
            depth_path = depth_base / depth_path
        if validate_paths and (not rgb_path.is_file() or not depth_path.is_file()):
            raise FileNotFoundError(
                f"NYUD files missing for {values.image_id}: rgb={rgb_path}, depth={depth_path}"
            )
        rows.append(
            {
                "id": str(values.image_id),
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "label": mapping[label_name],
                "label_name": label_name,
            }
        )
    return rows, mapping


def split_nyud_rows(
    rows: list[dict[str, Any]], test_size: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [row["label"] for row in rows]
    indices = np.arange(len(rows))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return [rows[index] for index in train_indices], [rows[index] for index in test_indices]


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_depth_as_rgb(path: str | Path, value_max: float | None = None) -> Image.Image:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    original_dtype = array.dtype
    values = np.asarray(array, dtype=np.float32)
    values = np.where(np.isfinite(values), values, 0.0)
    if value_max is not None:
        denominator = float(value_max)
    elif np.issubdtype(original_dtype, np.integer):
        denominator = float(np.iinfo(original_dtype).max)
    else:
        denominator = float(max(values.max(initial=0.0), 1.0))
    if denominator <= 0:
        raise ValueError(f"Invalid depth scaling denominator: {denominator}")
    image_u8 = np.rint(np.clip(values / denominator, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.repeat(image_u8[..., None], 3, axis=-1)
    return Image.fromarray(rgb, mode="RGB")

