"""MVSA image/text row loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .food101 import read_csv_safe


def _resolve_image_path(raw: str, csv_dir: Path, image_root: Path | None) -> Path:
    raw_path = Path(raw)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(csv_dir / raw_path)
        if image_root is not None:
            candidates.append(image_root / raw_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"MVSA image was not found. Tried:\n{tried}")


def load_mvsa_rows(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    label_to_id: Mapping[str, int] | None = None,
    *,
    validate_paths: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame = read_csv_safe(csv_path)
    required = {"id", "text", "image_path", "final_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    for column in ("id", "image_path", "final_label"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["text"] = frame["text"].fillna("").astype(str)

    mapping = dict(label_to_id or {})
    if not mapping:
        mapping = {
            label: index for index, label in enumerate(sorted(frame["final_label"].unique()))
        }
    unknown = sorted(set(frame["final_label"]) - set(mapping))
    if unknown:
        raise ValueError(f"Labels absent from the training split: {unknown}")

    csv_dir = Path(csv_path).resolve().parent
    root = Path(image_root).resolve() if image_root is not None else None
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        path = (
            _resolve_image_path(str(row.image_path), csv_dir, root)
            if validate_paths
            else Path(str(row.image_path))
        )
        rows.append(
            {
                "id": str(row.id),
                "text": str(row.text),
                "image_path": path,
                "label": mapping[str(row.final_label)],
                "label_name": str(row.final_label),
            }
        )
    return rows, mapping

