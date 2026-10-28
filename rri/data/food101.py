"""Food-101 image/text row loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1", "ISO-8859-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def load_food101_rows(
    csv_path: str | Path,
    image_root: str | Path,
    label_to_id: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame = read_csv_safe(csv_path)
    required = {"Id", "Text", "Label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["Id"] = frame["Id"].astype(str).str.strip()
    frame["Text"] = frame["Text"].fillna("").astype(str)
    frame["Label"] = frame["Label"].astype(str).str.strip()

    mapping = dict(label_to_id or {})
    if not mapping:
        mapping = {label: index for index, label in enumerate(sorted(frame["Label"].unique()))}
    unknown = sorted(set(frame["Label"]) - set(mapping))
    if unknown:
        raise ValueError(f"Labels absent from the training split: {unknown}")

    root = Path(image_root)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        image_path = root / str(row.Label) / str(row.Id)
        rows.append(
            {
                "id": str(row.Id),
                "text": str(row.Text),
                "image_path": image_path,
                "label": mapping[str(row.Label)],
                "label_name": str(row.Label),
            }
        )
    return rows, mapping


def validate_image_paths(rows: list[dict[str, Any]]) -> None:
    missing = [str(row["image_path"]) for row in rows if not Path(row["image_path"]).is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} Food-101 image(s):\n{preview}")

