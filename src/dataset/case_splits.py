"""Dependency-light case-level split manifest validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from src.dataset.case_ids import normalize_case_id


def load_case_splits(
    path: str | Path, *, case_id_mode: str = "literal"
) -> dict[str, tuple[str, ...]]:
    """Load disjoint, non-empty train/validation/test case IDs from JSON.

    The supplied ImageCAS split manifests store source paths rather than case
    IDs. ``case_id_mode="imagecas_numeric"`` extracts the physical case number
    from both RCA filenames and LCA grouped-directory paths.
    """

    split_path = Path(path)
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Split manifest {split_path} must contain a JSON object.")
    required = ("train", "val", "test")
    result: dict[str, tuple[str, ...]] = {}
    for split in required:
        values = raw.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Split manifest {split_path} requires a non-empty {split!r} list."
            )
        try:
            case_ids = tuple(
                normalize_case_id(value, mode=case_id_mode) for value in values
            )
        except (TypeError, UnicodeError, ValueError) as error:
            raise ValueError(
                f"Split {split!r} in {split_path} contains an invalid case ID: "
                f"{error}"
            ) from error
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"Split {split!r} contains duplicate case IDs.")
        result[split] = case_ids

    for left_index, left_name in enumerate(required):
        for right_name in required[left_index + 1 :]:
            overlap = sorted(set(result[left_name]) & set(result[right_name]))
            if overlap:
                raise ValueError(
                    f"Case leakage between {left_name!r} and {right_name!r}: "
                    + ", ".join(overlap)
                )
    return result
