"""Case-ID normalisation shared by NPZ discovery and split manifests."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any


VALID_CASE_ID_MODES = ("literal", "imagecas_numeric")
_ARTERY_CASE = re.compile(r"^(?:lca|rca)[_-]?0*(\d+)$", re.IGNORECASE)
_NUMERIC_CASE = re.compile(r"^\d+$")


def normalize_case_id(value: Any, mode: str = "literal") -> str:
    """Return the ID used to join projections, voxels, and split entries.

    ``literal`` preserves the historical behaviour. ``imagecas_numeric``
    accepts numeric IDs, names such as ``rca_0508.npz``, and grouped paths
    such as ``.../lca/23/prefix_02.npz`` and returns an unpadded numeric ID.
    """

    if mode not in VALID_CASE_ID_MODES:
        raise ValueError(
            f"case_id_mode must be one of {VALID_CASE_ID_MODES}, got {mode!r}."
        )
    if isinstance(value, bytes):
        text = value.decode("utf-8").strip()
    else:
        text = str(value).strip()
    if not text:
        raise ValueError("Case ID cannot be empty.")
    if mode == "literal":
        return text

    path = PurePosixPath(text.replace("\\", "/"))
    candidates = (path.stem, path.parent.name, text)
    for candidate in candidates:
        artery_match = _ARTERY_CASE.fullmatch(candidate)
        if artery_match is not None:
            return str(int(artery_match.group(1)))
        if _NUMERIC_CASE.fullmatch(candidate):
            return str(int(candidate))
    raise ValueError(
        "ImageCAS case IDs must be numeric, named like lca_0001/rca_0001, "
        "or stored below a numeric case directory; got " + repr(text)
    )
