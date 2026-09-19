#!/usr/bin/env python3
"""Evaluate saved AutoCAR volumes against common raw vessel-code targets."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_prediction_directory import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
