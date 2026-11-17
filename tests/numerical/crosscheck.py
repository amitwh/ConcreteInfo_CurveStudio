"""Loader for the stored Statsmodels cross-check fixtures.

The fixtures hold the datasets themselves (x, y, w), so no test depends on a random stream,
and the reference statistics Statsmodels computed for them. They are produced by
``scripts/make_crosscheck_fixtures.py`` and committed under ``tests/data/crosscheck``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from curvestudio.data import XY

FIXTURE_FILE = Path(__file__).resolve().parents[1] / "data" / "crosscheck" / "statsmodels.json"


def load_fixtures() -> dict[str, dict[str, Any]]:
    """Dataset name -> entry (``degree``, ``weighted``, ``x``, ``y``, ``w``, ``fit``, ...)."""
    with FIXTURE_FILE.open(encoding="utf-8") as handle:
        datasets: dict[str, dict[str, Any]] = json.load(handle)["datasets"]
    return datasets


def fixture_xy(name: str, entry: dict[str, Any]) -> XY:
    """The stored dataset as validated fit input (fingerprint ``crosscheck-<name>``)."""
    return XY(
        x=np.asarray(entry["x"], dtype=float),
        y=np.asarray(entry["y"], dtype=float),
        weights=None if entry["w"] is None else np.asarray(entry["w"], dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=f"crosscheck-{name}",
    )
