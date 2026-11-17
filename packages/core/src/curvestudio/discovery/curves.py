"""Curve and band samples of one fitted candidate, so a UI can plot without refitting."""

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from curvestudio.data import XY
from curvestudio.fitting import FitResult
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.models import ModelSpec
from curvestudio.statistics import (
    StatisticsError,
    build_view,
    confidence_band,
    prediction_band,
)
from curvestudio.statistics.view import FitView

FloatArray = NDArray[np.float64]

CURVE_POINTS = 200
CURVE_MARGIN = 0.05
BAND_LEVEL = 0.95


class RowCurve(BaseModel):
    """A fitted curve at ``CURVE_POINTS`` x values with its 95 % bands.

    The x range is the data range plus a 5 % margin on each side, trimmed on a side where
    the model is not defined (``ln(x)`` below zero, say). The band lists are ``None`` when
    the fit has no covariance; ``bands_reason`` then says why. Bands are pointwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: list[FiniteFloat]
    y: list[FiniteFloat]
    confidence_lower: list[FiniteFloat] | None = None
    confidence_upper: list[FiniteFloat] | None = None
    prediction_lower: list[FiniteFloat] | None = None
    prediction_upper: list[FiniteFloat] | None = None
    bands_reason: str | None = None
    notes: tuple[str, ...] = ()


def _grid(view: FitView, x: FloatArray) -> tuple[FloatArray, tuple[str, ...]]:
    """The curve's x values: the data range plus a margin, minus any side the model rejects."""
    low, high = float(x.min()), float(x.max())
    margin = CURVE_MARGIN * (high - low if high > low else max(abs(low), 1.0))
    grid = np.linspace(low - margin, high + margin, CURVE_POINTS)
    with np.errstate(all="ignore"):
        defined = np.isfinite(view.predict(grid)) & np.all(
            np.isfinite(view.jacobian_at(grid)), axis=1
        )
    left_ok = bool(np.all(defined[grid < low]))
    right_ok = bool(np.all(defined[grid > high]))
    if left_ok and right_ok:
        return grid, ()
    notes = tuple(
        f"the curve was not extended past the data on the {side} because the model is "
        "undefined there"
        for side, ok in (("left", left_ok), ("right", right_ok))
        if not ok
    )
    trimmed = np.linspace(
        low - (margin if left_ok else 0.0), high + (margin if right_ok else 0.0), CURVE_POINTS
    )
    return trimmed, notes


def sample_curve(
    data: XY, fit: FitResult, spec: ModelSpec | None
) -> tuple[RowCurve | None, str | None]:
    """Sample ``fit`` (a converged fit of ``data``; ``spec`` is ``None`` for polynomials).

    Returns ``(curve, None)``, or ``(None, reason)`` when the fit cannot be drawn: it did
    not converge, does not match the data, or the model is not finite over the plotted range.
    """
    try:
        view = build_view(data, fit, spec)
    except StatisticsError as exc:
        return None, f"{exc.code}: {exc}"
    grid, notes = _grid(view, view.x)
    with np.errstate(all="ignore"):
        values = view.predict(grid)
    if not bool(np.all(np.isfinite(values))):
        return None, (
            f"the fitted model is not finite (outside its domain) between x={grid[0]:.6g} "
            f"and x={grid[-1]:.6g}"
        )
    base = {"x": grid.tolist(), "y": values.tolist(), "notes": notes}
    try:
        conf = confidence_band(view, grid, BAND_LEVEL)
        pred = prediction_band(view, grid, BAND_LEVEL)
    except StatisticsError as exc:
        return RowCurve(**base, bands_reason=f"{exc.code}: {exc}"), None
    edges = (*conf.lower, *conf.upper, *pred.lower, *pred.upper)
    if not all(math.isfinite(v) for v in edges):
        return RowCurve(**base, bands_reason="the bands are not finite at some curve points"), None
    return (
        RowCurve(
            **base,
            confidence_lower=conf.lower,
            confidence_upper=conf.upper,
            prediction_lower=pred.lower,
            prediction_upper=pred.upper,
        ),
        None,
    )
