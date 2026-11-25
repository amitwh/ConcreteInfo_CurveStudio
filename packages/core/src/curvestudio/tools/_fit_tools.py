"""The kit the fit-centric tools share: the target, the grid and the curves drawn on it.

The discernibility report, the feasibility simulator and the next-measurement advisor each
start from a finished fit, and each needs the same few things from it: the fit as one object
with its model and the bounds it ran within (:class:`FitTarget`, :func:`resolve_target`), a
grid over the data's own x range (:func:`evaluation_grid`), the prediction interval's
half-width at that grid (:func:`prediction_half_width`) and the fitted curve itself
(:func:`curve_of`).

A target is never invented. Given a :class:`~curvestudio.workflow.FitBundle` the kit uses
the fit the bundle carries; given a bare :class:`~curvestudio.fitting.FitResult` it uses that
result; given neither it refits the data through
:func:`~curvestudio.workflow.fit_dataset`, one deterministic start and no random restarts,
and says so with ``refitted=True``. Every function is pure: it fits nothing it was not
asked to fit, reads nothing but what it is given, and returns plain NumPy arrays.

``FitTarget.model`` is ``None`` for a polynomial, which has no
:class:`~curvestudio.models.ModelSpec` and is evaluated from its powers;
:func:`curve_of` follows the workflow's own evaluation in either case.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import polynomial as np_polynomial
from numpy.typing import NDArray

from curvestudio.data import XY
from curvestudio.fitting import FitResult
from curvestudio.models import ModelSpec
from curvestudio.statistics import FitView, prediction_band
from curvestudio.workflow import FitBundle, FitSpec, check_spec, fit_dataset

FloatArray = NDArray[np.float64]

DEFAULT_GRID_POINTS = 200
"""How many grid points :func:`evaluation_grid` places over the data's range by default."""

REFIT_NOTE = (
    "The fit was recomputed from the given data with a single deterministic start; it is "
    "not a stored fit."
)
"""What a target built without a bundle says about its fit."""


class CancelledFitTool(Exception):
    """A fit-centric tool was stopped by its cancellation token.

    Private to the tools: a tool catches it where it can return the part it finished and a
    ``cancelled`` warning, and never lets it reach a caller as an error.
    """


@dataclass(frozen=True)
class FitTarget:
    """One fit as every fit-centric tool reads it.

    ``bundle`` is the stored fit when one was given, ``None`` for a refit or a bare result.
    ``model`` is the fitted :class:`~curvestudio.models.ModelSpec`, or ``None`` for a
    polynomial. ``bounds`` maps each parameter to the ``(lower, upper)`` it was fitted
    within, ``None`` on a side that is unbounded. ``refitted`` is true when the kit had to
    fit the data itself, and ``note`` then says so; a target built from a bundle or a given
    result carries no note.
    """

    data: XY
    bundle: FitBundle | None
    model: ModelSpec | None
    fit_spec: FitSpec
    result: FitResult
    bounds: dict[str, tuple[float | None, float | None]]
    refitted: bool
    note: str | None


def _never_cancelled() -> None:
    """The checker of a tool that cannot be cancelled: it has nothing to read."""


def cancel_checker(should_cancel: Callable[[], bool] | None) -> Callable[[], None]:
    """A function that raises :class:`CancelledFitTool` once ``should_cancel`` turns true.

    Without a token the checker is a no-op, so a tool calls it between units of work without
    asking whether it was given one.
    """
    if should_cancel is None:
        return _never_cancelled

    def check() -> None:
        if should_cancel():
            raise CancelledFitTool("the fit tool was cancelled")

    return check


def _bare(value: float) -> float | None:
    """A bound as a target keeps it: ``None`` when infinite, the number otherwise."""
    return value if math.isfinite(value) else None


def _effective_bounds(
    model: ModelSpec | None, spec: FitSpec
) -> dict[str, tuple[float | None, float | None]]:
    """The bounds the fit ran within: the model's declared ones, then the specification's.

    A polynomial declares none, and the workflow refuses a specification that sets one, so
    its map is empty. A side the specification leaves ``None`` is unbounded there; a
    declared infinite bound reads ``None`` the same way.
    """
    bounds: dict[str, tuple[float | None, float | None]] = {}
    if model is not None:
        bounds = {
            parameter.name: (_bare(parameter.lower), _bare(parameter.upper))
            for parameter in model.parameters
        }
    bounds.update(spec.bounds)
    return bounds


def resolve_target(
    data: XY,
    spec: FitSpec,
    result: FitResult | None = None,
    *,
    bundle: FitBundle | None = None,
) -> FitTarget:
    """Resolve the fit a tool works on, refitting only when nothing was given.

    A bundle wins over a bare result. With neither, the data are fitted through
    :func:`~curvestudio.workflow.fit_dataset`, which uses the specification's one start (its
    seed, no random restarts), so the same data and specification always give the same
    numbers.

    Args:
        data: the data the fit was computed from, every row.
        spec: the specification the fit ran with. Its bounds are the ones reported.
        result: a bare fit of ``data`` for ``spec``; ignored when ``bundle`` is given.
        bundle: the stored fit, whose result is used.

    Returns:
        The target with its model, effective bounds and whether it was refitted.

    Raises:
        WorkflowError: from :func:`~curvestudio.workflow.check_spec` when the fit is not the
            fit ``spec`` describes (``MODEL_MISMATCH``, ``UNKNOWN_MODEL``,
            ``INVALID_EQUATION``); from the refit, the codes ``fit_dataset`` raises.
    """
    if bundle is not None:
        fit = bundle.result
        refitted = False
        note = None
    elif result is not None:
        fit = result
        refitted = False
        note = None
    else:
        bundle = fit_dataset(data, spec)
        fit = bundle.result
        refitted = True
        note = REFIT_NOTE
    model = check_spec(fit, spec)
    return FitTarget(
        data=data,
        bundle=bundle,
        model=model,
        fit_spec=spec,
        result=fit,
        bounds=_effective_bounds(model, spec),
        refitted=refitted,
        note=note,
    )


def evaluation_grid(
    data: XY,
    *,
    points: int = DEFAULT_GRID_POINTS,
    low: float | None = None,
    high: float | None = None,
) -> FloatArray:
    """A grid of evenly spaced x values over the data's own range, ends included.

    ``low`` and ``high`` replace the data's minimum and maximum on the sides they are given,
    for a tool the request lets look outside the data (the advisor's allowed range).

    Args:
        data: the data whose x range the grid covers.
        points: how many values to place, 2 or more.
        low: the grid's first value; the data's smallest x when ``None``.
        high: the grid's last value; the data's largest x when ``None``.

    Returns:
        The finite, strictly increasing values, de-duplicated when the range is too narrow to
        hold ``points`` distinct numbers. It is a fresh array.

    Raises:
        ValueError: ``points`` is not an int of at least 2; the data hold fewer than 2 usable
            x values; a bound is not finite; or ``low`` is not below ``high``.
    """
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError(f"a grid needs at least 2 points, got {points!r}")
    xs = np.asarray(data.x, dtype=np.float64)
    usable = xs[np.isfinite(xs)]
    if usable.size < 2:
        raise ValueError(
            f"a grid needs at least 2 usable x values, and the data hold {usable.size}"
        )
    start = float(usable.min()) if low is None else float(low)
    end = float(usable.max()) if high is None else float(high)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"the grid's range must be finite, got {start!r} to {end!r}")
    if start >= end:
        raise ValueError(f"the grid's range needs low < high, got {start!r} to {end!r}")
    grid = np.linspace(start, end, points)
    return np.unique(grid[np.isfinite(grid)])


def prediction_half_width(view: FitView, x: FloatArray, level: float) -> FloatArray:
    """The pointwise prediction interval's half-width at each ``x``, at ``level``.

    The interval is the shipped :func:`~curvestudio.statistics.prediction_band`'s, read on
    the view's active points: ``s2`` and the fit's residual degrees of freedom are the ones
    the fit used, and a new observation has weight 1.

    Raises:
        StatisticsError: ``INVALID_LEVEL`` (``level`` not in (0, 1)); ``INVALID_X`` (empty,
            non-1-D, non-finite ``x``, or a model that is not finite there);
            ``NO_DEGREES_OF_FREEDOM`` / ``NO_COVARIANCE`` (the fit gives no uncertainty).
    """
    band = prediction_band(view, np.asarray(x, dtype=np.float64), level)
    upper = np.asarray(band.upper, dtype=np.float64)
    lower = np.asarray(band.lower, dtype=np.float64)
    return (upper - lower) / 2.0


def curve_of(target: FitTarget, x: FloatArray) -> FloatArray:
    """The target's fitted curve at ``x``.

    A registered or custom model is evaluated by
    :meth:`~curvestudio.models.ModelSpec.evaluate` at the fitted values, in the model's
    parameter order. A polynomial is evaluated from its powers, as
    :func:`~curvestudio.fitting.fit_polynomial` does (``b0 + b1*x + ...``, or from ``b1`` for
    a fit through the origin).

    Raises:
        ValueError: the fit carries no parameter values (it failed before estimating any).
        KeyError: the model declares a parameter the fit did not estimate.
    """
    points = np.asarray(x, dtype=np.float64)
    values = {parameter.name: parameter.value for parameter in target.result.parameters}
    if not values:
        raise ValueError("the fit carries no parameter values to evaluate")
    if target.model is not None:
        fitted = {parameter.name: values[parameter.name] for parameter in target.model.parameters}
        return np.asarray(target.model.evaluate(points, **fitted), dtype=np.float64)
    powers = np.array([int(name.removeprefix("b")) for name in values], dtype=np.intp)
    coefficients = np.zeros(int(powers.max()) + 1)
    coefficients[powers] = list(values.values())
    return np.asarray(np_polynomial.polyval(points, coefficients), dtype=np.float64)
