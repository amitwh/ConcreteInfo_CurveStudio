"""Shared numeric helpers for the engineering families' initial guesses.

This is a private module (the family loader skips names starting with an underscore), so it
is never treated as a family. Everything here is NumPy-only, tolerates degenerate data
(empty, constant, non-finite, huge) without raising, and every estimator built with
:func:`bespoke_guess` returns finite values strictly inside the parameter bounds.

The guesses favour *exact* linearisations where the model has one (log-log, reciprocal,
three-point exponential), so a noiseless example is recovered by a single start. Where no
linearisation exists, a small deterministic grid over the one or two shape parameters picks
the most linear transform and a second, finer stage refines it; nothing is random.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]

_EXP_LIMIT = 700.0
"""Largest exponent passed to ``math.exp`` before it overflows a float."""


# -- data summaries ---------------------------------------------------------------------


def clean(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Finite ``(x, y)`` pairs ordered by ``x``; fresh arrays."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    xa, ya = xa[:n], ya[:n]
    keep = np.isfinite(xa) & np.isfinite(ya)
    order = np.argsort(xa[keep], kind="stable")
    return xa[keep][order], ya[keep][order]


def scale(values: FloatArray) -> float:
    """Largest magnitude of ``values``, or 1 when there is none or all are zero."""
    top = float(np.max(np.abs(values))) if values.size else 0.0
    return top if top > 0.0 else 1.0


def span(x: FloatArray) -> float:
    """Extent of ``x``; its magnitude (or 1) when ``x`` is constant."""
    if x.size and float(np.ptp(x)) > 0.0:
        return float(np.ptp(x))
    return scale(x)


def weighted_line(u: FloatArray, v: FloatArray, weights: FloatArray) -> tuple[float, float] | None:
    """Weighted least squares ``v = intercept + slope * u``; ``None`` if not determined."""
    if u.size < 2:
        return None
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return None
    mean_u = float(np.sum(weights * u)) / total
    mean_v = float(np.sum(weights * v)) / total
    sxx = float(np.sum(weights * (u - mean_u) ** 2))
    if not math.isfinite(sxx) or sxx <= 0.0:
        return None
    slope = float(np.sum(weights * (u - mean_u) * (v - mean_v))) / sxx
    intercept = mean_v - slope * mean_u
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        return None
    return slope, intercept


def log_fit(u: FloatArray, y: FloatArray, sign: float = 1.0) -> tuple[float, float] | None:
    """Fit ``log(sign*y) = intercept + slope*u`` on the points where ``sign*y > 0``.

    The weight ``y**2`` is the delta-method inverse variance of ``log y`` for
    constant-variance noise on ``y``, so near-zero points do not dominate.
    """
    keep = (sign * y > 0.0) & np.isfinite(u)
    if int(np.count_nonzero(keep)) < 2:
        return None
    yk = y[keep]
    return weighted_line(u[keep], np.log(sign * yk), yk**2)


def least_squares(
    columns: Sequence[FloatArray], target: FloatArray, weights: FloatArray
) -> FloatArray | None:
    """Weighted linear least squares ``target ~ columns``; ``None`` if not determined.

    Columns are scaled to unit length first so ``x**2`` and ``1`` do not fight over the
    rank tolerance. The returned coefficients are in the original column units.
    """
    if target.size < len(columns):
        return None
    design = np.column_stack(columns)
    root = np.sqrt(weights)
    with np.errstate(all="ignore"):
        norms = np.linalg.norm(design * root[:, None], axis=0)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
            return None
        scaled = design / norms
        coef, _, rank, _ = np.linalg.lstsq(scaled * root[:, None], target * root, rcond=None)
    if rank < design.shape[1] or not np.all(np.isfinite(coef)):
        return None
    return np.asarray(coef / norms, dtype=np.float64)


def unexplained(x: FloatArray, y: FloatArray, curve: FloatArray) -> float:
    """Fraction of the variance of ``y`` the ``curve`` leaves unexplained; ``inf`` if unusable."""
    if y.size == 0:
        return math.inf
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if not math.isfinite(sst) or sst <= 0.0:
        return math.inf
    with np.errstate(all="ignore"):
        sse = float(np.sum((y - curve) ** 2))
    return sse / sst if math.isfinite(sse) else math.inf


# -- candidate scans --------------------------------------------------------------------


def limits_above(y: FloatArray, *, factor: float = 4.0, steps: int = 32) -> FloatArray:
    """Geometric candidate asymptotes just above ``max(y)``; empty when unusable.

    The first candidate sits ``1 + 1e-6`` above the maximum so an asymptote the data almost
    reach is still reachable; the last is ``factor`` times the maximum, so data far from the
    plateau are covered too.
    """
    if not y.size:
        return np.empty(0)
    top = float(np.max(y))
    if not math.isfinite(top) or top <= 0.0:
        return np.empty(0)
    high = min(factor, 1e300 / top)
    low = top * (1.0 + 1e-6)
    if not high > 1.0:
        return np.empty(0)
    return np.geomspace(low, top * high, steps)


def refine_around(center: float, *, spread: float = 0.12, steps: int = 21) -> FloatArray:
    """A fine geometric scan of ``center`` within ``1 +/- spread`` (empty if unusable)."""
    if not (math.isfinite(center) and center > 0.0):
        return np.empty(0)
    return np.geomspace(center * (1.0 - spread), center * (1.0 + spread), steps)


def three_point_decay(x: FloatArray, y: FloatArray) -> tuple[float, float, float] | None:
    """Fit ``y = limit - amplitude * exp(-rate * x)`` through three interpolated points.

    With ``x1 < x2 < x3`` equally spaced and ``y = limit - amplitude*exp(-rate*x)``, the
    ratios ``(y3-y2)/(y2-y1)`` and ``(y2-y1)/(limit-y1)`` are both ``exp(-rate*h)`` and
    ``1 - exp(-rate*h)``, which gives every parameter in closed form. That makes the guess
    exact for noiseless data and a sound start for noisy data.

    Returns:
        ``(limit, rate, amplitude)``, or ``None`` when the data are too short, not monotone
        in the required way, or would overflow.
    """
    if x.size < 3:
        return None
    x1, x3 = float(x[0]), float(x[-1])
    if not (math.isfinite(x1) and math.isfinite(x3)) or x3 <= x1:
        return None
    middle = x1 / 2.0 + x3 / 2.0
    y1 = float(y[0])
    y3 = float(y[-1])
    y2 = float(np.interp(middle, x, y))
    first = y2 - y1
    second = y3 - y2
    if not (math.isfinite(first) and math.isfinite(second)) or first == 0.0:
        return None
    ratio = second / first
    if not (0.0 < ratio < 1.0):
        return None
    h = (x3 - x1) / 2.0
    limit = y1 + first / (1.0 - ratio)
    rate = -math.log(ratio) / h
    if not (math.isfinite(limit) and math.isfinite(rate)) or rate <= 0.0:
        return None
    if rate * x1 > _EXP_LIMIT:
        return None
    amplitude = (limit - y1) * math.exp(rate * x1)
    if not math.isfinite(amplitude):
        return None
    return limit, rate, amplitude


# -- initial-guess plumbing -------------------------------------------------------------


def bespoke_guess(
    parameters: tuple[Parameter, ...], estimate: Estimator
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """``initial_guess`` from an estimator; values are moved strictly inside the bounds.

    ``estimate`` must compute every parameter from clean finite ``(x, y)`` arrays and must
    not raise. Non-finite estimates become 0 and :func:`rules_guess` moves them inside the
    parameter bounds, so the contract (finite, inside bounds, every name present) holds even
    for empty or degenerate data.
    """
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = clean(x, y)
        with np.errstate(all="ignore"):
            estimates = estimate(xs, ys)
        values: dict[str, float] = {}
        for name in bounds:
            value = float(estimates[name])
            values[name] = value if math.isfinite(value) else 0.0
        return rules_guess(values, bounds)(xs, ys)

    return guess


def domain_check(
    *,
    x_positive_for: str | None = None,
    x_non_negative: bool = False,
    x_below: tuple[str, float] | None = None,
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data plus the model's restrictions on ``x``.

    ``x_positive_for`` names the expression that needs ``x > 0``; ``x_non_negative`` states
    that ``x`` is a physical quantity (age, dosage, content, depth); ``x_below`` is a
    ``(name, limit)`` pair for a fraction that must stay below 1 (a replacement level or a
    paste compacity).
    """

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        bad_x = int(np.count_nonzero(~np.isfinite(xa)))
        bad_y = int(np.count_nonzero(~np.isfinite(ya)))
        if bad_x:
            reasons.append(f"x must be finite: found {_count(bad_x)} that are NaN or infinite")
        if bad_y:
            reasons.append(f"y must be finite: found {_count(bad_y)} that are NaN or infinite")
        finite_x = xa[np.isfinite(xa)]
        if x_positive_for is not None:
            n = int(np.count_nonzero(finite_x <= 0.0))
            if n:
                reasons.append(
                    f"x must be strictly positive for {x_positive_for}: "
                    f"found {_count(n)} that are zero or negative"
                )
        if x_non_negative:
            n = int(np.count_nonzero(finite_x < 0.0))
            if n:
                reasons.append(
                    f"x must be non-negative (a physical quantity): found {_count(n)} negative"
                )
        if x_below is not None:
            what, limit = x_below
            n = int(np.count_nonzero(finite_x >= limit))
            if n:
                reasons.append(
                    f"x must be below {limit:g} for {what}: found {_count(n)} at or above it"
                )
        return reasons

    return check


def _count(n: int) -> str:
    return f"{n} value" if n == 1 else f"{n} values"
