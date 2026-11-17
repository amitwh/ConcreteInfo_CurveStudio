"""Shared numeric helpers for the hand-written family modules.

Every family module needs the same few things: clean data, a weighted linear
least-squares solve, a wrapper that turns an estimator into an ``initial_guess``
that is finite and inside the bounds for every input, and small one- and
two-dimensional minimisers used to search a non-linear shape parameter (a
frequency, a pole position, a rate). They live here so the family modules stay
readable; the module is private (its name starts with an underscore) and is
therefore not picked up as a family by the registration machinery.

Only NumPy is used: the catalogue is imported by every command-line start, so it
must stay cheap.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]

__all__ = [
    "FloatArray",
    "bespoke_guess",
    "best_of_grid",
    "clean",
    "domain_check",
    "golden_minimum",
    "lstsq",
    "pattern_minimum",
    "scale",
    "scan_refine",
    "span",
]


def clean(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Finite ``(x, y)`` pairs ordered by ``x``; fresh arrays, the inputs untouched."""
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


def lstsq(
    columns: Sequence[FloatArray], target: FloatArray, weights: FloatArray
) -> FloatArray | None:
    """Weighted linear least squares ``target ~ columns``; ``None`` when not determined.

    The columns are scaled to unit length first, so a column of ones and a column
    of ``x**2`` do not fight over the tolerance of the rank decision.
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


def bespoke_guess(
    parameters: tuple[Parameter, ...], estimate: Estimator
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """``initial_guess`` from an estimator on clean data; values are moved inside the bounds.

    ``estimate`` returns one finite value per parameter for every input (its own
    fallback covers degenerate data); :func:`rules_guess` with literal numbers then
    keeps them strictly inside the parameter bounds.
    """
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = clean(x, y)
        with np.errstate(all="ignore"):  # overflow on extreme data falls back
            estimates = estimate(xs, ys)
        values = {k: (v if math.isfinite(v) else 0.0) for k, v in estimates.items()}
        return rules_guess(values, bounds)(xs, ys)

    return guess


def domain_check(
    *,
    x_positive_for: str | None = None,
    x_non_negative: bool = False,
    x_nonzero_for: str | None = None,
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, plus the restriction ``x`` needs for the model."""

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        if (n := int(np.count_nonzero(~np.isfinite(xa)))) > 0:
            reasons.append(f"x must be finite: found {n} value(s) that are NaN or infinite")
        if (n := int(np.count_nonzero(~np.isfinite(ya)))) > 0:
            reasons.append(f"y must be finite: found {n} value(s) that are NaN or infinite")
        finite_x = xa[np.isfinite(xa)]
        if x_positive_for is not None and (n := int(np.count_nonzero(finite_x <= 0.0))) > 0:
            reasons.append(
                f"x must be strictly positive for {x_positive_for}: found {n} non-positive value(s)"
            )
        if x_nonzero_for is not None and (n := int(np.count_nonzero(finite_x == 0.0))) > 0:
            reasons.append(f"x must not be zero for {x_nonzero_for}: found {n} zero(s)")
        if x_non_negative and (n := int(np.count_nonzero(finite_x < 0.0))) > 0:
            reasons.append(f"x must be non-negative: found {n} negative value(s)")
        return reasons

    return check


def golden_minimum(
    function: Callable[[float], float], low: float, high: float, *, iterations: int = 90
) -> tuple[float, float]:
    """Minimise a unimodal function on ``[low, high]`` by golden-section search.

    Returns ``(argmin, value)``. A non-finite value is treated as "worse than
    anything", so a search never leaves the bracket through a bad point.
    """

    def safe(value: float) -> float:
        return value if math.isfinite(value) else math.inf

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = float(low), float(high)
    if not left < right:
        return left, safe(function(left))
    inner_left = right - ratio * (right - left)
    inner_right = left + ratio * (right - left)
    f_left = safe(function(inner_left))
    f_right = safe(function(inner_right))
    for _ in range(iterations):
        if f_left < f_right:
            right, inner_right, f_right = inner_right, inner_left, f_left
            inner_left = right - ratio * (right - left)
            f_left = safe(function(inner_left))
        else:
            left, inner_left, f_left = inner_left, inner_right, f_right
            inner_right = left + ratio * (right - left)
            f_right = safe(function(inner_right))
        if right - left <= 1e-12 * max(1.0, abs(right)):
            break
    point = 0.5 * (left + right)
    return point, safe(function(point))


def scan_refine(
    function: Callable[[float], float],
    low: float,
    high: float,
    *,
    count: int = 60,
    geometric: bool = True,
) -> tuple[float, float]:
    """Minimise a function of one variable by a scan and a local refinement.

    The scan finds the basin (the function is often sharply peaked, which a
    golden-section search over a wide bracket can miss); the refinement then
    narrows the bracket to the two scan points around the best one and runs a
    golden-section search inside it. The scan is logarithmic by default, which
    suits a scale parameter, and evenly spaced when ``geometric`` is false.
    """
    if not low < high:
        return low, function(low)
    if geometric:
        grid = np.geomspace(max(low, 1e-300), high, count)
    else:
        grid = np.linspace(low, high, count)
    values = [function(float(point)) for point in grid]
    best = int(np.argmin([value if math.isfinite(value) else math.inf for value in values]))
    low = float(grid[max(best - 1, 0)])
    high = float(grid[min(best + 1, count - 1)])
    if not low < high:
        return float(grid[best]), values[best]
    point, value = golden_minimum(function, low, high)
    if value <= values[best]:
        return point, value
    return float(grid[best]), values[best]


def best_of_grid(
    error: Callable[[FloatArray], float], first: FloatArray, second: FloatArray
) -> tuple[FloatArray, float]:
    """Evaluate ``error`` on the cross product of two value lists; return the best point.

    The point is the pair of values that gave the smallest (finite) error.
    """
    best_point = np.array([first[0], second[0]], dtype=np.float64)
    best_value = math.inf
    for one in first:
        for other in second:
            value = error(np.array([one, other], dtype=np.float64))
            if math.isfinite(value) and value < best_value:
                best_value, best_point = value, np.array([one, other], dtype=np.float64)
    return best_point, best_value


def pattern_minimum(
    function: Callable[[FloatArray], float],
    start: FloatArray,
    step: float,
    *,
    rounds: int = 60,
    shrink: float = 0.5,
) -> tuple[FloatArray, float]:
    """Minimise a function of two variables by a shrinking pattern search.

    A two-dimensional search that needs no derivatives and no second library: the
    point is moved along each axis in turn while that helps, and the step shrinks
    when no direction helps any more. Returns ``(point, value)``.
    """
    point = np.asarray(start, dtype=np.float64).copy()
    best = function(point)
    if not math.isfinite(best):
        best = math.inf
    steps = np.full(point.shape, float(step), dtype=np.float64)
    for _ in range(rounds):
        improved = False
        for index in range(point.size):
            for delta in (1.0, -1.0):
                trial = point.copy()
                trial[index] += delta * steps[index]
                value = function(trial)
                if math.isfinite(value) and value < best:
                    best, point, improved = value, trial, True
        if not improved:
            steps = steps * shrink
            if float(np.max(steps)) <= 1e-12:
                break
    return point, best
