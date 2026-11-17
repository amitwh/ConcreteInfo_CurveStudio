"""Declarative initial-guess rules: say what each parameter starts at in one word.

Hand-writing an ``initial_guess`` function for every model in a large catalogue is
repetitive. :func:`rules_guess` builds one from a mapping ``parameter -> rule`` where a rule
is either a number or a named summary of the data (:data:`GuessRule`).

The guess never raises and never returns NaN or infinity, whatever the data: non-finite
points are ignored, and when a summary is not defined (no points, constant ``y``) a rule
falls back to a finite value derived from the data scale. When parameter bounds are
supplied, every value is moved strictly inside them, so the engine never starts on a
bound.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

from curvestudio.models.spec import Parameter

FloatArray = NDArray[np.float64]

RuleName = Literal[
    "y_max",
    "y_min",
    "y_range",
    "y_mean",
    "y_first",
    "y_last",
    "x_min",
    "x_max",
    "x_median",
    "x_span",
    "inv_x_span",
    "x_at_half_y",
]
GuessRule = RuleName | float
"""A named data summary or a literal number. Meanings (all on finite points only):

``y_max`` / ``y_min`` / ``y_mean``: largest, smallest and mean ``y`` (0 without data).
``y_range``: ``ptp(y)``; when ``y`` is constant, ``max |y|``, then 1.
``y_first`` / ``y_last``: ``y`` at the smallest and the largest ``x`` (0 without data).
``x_min`` / ``x_max`` / ``x_median``: smallest, largest and median ``x`` (0 without data).
``x_span``: ``ptp(x)``; when ``x`` is constant, ``max |x|``, then 1.
``inv_x_span``: ``1 / ptp(x)``; when ``x`` is constant, ``1 / max |x|``, then 1.
``x_at_half_y``: the ``x`` where the curve first crosses the middle of its ``y`` range
(linear interpolation between neighbouring points); the median of ``x`` if it never does.
"""

_INTERIOR = 1e-6
"""Relative distance by which a guess is kept away from a finite bound."""

_RULE_NAMES = frozenset(get_args(RuleName))


def rules_guess(
    rules: Mapping[str, GuessRule],
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """Build an ``initial_guess(x, y)`` from per-parameter rules.

    Args:
        rules: parameter name to rule (see :data:`GuessRule`).
        bounds: optional ``name -> (lower, upper)``; values are moved strictly inside.
            Parameters missing from ``bounds`` are unbounded.

    Raises:
        ValueError: a rule is not a known name or a finite number, or ``bounds`` names a
            parameter that has no rule or has lower above upper.
    """
    frozen = dict(rules)
    for name, rule in frozen.items():
        if isinstance(rule, str):
            if rule not in _RULE_NAMES:
                raise ValueError(
                    f"unknown guess rule {rule!r} for {name!r}; use one of {sorted(_RULE_NAMES)} "
                    "or a number"
                )
        elif isinstance(rule, bool) or not math.isfinite(rule):
            raise ValueError(f"guess rule for {name!r} must be a rule name or a finite number")
    limits = dict(bounds or {})
    for name, (low, high) in limits.items():
        if name not in frozen:
            raise ValueError(f"bounds given for {name!r}, which has no guess rule")
        if not low <= high:
            raise ValueError(f"bounds for {name!r} have lower above upper")

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _sorted_finite(x, y)
        summary = _Summary(xs, ys)
        out: dict[str, float] = {}
        for name, rule in frozen.items():
            value = float(rule) if not isinstance(rule, str) else summary.rule(rule)
            low, high = limits.get(name, (-math.inf, math.inf))
            out[name] = _strictly_inside(value, low, high)
        return out

    return guess


def guess_from_rules(
    parameters: Sequence[Parameter], rules: Mapping[str, GuessRule]
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """:func:`rules_guess` with the bounds taken from the spec's own ``Parameter`` objects.

    ``rules`` must name exactly the parameters, so a renamed or forgotten parameter fails
    when the model is defined rather than as a missing key during a fit.

    Raises:
        ValueError: ``rules`` and ``parameters`` do not name the same parameters, or a
            rule is invalid (see :func:`rules_guess`).
    """
    names = [p.name for p in parameters]
    if set(rules) != set(names):
        raise ValueError(
            f"guess rules must name exactly the parameters {names}; got {sorted(rules)}"
        )
    return rules_guess(
        {n: rules[n] for n in names}, {p.name: (p.lower, p.upper) for p in parameters}
    )


def _sorted_finite(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    xa, ya = xa[:n], ya[:n]
    keep = np.isfinite(xa) & np.isfinite(ya)
    order = np.argsort(xa[keep], kind="stable")
    return xa[keep][order], ya[keep][order]


class _Summary:
    """Data summaries on sorted finite points; every one is a finite float."""

    def __init__(self, x: FloatArray, y: FloatArray) -> None:
        self.x = x
        self.y = y

    def rule(self, name: str) -> float:
        match name:
            case "y_max":
                return _finite(float(np.max(self.y)) if self.y.size else 0.0)
            case "y_min":
                return _finite(float(np.min(self.y)) if self.y.size else 0.0)
            case "y_mean":
                return _finite(_mean(self.y))
            case "y_range":
                return _extent(self.y)
            case "y_first":
                return _finite(float(self.y[0]) if self.y.size else 0.0)
            case "y_last":
                return _finite(float(self.y[-1]) if self.y.size else 0.0)
            case "x_min":
                return _finite(float(self.x[0]) if self.x.size else 0.0)
            case "x_max":
                return _finite(float(self.x[-1]) if self.x.size else 0.0)
            case "x_median":
                return _finite(float(np.median(self.x)) if self.x.size else 0.0)
            case "x_span":
                return _extent(self.x)
            case "inv_x_span":
                return _finite(1.0 / _extent(self.x), 1.0)
            case _:  # "x_at_half_y": the only name left after validation
                return self._x_at_half_y()

    def _x_at_half_y(self) -> float:
        if self.y.size < 2:
            return _finite(float(np.median(self.x)) if self.x.size else 0.0)
        low, high = float(np.min(self.y)), float(np.max(self.y))
        level = low / 2.0 + high / 2.0
        for i in range(self.y.size - 1):
            y0, y1 = float(self.y[i]), float(self.y[i + 1])
            if (y0 - level) * (y1 - level) <= 0.0 and y0 != y1:
                t = (level - y0) / (y1 - y0)
                x0, x1 = float(self.x[i]), float(self.x[i + 1])
                return _finite(x0 + t * (x1 - x0), float(np.median(self.x)))
        return _finite(float(np.median(self.x)))


def _finite(value: float, fallback: float = 0.0) -> float:
    return value if math.isfinite(value) else fallback


def _mean(values: FloatArray) -> float:
    if not values.size:
        return 0.0
    # Scaling first keeps the sum from overflowing for values near the float limit.
    top = float(np.max(np.abs(values)))
    if top == 0.0:
        return 0.0
    return float(np.mean(values / top)) * top


def _extent(values: FloatArray) -> float:
    """Spread of ``values``; ``max |values|`` when constant, then 1; always finite and > 0."""
    if values.size:
        spread = float(np.max(values)) / 2.0 - float(np.min(values)) / 2.0
        spread *= 2.0  # halving first avoids overflow of max - min near the float limit
        if math.isfinite(spread) and spread > 0.0:
            return spread
        top = float(np.max(np.abs(values)))
        if top > 0.0:
            return top
    return 1.0


def _strictly_inside(value: float, low: float, high: float) -> float:
    """``value`` moved strictly inside ``(low, high)``; always finite."""
    if not math.isfinite(value):
        value = 0.0
    if math.isinf(low) and math.isinf(high):
        return value
    if math.isfinite(low) and math.isfinite(high):
        if low == high:
            return low
        width = high / 2.0 - low / 2.0  # half the width, safe from overflow
        margin = _INTERIOR * 2.0 * width
        if not math.isfinite(margin) or low + margin >= high - margin:
            return low / 2.0 + high / 2.0
        return min(max(value, low + margin), high - margin)
    if math.isfinite(low):
        scale = max(abs(low), abs(value)) or 1.0
        return max(value, low + _INTERIOR * scale)
    scale = max(abs(high), abs(value)) or 1.0
    return min(value, high - _INTERIOR * scale)
