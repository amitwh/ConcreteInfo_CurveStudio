"""Sigmoidal curves: symmetric and asymmetric S-shapes between two asymptotes.

Every model is fitted on the original response scale. The guesses exploit one exact
linearisation: for fixed location and scale a logistic-type curve is linear in its two
asymptotes (or in its asymptote and its drift), and a Weibull or Frechet curve is linear in
the same way once its scale and shape are fixed. The remaining shape parameters are found by
a full grid scan followed by a local golden-section polish, which is deterministic and, on
the sixty-point example data, cheap. No expression raises a clamped argument to a free
power, so the analytic Jacobian is finite wherever the model is defined.

``logistic4``, ``boltzmann`` and ``tanh_sigmoid`` are the same curve under three
parametrisations that people search for separately; they are declared in
``EQUIVALENT_PAIRS`` so the catalogue accepts them.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from math import erf

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]
_erf = np.vectorize(erf)
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]
Basis = Callable[[FloatArray, Mapping[str, float]], list[FloatArray]]
Shapes = Mapping[str, tuple[float, float, float]]

_BAYES_BATES = (
    "Bates D.M., Watts D.G. (1988) Nonlinear Regression Analysis and Its Applications, Wiley"
)
_SEBER_WILD = "Seber G.A.F., Wild C.J. (1989) Nonlinear Regression, Wiley"
_RICHARDS = (
    "Richards F.J. (1959) A flexible growth function for empirical use. Journal of Experimental "
    "Botany 10(2):290-301"
)
_WEIBULL = (
    "Weibull W. (1951) A statistical distribution function of wide applicability. Journal of "
    "Applied Mechanics 18(3):293-297"
)
_HILL = (
    "Hill A.V. (1910) The possible effects of the aggregation of the molecules of haemoglobin on "
    "its dissociation curves. Journal of Physiology 40:iv-vii"
)
_BOLTZMANN = (
    "Boltzmann L. (1877) Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen "
    "Wärmetheorie und der Wahrscheinlichkeitsrechnung. Sitzungsberichte der Kaiserlichen "
    "Akademie der Wissenschaften Wien 76:373-435"
)
_ERRF = (
    "Abramowitz M., Stegun I.A. (1964) Handbook of Mathematical Functions with Formulas, Graphs "
    "and Mathematical Tables, National Bureau of Standards Applied Mathematics Series 55"
)
_FRECHET = (
    "Fréchet M. (1927) Sur la loi de probabilité de l'écart maximum. Annales de la Société "
    "Polonaise de Mathématique 6:93-116"
)


# -- numeric helpers --------------------------------------------------------------------


def _clean(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Finite ``(x, y)`` pairs ordered by ``x``; fresh arrays."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    xa, ya = xa[:n], ya[:n]
    keep = np.isfinite(xa) & np.isfinite(ya)
    order = np.argsort(xa[keep], kind="stable")
    return xa[keep][order], ya[keep][order]


def _span(x: FloatArray) -> float:
    """Extent of ``x``; its magnitude (or 1) when ``x`` is constant."""
    if x.size:
        spread = float(np.max(x)) / 2.0 - float(np.min(x)) / 2.0
        spread *= 2.0  # halving first avoids overflow of max - min near the float limit
        if math.isfinite(spread) and spread > 0.0:
            return spread
        top = float(np.max(np.abs(x)))
        if top > 0.0:
            return top
    return 1.0


def _top(y: FloatArray) -> float:
    """Largest ``|y|``, or 0 when there is no data."""
    return float(np.max(np.abs(y))) if y.size else 0.0


def _extent(values: FloatArray) -> float:
    """Spread of ``values``; ``max |values|`` when constant, then 1; always finite and > 0."""
    if values.size:
        spread = float(np.max(values)) / 2.0 - float(np.min(values)) / 2.0
        spread *= 2.0
        if math.isfinite(spread) and spread > 0.0:
            return spread
        top = _top(values)
        if top > 0.0:
            return top
    return 1.0


def _ymin(y: FloatArray) -> float:
    return float(np.min(y)) if y.size else 0.0


def _ymax(y: FloatArray) -> float:
    return float(np.max(y)) if y.size else 0.0


def _median(x: FloatArray) -> float:
    return float(np.median(x)) if x.size else 0.0


def _half_y(x: FloatArray, y: FloatArray) -> float:
    """``x`` where the curve crosses the middle of its ``y`` range (linear interpolation)."""
    if y.size < 2:
        return _median(x)
    level = float(np.min(y)) / 2.0 + float(np.max(y)) / 2.0
    for i in range(y.size - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if (y0 - level) * (y1 - level) <= 0.0 and y0 != y1:
            t = (level - y0) / (y1 - y0)
            return float(x[i]) + t * (float(x[i + 1]) - float(x[i]))
    return _median(x)


def _least_squares(
    columns: Sequence[FloatArray], target: FloatArray, weights: FloatArray
) -> FloatArray | None:
    """Weighted linear least squares ``target ~ columns``; ``None`` if not determined.

    Columns are scaled to unit length first so wildly different magnitudes do not fight over
    the tolerance of the rank decision.
    """
    if target.size < len(columns) or len(columns) == 0:
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


def _linear_sse(
    target: FloatArray, columns: Sequence[FloatArray]
) -> tuple[FloatArray | None, float]:
    """Best linear combination of ``columns`` for ``target`` and its residual sum of squares."""
    coef = _least_squares(columns, target, np.ones(target.size))
    if coef is None:
        return None, math.inf
    model = np.zeros_like(target)
    for value, column in zip(coef, columns, strict=True):
        model = model + value * column
    if not np.all(np.isfinite(model)):
        return None, math.inf
    residual = target - model
    with np.errstate(all="ignore"):
        sse = float(np.sum(residual * residual))
    return coef, sse if math.isfinite(sse) else math.inf


def _golden(objective: Callable[[float], float], low: float, high: float) -> float:
    """Golden-section minimiser of a unimodal ``objective`` on ``[low, high]``.

    Both endpoints must be finite with ``low < high``; a non-finite objective simply loses to
    a finite competitor.
    """
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        return low if math.isfinite(low) else 0.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = low, high
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(40):  # 40 halvings of 0.618**2 leave the interval at rounding resolution
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    return c if fc <= fd else d


_GRID_COARSE = 21
"""Candidate values per shape parameter in a two-shape coarse scan."""
_GRID_THREE = 9
"""Candidate values per shape parameter when three shapes are scanned together."""
_GRID_SINGLE = 15
"""Candidate values when there is only one shape parameter to scan."""


def _count_for(shapes: int) -> int:
    """Grid points per shape parameter, so the full product stays a few hundred points."""
    if shapes == 1:
        return _GRID_SINGLE
    return _GRID_THREE if shapes >= 3 else _GRID_COARSE


def _grid_values(low: float, high: float, count: int) -> list[float]:
    """Candidate shape values: geometric over a wide positive range, linear otherwise."""
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high or count < 2:
        return [low]
    if low > 0.0 and high / low >= 20.0:
        ratio = (high / low) ** (1.0 / (count - 1))
        return [low * ratio**i for i in range(count)]
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def _grid_search(
    values: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    objective: Callable[[Mapping[str, float]], float],
) -> dict[str, float]:
    """Best point of the full cartesian grid over the shape parameters.

    A full product, not a coordinate sweep: location and scale are strongly correlated, so
    sweeping one while the other is fixed can settle in a valley that is not the optimum.
    """
    names = list(bounds)
    count = _count_for(len(names))
    grids = [_grid_values(bounds[name][0], bounds[name][1], count) for name in names]
    best_values = dict(values)
    best = objective(values)
    for combination in itertools.product(*grids):
        trial = dict(values)
        trial.update(zip(names, combination, strict=True))
        score = objective(trial)
        if score < best:
            best, best_values = score, trial
    return best_values


def _polish(
    values: dict[str, float],
    bounds: Mapping[str, tuple[float, float]],
    objective: Callable[[Mapping[str, float]], float],
    *,
    rounds: int = 2,
) -> dict[str, float]:
    """Golden-section polish of every shape parameter between its neighbouring grid values.

    A polish is kept only when it lowers the objective: golden-section search assumes a
    unimodal bracket, and a strongly coupled pair of shapes can break that assumption.
    """
    current = objective(values)
    for _ in range(max(rounds, 1)):
        for name, (low, high) in bounds.items():
            count = _count_for(len(bounds))
            grid = _grid_values(low, high, count)
            ordered = sorted(set(grid) | {values[name]})
            position = ordered.index(values[name])
            left = ordered[max(position - 1, 0)]
            right = ordered[min(position + 1, len(ordered) - 1)]

            def polish(t: float, name: str = name) -> float:
                trial = dict(values)
                trial[name] = t
                return objective(trial)

            if left < right:
                candidate = _golden(polish, left, right)
                trial = dict(values)
                trial[name] = candidate
                score = objective(trial)
                if score < current:
                    values[name] = candidate
                    current = score
    return values


def _refine_shapes(
    x: FloatArray,
    y: FloatArray,
    basis: Basis,
    shapes: Shapes,
    *,
    rounds: int = 2,
) -> tuple[dict[str, float], FloatArray | None]:
    """Grid scan and local polish of the shape parameters of a linear basis.

    ``basis(x, shapes)`` returns the columns whose linear combination is the model; for fixed
    shapes the coefficients are solved by least squares and the shapes are optimised on the
    resulting residual sum of squares. Starting values are the third entry of each tuple.
    """
    values = {name: initial for name, (_, _, initial) in shapes.items()}
    bounds = {name: (low, high) for name, (low, high, _) in shapes.items()}

    def evaluate(current: Mapping[str, float]) -> tuple[FloatArray | None, float]:
        try:
            columns = basis(x, current)
        except FloatingPointError, OverflowError, ValueError:
            return None, math.inf
        return _linear_sse(y, columns)

    def scored(current: Mapping[str, float]) -> float:
        return evaluate(current)[1]

    values = _polish(_grid_search(values, bounds, scored), bounds, scored, rounds=rounds)
    best_coef, _ = evaluate(values)
    return values, best_coef


def _bespoke_guess(
    parameters: tuple[Parameter, ...], estimate: Estimator
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """``initial_guess`` from an estimator; every value is moved strictly inside the bounds."""
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _clean(x, y)
        with np.errstate(all="ignore"):
            try:
                estimates = estimate(xs, ys)
            except FloatingPointError, OverflowError, ValueError:
                estimates = dict.fromkeys((p.name for p in parameters), 0.0)
        values = {k: (v if math.isfinite(v) else 0.0) for k, v in estimates.items()}
        return rules_guess(values, bounds)(xs, ys)

    return guess


# -- applicability ----------------------------------------------------------------------


def _count(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


def _domain_check(
    *,
    x_positive_for: str | None = None,
    x_non_negative: bool = False,
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, plus an optional restriction on the sign of ``x``."""

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        if (n := int(np.count_nonzero(~np.isfinite(xa)))) > 0:
            reasons.append(f"x must be finite: found {_count(n, 'value')} that are NaN or infinite")
        if (n := int(np.count_nonzero(~np.isfinite(ya)))) > 0:
            reasons.append(f"y must be finite: found {_count(n, 'value')} that are NaN or infinite")
        finite_x = xa[np.isfinite(xa)]
        if x_positive_for is not None and (n := int(np.count_nonzero(finite_x <= 0.0))) > 0:
            reasons.append(
                f"x must be strictly positive for {x_positive_for}: found {_count(n, 'value')} "
                "that are zero or negative"
            )
        if x_non_negative and (n := int(np.count_nonzero(finite_x < 0.0))) > 0:
            reasons.append(
                f"x must be non-negative (a time or an amount): found {_count(n, 'negative value')}"
            )
        return reasons

    return check


# -- initial guesses --------------------------------------------------------------------


def _logistic_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = 1.0 / (1.0 + np.exp(-shapes["b"] * (x - shapes["x0"])))
    return [1.0 - s, s]


def _logistic4(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _logistic_basis,
        {
            "b": (1e-3 / span, 1e3 / span, 4.0 / span),
            "x0": (low, high, _half_y(x, y)),
        },
    )
    if coef is None:
        centre = _half_y(x, y)
        amplitude = _extent(y)
        return {"a": _ymin(y), "k": _ymax(y) + amplitude, "b": 4.0 / span, "x0": centre}
    return {"a": float(coef[0]), "k": float(coef[1]), "b": values["b"], "x0": values["x0"]}


def _logistic5_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = np.power(1.0 + np.exp(-shapes["b"] * (x - shapes["x0"])), -shapes["g"])
    return [1.0 - s, s]


def _logistic5(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _logistic5_basis,
        {
            "b": (1e-3 / span, 1e3 / span, 4.0 / span),
            "x0": (low, high, _half_y(x, y)),
            "g": (0.1, 10.0, 1.0),
        },
    )
    if coef is None:
        amplitude = _extent(y)
        return {
            "a": _ymin(y),
            "k": _ymax(y) + amplitude,
            "b": 4.0 / span,
            "x0": _half_y(x, y),
            "g": 1.0,
        }
    return {
        "a": float(coef[0]),
        "k": float(coef[1]),
        "b": values["b"],
        "x0": values["x0"],
        "g": values["g"],
    }


def _boltzmann(x: FloatArray, y: FloatArray) -> dict[str, float]:
    fit = _logistic4(x, y)
    if not (math.isfinite(fit["b"]) and fit["b"] > 0.0):
        return {"a": _ymin(y), "b": _extent(y), "c": _half_y(x, y), "d": _span(x) / 4.0}
    return {"a": fit["a"], "b": fit["k"] - fit["a"], "c": fit["x0"], "d": 1.0 / fit["b"]}


def _tanh_sigmoid(x: FloatArray, y: FloatArray) -> dict[str, float]:
    fit = _logistic4(x, y)
    if not (math.isfinite(fit["b"]) and fit["b"] > 0.0):
        return {"a": _median(y), "b": 0.5 * _extent(y), "c": _half_y(x, y), "d": _span(x) / 8.0}
    return {
        "a": 0.5 * (fit["a"] + fit["k"]),
        "b": 0.5 * (fit["k"] - fit["a"]),
        "c": fit["x0"],
        "d": 2.0 / fit["b"],
    }


def _erf_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = 0.5 * (1.0 + _erf((x - shapes["c"]) / shapes["d"]))
    return [np.ones_like(x), s]


def _erf_sigmoid(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _erf_basis,
        {
            "c": (low, high, _half_y(x, y)),
            "d": (1e-3 * span, 1e2 * span, span / 4.0),
        },
    )
    if coef is None:
        return {"a": _ymin(y), "b": _extent(y), "c": _half_y(x, y), "d": span / 4.0}
    return {"a": float(coef[0]), "b": float(coef[1]), "c": values["c"], "d": values["d"]}


def _arctan_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = np.arctan((x - shapes["c"]) / shapes["d"]) / np.pi + 0.5
    return [np.ones_like(x), s]


def _arctan_sigmoid(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _arctan_basis,
        {
            "c": (low, high, _half_y(x, y)),
            "d": (1e-3 * span, 1e2 * span, span / 4.0),
        },
    )
    if coef is None:
        return {"a": _ymin(y), "b": _extent(y), "c": _half_y(x, y), "d": span / 4.0}
    return {"a": float(coef[0]), "b": float(coef[1]), "c": values["c"], "d": values["d"]}


def _weibull_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        f = 1.0 - np.exp(-np.power(x / shapes["b"], shapes["c"]))
    return [f]


def _weibull_cdf3(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x,
        y,
        _weibull_basis,
        {
            "b": (1e-3 * span, 1e3 * span, span / 3.0),
            "c": (0.05, 20.0, 2.0),
        },
    )
    a = float(coef[0]) if coef is not None else _ymax(y)
    return {"a": a, "b": values["b"], "c": values["c"]}


def _logistic_quadratic_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    u = x - shapes["d"]
    s = 1.0 / (1.0 + np.exp(-(shapes["c"] * u + shapes["e"] * u**2)))
    return [np.ones_like(x), s]


def _logistic_quadratic5(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Logistic4 gives the centre and steepness; search only around that solution.

    The quadratic term is a small correction to the logit, so a grid over the whole real
    line for ``d`` (or over many decades for ``c``) would miss the good basin once noise is
    present. The four-parameter logistic locates it first, and the five-parameter search is
    confined to a box around it.
    """
    span = _span(x)
    base = _logistic4(x, y)
    centre = base["x0"] if math.isfinite(base["x0"]) else _half_y(x, y)
    rate = base["b"] if math.isfinite(base["b"]) and base["b"] > 0.0 else 2.0 / span
    values, coef = _refine_shapes(
        x,
        y,
        _logistic_quadratic_basis,
        {
            "c": (max(0.05 * rate, 1e-3 / span), 20.0 * rate, rate),
            "d": (centre - 0.75 * span, centre + 0.75 * span, centre),
            "e": (-2.0 * rate / span, 2.0 * rate / span, 0.0),
        },
    )
    if coef is None:
        return {
            "a": base["a"],
            "b": max(base["k"] - base["a"], 1e-6),
            "c": rate,
            "d": centre,
            "e": 0.0,
        }
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": values["c"],
        "d": values["d"],
        "e": values["e"],
    }


def _arctan_drift_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = np.arctan((x - shapes["c"]) / shapes["d"]) / np.pi + 0.5
    return [np.ones_like(x), s, x]


def _arctan_drift5(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _arctan_drift_basis,
        {
            "c": (low, high, _half_y(x, y)),
            "d": (1e-3 * span, 1e2 * span, span / 4.0),
        },
    )
    if coef is None:
        return {"a": _ymin(y), "b": _extent(y), "c": _half_y(x, y), "d": span / 4.0, "e": 0.0}
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": values["c"],
        "d": values["d"],
        "e": float(coef[2]),
    }


def _hill_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        xn = np.power(x, shapes["n"])
        return [xn / (np.power(shapes["km"], shapes["n"]) + xn)]


def _hill_sigmoid(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x,
        y,
        _hill_basis,
        {
            "km": (1e-3 * span, 1e3 * span, _half_y(x, y)),
            "n": (0.05, 20.0, 1.5),
        },
    )
    vmax = float(coef[0]) if coef is not None else _ymax(y)
    return {"vmax": vmax, "km": values["km"], "n": values["n"]}


def _algebraic_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [np.ones_like(x), x / np.sqrt(shapes["c"] ** 2 + x**2)]


def _algebraic_sigmoid(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x, y, _algebraic_basis, {"c": (1e-3 * span, 1e3 * span, span / 2.0)}
    )
    if coef is None:
        return {"a": _median(y), "b": _extent(y), "c": span / 2.0}
    return {"a": float(coef[0]), "b": float(coef[1]), "c": values["c"]}


def _logistic_drift_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = 1.0 / (1.0 + np.exp(-shapes["b"] * (x - shapes["x0"])))
    return [1.0 - s, s, x]


def _logistic_drift5(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _logistic_drift_basis,
        {
            "b": (1e-3 / span, 1e3 / span, 4.0 / span),
            "x0": (low, high, _half_y(x, y)),
        },
    )
    if coef is None:
        amplitude = _extent(y)
        return {
            "a": _ymin(y),
            "k": _ymax(y) + amplitude,
            "b": 4.0 / span,
            "x0": _half_y(x, y),
            "e": 0.0,
        }
    return {
        "a": float(coef[0]),
        "k": float(coef[1]),
        "b": values["b"],
        "x0": values["x0"],
        "e": float(coef[2]),
    }


def _frechet_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        f = np.exp(-np.power(x / shapes["e"], -shapes["b"]))
    return [1.0 - f, f]


def _frechet_cdf(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x,
        y,
        _frechet_basis,
        {
            "e": (1e-3 * span, 1e3 * span, _half_y(x, y)),
            "b": (0.05, 20.0, 1.5),
        },
    )
    if coef is None:
        return {"c": _ymin(y), "d": _ymax(y) + _extent(y), "e": span / 2.0, "b": 1.5}
    return {
        "c": float(coef[0]),
        "d": float(coef[1]),
        "e": values["e"],
        "b": values["b"],
    }


def _erf_drift_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = 0.5 * (1.0 + _erf((x - shapes["c"]) / shapes["d"]))
    return [np.ones_like(x), s, x]


def _erf_drift5(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _erf_drift_basis,
        {
            "c": (low, high, _half_y(x, y)),
            "d": (1e-3 * span, 1e2 * span, span / 4.0),
        },
    )
    if coef is None:
        return {
            "a": _ymin(y),
            "b": _extent(y),
            "c": _half_y(x, y),
            "d": span / 4.0,
            "e": 0.0,
        }
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": values["c"],
        "d": values["d"],
        "e": float(coef[2]),
    }


# -- the models -------------------------------------------------------------------------

_P_LOGISTIC4 = (
    Parameter("a", "Lower asymptote of the S-curve"),
    Parameter("k", "Upper asymptote of the S-curve"),
    Parameter("b", "Steepness: growth rate per unit x at the midpoint", lower=0.0),
    Parameter("x0", "Midpoint: the x at which y is halfway between the asymptotes"),
)
_P_LOGISTIC5 = (
    Parameter("a", "Lower asymptote of the S-curve"),
    Parameter("k", "Upper asymptote of the S-curve"),
    Parameter("b", "Steepness of the underlying logistic", lower=0.0),
    Parameter("x0", "Midpoint of the underlying logistic"),
    Parameter("g", "Asymmetry exponent: 1 is symmetric, other values skew the S-curve", lower=0.0),
)
_P_BOLTZMANN = (
    Parameter("a", "Lower asymptote"),
    Parameter("b", "Amplitude: the rise from the lower to the upper asymptote", lower=0.0),
    Parameter("c", "Midpoint: the x at which y is halfway up"),
    Parameter("d", "Scale of x: larger values give a gentler curve", lower=0.0),
)
_P_TANH = (
    Parameter("a", "Centre of the curve (the value at the midpoint)"),
    Parameter("b", "Half-amplitude of the swing", lower=0.0),
    Parameter("c", "Midpoint: the x at which the curve crosses a"),
    Parameter("d", "Scale of x: larger values give a gentler curve", lower=0.0),
)
_P_ERF = (
    Parameter("a", "Lower asymptote"),
    Parameter("b", "Amplitude of the rise", lower=0.0),
    Parameter("c", "Midpoint: the x at which y is halfway up"),
    Parameter("d", "Scale of x in the error function", lower=0.0),
)
_P_ARCTAN = (
    Parameter("a", "Lower asymptote"),
    Parameter("b", "Amplitude of the rise", lower=0.0),
    Parameter("c", "Midpoint: the x at which y is halfway up"),
    Parameter("d", "Scale of x in the arctangent", lower=0.0),
)
_P_WEIBULL3 = (
    Parameter("a", "Upper asymptote of the response", lower=0.0),
    Parameter("b", "Scale: the x at which a fixed fraction of the rise is complete", lower=0.0),
    Parameter("c", "Shape exponent of the Weibull curve", lower=0.0),
)
_P_LOGISTIC_QUADRATIC = (
    Parameter("a", "Lower asymptote of the S-curve"),
    Parameter("b", "Amplitude of the rise", lower=0.0),
    Parameter("c", "Linear steepness of the logit at the centre", lower=0.0),
    Parameter("d", "Centre of the S-curve: the x at which the logit's linear term is zero"),
    Parameter("e", "Quadratic term of the logit: skews the curve without moving its tails"),
)
_P_ARCTAN_DRIFT = (
    Parameter("a", "Lower asymptote of the arctangent part"),
    Parameter("b", "Amplitude of the arctangent part", lower=0.0),
    Parameter("c", "Midpoint of the arctangent part"),
    Parameter("d", "Scale of x in the arctangent", lower=0.0),
    Parameter("e", "Linear drift: change of y per unit x on top of the S-curve"),
)
_P_HILL = (
    Parameter("vmax", "Upper asymptote of the Hill curve", lower=0.0),
    Parameter("km", "Half-saturation value: the x at which y is vmax/2", lower=0.0),
    Parameter("n", "Hill coefficient: steeper curves have larger n", lower=0.0),
)
_P_ALGEBRAIC = (
    Parameter("a", "Value of y at x = 0 (the centre of the curve)"),
    Parameter("b", "Half-amplitude of the swing", lower=0.0),
    Parameter("c", "Scale of x: larger values give a gentler curve", lower=0.0),
)
_P_LOGISTIC_DRIFT = (
    Parameter("a", "Lower asymptote of the logistic part"),
    Parameter("k", "Upper asymptote of the logistic part"),
    Parameter("b", "Steepness of the logistic part", lower=0.0),
    Parameter("x0", "Midpoint of the logistic part"),
    Parameter("e", "Linear drift: change of y per unit x on top of the S-curve"),
)
_P_FRECHET = (
    Parameter("c", "Lower asymptote"),
    Parameter("d", "Upper asymptote"),
    Parameter("e", "Scale of the Frechet distribution", lower=0.0),
    Parameter("b", "Shape exponent of the Frechet distribution", lower=0.0),
)
_P_ERF_DRIFT = (
    Parameter("a", "Lower asymptote of the error-function part"),
    Parameter("b", "Amplitude of the error-function part", lower=0.0),
    Parameter("c", "Midpoint of the error-function part"),
    Parameter("d", "Scale of x in the error function", lower=0.0),
    Parameter("e", "Linear drift: change of y per unit x on top of the S-curve"),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="logistic4",
        name="Four-parameter logistic",
        category="sigmoidal",
        description="Symmetric S-curve between two free asymptotes.",
        example=ModelExample({"a": 1.0, "k": 10.0, "b": 1.2, "x0": 3.0}, x_range=(0.0, 8.0)),
        expression="a+(k-a)/(1+exp(-b*(x-x0)))",
        parameters=_P_LOGISTIC4,
        initial_guess=_bespoke_guess(_P_LOGISTIC4, _logistic4),
        domain_check=_domain_check(),
        references=(_RICHARDS, _SEBER_WILD),
        assumptions=(
            "The curve is symmetric about its midpoint and bounded by the two asymptotes; "
            "both asymptotes are estimated from data that span the transition."
        ),
        aliases=("logistic_4_parameter", "logistic_curve_4"),
    ),
    ModelSpec(
        id="logistic5",
        name="Five-parameter logistic",
        category="sigmoidal",
        description="Asymmetric logistic S-curve with a free asymmetry exponent.",
        example=ModelExample(
            {"a": 1.0, "k": 10.0, "b": 1.5, "x0": 3.0, "g": 0.7}, x_range=(0.0, 8.0)
        ),
        expression="a+(k-a)/(1+exp(-b*(x-x0)))**g",
        parameters=_P_LOGISTIC5,
        initial_guess=_bespoke_guess(_P_LOGISTIC5, _logistic5),
        domain_check=_domain_check(),
        references=(_RICHARDS, _BAYES_BATES),
        assumptions=(
            "The S-curve is asymmetric: the approach to one asymptote is stretched by the "
            "exponent g (g = 1 makes the curve the four-parameter logistic)."
        ),
        aliases=("generalised_logistic", "generalized_logistic"),
    ),
    ModelSpec(
        id="boltzmann",
        name="Boltzmann sigmoid",
        category="sigmoidal",
        description="Sigmoid written with an explicit amplitude and a scale of x.",
        example=ModelExample({"a": 1.0, "b": 9.0, "c": 3.0, "d": 1.25}, x_range=(0.0, 8.0)),
        expression="a+b/(1+exp(-(x-c)/d))",
        parameters=_P_BOLTZMANN,
        initial_guess=_bespoke_guess(_P_BOLTZMANN, _boltzmann),
        domain_check=_domain_check(),
        references=(_BOLTZMANN,),
        assumptions=(
            "The curve rises from a to a + b with a scale d; it is the four-parameter "
            "logistic with the upper asymptote written as a + b and the rate as 1/d."
        ),
        aliases=("boltzmann_sigmoid", "fermi_dirac"),
    ),
    ModelSpec(
        id="tanh_sigmoid",
        name="Hyperbolic-tangent sigmoid",
        category="sigmoidal",
        description="Symmetric S-curve written with the hyperbolic tangent.",
        example=ModelExample({"a": 5.5, "b": 4.5, "c": 3.0, "d": 0.75}, x_range=(0.0, 8.0)),
        expression="a+b*tanh((x-c)/d)",
        parameters=_P_TANH,
        initial_guess=_bespoke_guess(_P_TANH, _tanh_sigmoid),
        domain_check=_domain_check(),
        references=(_BOLTZMANN, _SEBER_WILD),
        assumptions=(
            "The curve is antisymmetric about (c, a) and bounded by a - b and a + b; the "
            "hyperbolic tangent is a logistic curve with rate 2/d."
        ),
        aliases=("hyperbolic_tangent_sigmoid", "tanh_curve"),
    ),
    ModelSpec(
        id="erf_sigmoid",
        name="Error-function (probit) sigmoid",
        category="sigmoidal",
        description="S-curve shaped like the normal distribution function.",
        example=ModelExample({"a": 1.0, "b": 9.0, "c": 3.0, "d": 1.5}, x_range=(0.0, 8.0)),
        expression="a+b*0.5*(1+erf((x-c)/d))",
        parameters=_P_ERF,
        initial_guess=_bespoke_guess(_P_ERF, _erf_sigmoid),
        domain_check=_domain_check(),
        references=(_ERRF, _SEBER_WILD),
        assumptions=(
            "The response is a normal (probit) curve with lighter tails than a logistic; "
            "d is the scale of x inside the error function."
        ),
        aliases=("probit", "normal_cdf_sigmoid", "error_function_sigmoid"),
    ),
    ModelSpec(
        id="arctan_sigmoid",
        name="Arctangent (cauchit) sigmoid",
        category="sigmoidal",
        description="S-curve with algebraic tails, the Cauchy distribution function.",
        example=ModelExample({"a": 1.0, "b": 9.0, "c": 3.0, "d": 1.5}, x_range=(0.0, 8.0)),
        expression="a+b*(atan((x-c)/d)/pi+0.5)",
        parameters=_P_ARCTAN,
        initial_guess=_bespoke_guess(_P_ARCTAN, _arctan_sigmoid),
        domain_check=_domain_check(),
        references=(_ERRF,),
        assumptions=(
            "The response follows a Cauchy (cauchit) curve, whose tails are heavier than a "
            "logistic's; d is the scale of x."
        ),
        aliases=("cauchit", "arctangent_sigmoid"),
    ),
    ModelSpec(
        id="weibull_cdf3",
        name="Weibull CDF (three parameters)",
        category="sigmoidal",
        description="Weibull distribution function rising from zero to an asymptote.",
        example=ModelExample({"a": 10.0, "b": 3.0, "c": 2.0}, x_range=(0.1, 10.0)),
        expression="a*(1-exp(-(x/b)**c))",
        parameters=_P_WEIBULL3,
        initial_guess=_bespoke_guess(_P_WEIBULL3, _weibull_cdf3),
        domain_check=_domain_check(x_positive_for="(x/b)**c"),
        references=(_WEIBULL,),
        assumptions=(
            "x is strictly positive and the response starts at zero; b is a scale and c a "
            "shape exponent (c = 1 gives a pure exponential CDF)."
        ),
        aliases=("weibull_cdf", "weibull_type2_3", "weibull_distribution"),
    ),
    ModelSpec(
        id="logistic_quadratic5",
        name="Logistic with a quadratic logit",
        category="sigmoidal",
        description="Asymmetric logistic S-curve whose logit is quadratic in x.",
        example=ModelExample(
            {"a": 1.0, "b": 9.0, "c": 1.2, "d": 3.0, "e": -0.04}, x_range=(0.0, 8.0)
        ),
        expression="a+b/(1+exp(-(c*(x-d)+e*(x-d)**2)))",
        parameters=_P_LOGISTIC_QUADRATIC,
        initial_guess=_bespoke_guess(_P_LOGISTIC_QUADRATIC, _logistic_quadratic5),
        domain_check=_domain_check(),
        references=(_RICHARDS, _SEBER_WILD),
        assumptions=(
            "The logit (log-odds) of the normalised response is a quadratic in x, so one "
            "tail of the S-curve is stretched relative to the other; e = 0 is the "
            "four-parameter logistic."
        ),
        aliases=("quadratic_logit", "asymmetric_logistic_quadratic"),
    ),
    ModelSpec(
        id="arctan_drift5",
        name="Arctangent sigmoid with linear drift",
        category="sigmoidal",
        description="Cauchit S-curve on top of a straight-line trend.",
        example=ModelExample(
            {"a": 1.0, "b": 9.0, "c": 3.0, "d": 1.5, "e": 0.1}, x_range=(0.0, 8.0)
        ),
        expression="a+b*(atan((x-c)/d)/pi+0.5)+e*x",
        parameters=_P_ARCTAN_DRIFT,
        initial_guess=_bespoke_guess(_P_ARCTAN_DRIFT, _arctan_drift5),
        domain_check=_domain_check(),
        references=(_ERRF, _SEBER_WILD),
        assumptions=(
            "A Cauchy (cauchit) S-curve plus the linear drift e*x; e may be zero, in which "
            "case the model is the arctangent sigmoid."
        ),
        aliases=("cauchit_linear_trend",),
    ),
    ModelSpec(
        id="hill_sigmoid",
        name="Hill equation",
        category="sigmoidal",
        description="Hill saturation curve with a cooperative binding exponent.",
        example=ModelExample({"vmax": 10.0, "km": 3.0, "n": 2.0}, x_range=(0.1, 20.0)),
        expression="vmax*x**n/(km**n+x**n)",
        parameters=_P_HILL,
        initial_guess=_bespoke_guess(_P_HILL, _hill_sigmoid),
        domain_check=_domain_check(x_positive_for="x**n"),
        references=(_HILL, _SEBER_WILD),
        assumptions=(
            "x is non-negative and the response rises from zero to vmax; n > 1 gives the "
            "cooperative S-shape, n = 1 the hyperbola."
        ),
        aliases=("hill_equation", "hill3"),
    ),
    ModelSpec(
        id="algebraic_sigmoid",
        name="Algebraic sigmoid",
        category="sigmoidal",
        description="S-curve built from x over the square root of a quadratic.",
        example=ModelExample({"a": 5.0, "b": 4.0, "c": 2.0}, x_range=(-3.0, 8.0)),
        expression="a+b*x/sqrt(c**2+x**2)",
        parameters=_P_ALGEBRAIC,
        initial_guess=_bespoke_guess(_P_ALGEBRAIC, _algebraic_sigmoid),
        domain_check=_domain_check(),
        references=(_ERRF,),
        assumptions=(
            "The curve is antisymmetric about a and bounded by a - b and a + b; c sets how "
            "abruptly it changes sign."
        ),
        aliases=("algebraic_saturation",),
    ),
    ModelSpec(
        id="logistic_drift5",
        name="Logistic with linear drift",
        category="sigmoidal",
        description="S-curve on top of a straight-line trend.",
        example=ModelExample(
            {"a": 1.0, "k": 10.0, "b": 1.2, "x0": 3.0, "e": 0.05}, x_range=(0.0, 8.0)
        ),
        expression="a+(k-a)/(1+exp(-b*(x-x0)))+e*x",
        parameters=_P_LOGISTIC_DRIFT,
        initial_guess=_bespoke_guess(_P_LOGISTIC_DRIFT, _logistic_drift5),
        domain_check=_domain_check(),
        references=(_SEBER_WILD, _BAYES_BATES),
        assumptions=(
            "The S-curve sits on a linear drift e*x; e may be zero, in which case the model "
            "reduces to the four-parameter logistic."
        ),
        aliases=("logistic_linear_trend",),
    ),
    ModelSpec(
        id="frechet_cdf",
        name="Frechet distribution function",
        category="sigmoidal",
        description="Heavy-tailed S-curve rising steeply from a finite lower asymptote.",
        example=ModelExample({"c": 1.0, "d": 9.0, "e": 3.0, "b": 1.4}, x_range=(0.2, 20.0)),
        expression="c+(d-c)*exp(-(x/e)**(-b))",
        parameters=_P_FRECHET,
        initial_guess=_bespoke_guess(_P_FRECHET, _frechet_cdf),
        domain_check=_domain_check(x_positive_for="(x/e)**(-b)"),
        references=(_FRECHET,),
        assumptions=(
            "x is strictly positive; the response rises from c to d with a heavy right tail "
            "and is steeper than a logistic for the same scale."
        ),
        aliases=("frechet_distribution",),
    ),
    ModelSpec(
        id="erf_drift5",
        name="Error-function sigmoid with linear drift",
        category="sigmoidal",
        description="Probit-shaped S-curve on top of a straight-line trend.",
        example=ModelExample(
            {"a": 1.0, "b": 9.0, "c": 3.0, "d": 1.5, "e": 0.1}, x_range=(0.0, 8.0)
        ),
        expression="a+b*0.5*(1+erf((x-c)/d))+e*x",
        parameters=_P_ERF_DRIFT,
        initial_guess=_bespoke_guess(_P_ERF_DRIFT, _erf_drift5),
        domain_check=_domain_check(),
        references=(_ERRF, _SEBER_WILD),
        assumptions=(
            "A normal (probit) S-curve plus the linear drift e*x; e may be zero, in which "
            "case the model is the error-function sigmoid."
        ),
        aliases=("probit_linear_trend",),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # a + (k-a)/(1+exp(-b*(x-x0))) is a + b'/(1+exp(-(x-c)/d)) with b' = k-a, d = 1/b.
    ("logistic4", "boltzmann"),
    # a + b*tanh((x-c)/d) = (a-b) + 2b/(1+exp(-2(x-c)/d)): the same logistic curve.
    ("logistic4", "tanh_sigmoid"),
    ("boltzmann", "tanh_sigmoid"),
)
