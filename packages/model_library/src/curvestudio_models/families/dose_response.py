"""Dose-response curves: log-logistic, Weibull, Emax and hormesis models.

Every model is fitted on the original response scale (no transformation of ``y`` or of ``x``
is part of the fit). For fixed scale and shape parameters most curves are linear in two or
three coefficients — the two asymptotes, or the baseline, the amplitude and the hormesis term
— so the guess solves that linear sub-problem over a grid of shapes and polishes the shapes
with a golden-section search. The normalised two-parameter log-logistic has no linear
coefficient and is scanned directly. The search is deterministic (no RNG).
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from math import erf

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]
Basis = Callable[[FloatArray, Mapping[str, float]], list[FloatArray]]
Shapes = Mapping[str, tuple[float, float, float]]

_erf = np.vectorize(erf)

_SEBAUGH = (
    "Sebaugh J.L. (2011) Guidelines for accurate EC50/IC50 estimation. Pharmaceutical "
    "Statistics 10(2):128-134"
)
_RITZ = (
    "Ritz C., Baty F., Streibig J.C., Gerhard D. (2015) Dose-response analysis using R. "
    "PLoS ONE 10(12):e0146021"
)
_FINNEY = "Finney D.J. (1978) Statistical Method in Biological Assay, 3rd ed., Griffin, London"
_WEIBULL = (
    "Weibull W. (1951) A statistical distribution function of wide applicability. Journal of "
    "Applied Mechanics 18(3):293-297"
)
_HOLFORD = (
    "Holford N.H.G., Sheiner L.B. (1981) Understanding the dose-effect relationship: clinical "
    "application of pharmacokinetic-pharmacodynamic models. Clinical Pharmacokinetics "
    "6(6):429-453"
)
_BRAIN_COUSENS = (
    "Brain P., Cousens R. (1989) An equation to describe dose responses where there is "
    "stimulation of growth at low doses. Weed Research 29(2):93-96"
)
_CEDER_GREEN = (
    "Cedergreen N., Ritz C., Streibig J.C. (2005) Improved empirical models describing "
    "hormesis. Environmental Toxicology and Chemistry 24(12):3166-3172"
)
_BLISS = "Bliss C.I. (1934) The method of probits. Science 79(2037):38-39"


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

    A full product, not a coordinate sweep: scale and shape are strongly correlated, so
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


def _coordinate_scan(
    values: dict[str, float],
    bounds: Mapping[str, tuple[float, float]],
    objective: Callable[[Mapping[str, float]], float],
    *,
    rounds: int = 2,
) -> dict[str, float]:
    """Full grid scan over the parameters, then a local golden-section polish."""
    return _polish(_grid_search(values, bounds, objective), bounds, objective, rounds=rounds)


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
                "x must be non-negative (a dose or a concentration): found "
                f"{_count(n, 'negative value')}"
            )
        return reasons

    return check


# -- initial guesses --------------------------------------------------------------------


def _logistic_reciprocal(x: FloatArray, shapes: Mapping[str, float]) -> FloatArray:
    with np.errstate(all="ignore"):
        return 1.0 / (1.0 + np.power(x / shapes["e"], shapes["b"]))


def _logistic_shape_bounds(x: FloatArray, y: FloatArray) -> dict[str, tuple[float, float, float]]:
    span = _span(x)
    return {
        "b": (0.05, 20.0, 1.5),
        "e": (1e-3 * span, 1e3 * span, max(_half_y(x, y), 1e-3 * span)),
    }


def _ll2(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)

    def objective(b: float, e: float) -> float:
        with np.errstate(all="ignore"):
            model = 1.0 / (1.0 + np.power(x / e, b))
            if not np.all(np.isfinite(model)):
                return math.inf
            sse = float(np.sum((y - model) ** 2))
        return sse if math.isfinite(sse) else math.inf

    bounds = {"b": (0.05, 20.0), "e": (1e-3 * span, 1e3 * span)}
    values = _coordinate_scan(
        {"b": 1.5, "e": max(_half_y(x, y), 1e-3 * span)},
        bounds,
        lambda current: objective(current["b"], current["e"]),
    )
    return {"e": values["e"], "b": values["b"]}


def _ll3_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [_logistic_reciprocal(x, shapes)]


def _ll3(x: FloatArray, y: FloatArray) -> dict[str, float]:
    values, coef = _refine_shapes(x, y, _ll3_basis, _logistic_shape_bounds(x, y))
    d = float(coef[0]) if coef is not None else _ymax(y)
    return {"d": d, "e": values["e"], "b": values["b"]}


def _four_pl_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = _logistic_reciprocal(x, shapes)
    return [1.0 - s, s]


def _four_pl(x: FloatArray, y: FloatArray) -> dict[str, float]:
    values, coef = _refine_shapes(x, y, _four_pl_basis, _logistic_shape_bounds(x, y))
    if coef is None:
        spread = _extent(y)
        return {
            "c": _ymin(y) - 0.1 * spread,
            "d": _ymax(y) + 0.1 * spread,
            "e": values["e"],
            "b": values["b"],
        }
    return {"c": float(coef[0]), "d": float(coef[1]), "e": values["e"], "b": values["b"]}


def _five_pl_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    s = np.power(_logistic_reciprocal(x, shapes), shapes["f"])
    return [1.0 - s, s]


def _five_pl(x: FloatArray, y: FloatArray) -> dict[str, float]:
    shapes = _logistic_shape_bounds(x, y)
    shapes["f"] = (0.05, 20.0, 1.0)
    values, coef = _refine_shapes(x, y, _five_pl_basis, shapes)
    if coef is None:
        spread = _extent(y)
        return {
            "c": _ymin(y) - 0.1 * spread,
            "d": _ymax(y) + 0.1 * spread,
            "e": values["e"],
            "b": values["b"],
            "f": values["f"],
        }
    return {
        "c": float(coef[0]),
        "d": float(coef[1]),
        "e": values["e"],
        "b": values["b"],
        "f": values["f"],
    }


def _exponential_shape_bounds(
    x: FloatArray, y: FloatArray
) -> dict[str, tuple[float, float, float]]:
    """Bounds for a scale ``e`` and a shape exponent ``b`` shared by the Weibull-type curves."""
    span = _span(x)
    return {
        "b": (0.05, 20.0, 1.5),
        "e": (1e-3 * span, 1e3 * span, max(_half_y(x, y), 1e-3 * span)),
    }


def _weibull_decay_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        f = np.exp(-np.power(x / shapes["e"], shapes["b"]))
    return [f]


def _weibull1_3(x: FloatArray, y: FloatArray) -> dict[str, float]:
    values, coef = _refine_shapes(x, y, _weibull_decay_basis, _exponential_shape_bounds(x, y))
    d = float(coef[0]) if coef is not None else _ymax(y)
    return {"d": d, "e": values["e"], "b": values["b"]}


def _emax_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [np.ones_like(x), x / (shapes["ec50"] + x)]


def _emax(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x, y, _emax_basis, {"ec50": (1e-3 * span, 1e3 * span, max(_half_y(x, y), span / 3.0))}
    )
    if coef is None:
        return {"e0": _ymin(y), "emax": _extent(y), "ec50": span / 3.0}
    return {"e0": float(coef[0]), "emax": float(coef[1]), "ec50": values["ec50"]}


def _inhibitory_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [np.ones_like(x), -x / (shapes["ic50"] + x)]


def _inhibitory_emax(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x, y, _inhibitory_basis, {"ic50": (1e-3 * span, 1e3 * span, max(_half_y(x, y), span / 3.0))}
    )
    if coef is None:
        return {"e0": _ymax(y), "imax": _extent(y), "ic50": span / 3.0}
    return {"e0": float(coef[0]), "imax": float(coef[1]), "ic50": values["ic50"]}


def _brain_cousens_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    reciprocal = _logistic_reciprocal(x, shapes)
    return [1.0 - reciprocal, reciprocal, x * reciprocal]


def _brain_cousens(x: FloatArray, y: FloatArray) -> dict[str, float]:
    values, coef = _refine_shapes(x, y, _brain_cousens_basis, _exponential_shape_bounds(x, y))
    if coef is None:
        spread = _extent(y)
        return {
            "b": values["b"],
            "c": _ymin(y) - 0.1 * spread,
            "d": _ymax(y),
            "e": values["e"],
            "f": 0.0,
        }
    return {
        "b": values["b"],
        "c": float(coef[0]),
        "d": float(coef[1]),
        "e": values["e"],
        "f": float(coef[2]),
    }


def _cedergreen_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    reciprocal = _logistic_reciprocal(x, shapes)
    with np.errstate(all="ignore"):
        hormesis = np.exp(-1.0 / np.power(x, shapes["a"]))
    return [1.0 - reciprocal, reciprocal, hormesis * reciprocal]


def _cedergreen(x: FloatArray, y: FloatArray) -> dict[str, float]:
    shapes = _exponential_shape_bounds(x, y)
    shapes["a"] = (0.05, 20.0, 1.0)
    values, coef = _refine_shapes(x, y, _cedergreen_basis, shapes)
    if coef is None:
        spread = _extent(y)
        return {
            "a": values["a"],
            "b": values["b"],
            "c": _ymin(y) - 0.1 * spread,
            "d": _ymax(y),
            "e": values["e"],
            "f": 0.0,
        }
    return {
        "a": values["a"],
        "b": values["b"],
        "c": float(coef[0]),
        "d": float(coef[1]),
        "e": values["e"],
        "f": float(coef[2]),
    }


def _lognormal_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        z = (np.log(x) - shapes["mu"]) / (shapes["sigma"] * math.sqrt(2.0))
        s = 0.5 * (1.0 + _erf(z))
    return [1.0 - s, s]


def _dose_lognormal(x: FloatArray, y: FloatArray) -> dict[str, float]:
    positive = x > 0.0
    logs = np.log(x[positive])
    span = _span(logs) if logs.size else 1.0
    values, coef = _refine_shapes(
        x,
        y,
        _lognormal_basis,
        {
            "mu": (_median(logs) - 3.0 * span, _median(logs) + 3.0 * span, _median(logs)),
            "sigma": (1e-3 * span, 1e2 * span, max(span / 4.0, 1e-3)),
        },
    )
    if coef is None:
        spread = _extent(y)
        return {
            "c": _ymin(y) - 0.1 * spread,
            "d": _ymax(y) + 0.1 * spread,
            "mu": _median(logs),
            "sigma": max(span / 4.0, 1e-3),
        }
    return {"c": float(coef[0]), "d": float(coef[1]), "mu": values["mu"], "sigma": values["sigma"]}


# -- the models -------------------------------------------------------------------------

_P_LL2 = (
    Parameter("e", "Dose giving half of the maximal response (ED50/EC50)", lower=0.0),
    Parameter("b", "Hill slope: steepness of the curve on log-dose axes", lower=0.0),
)
_P_LL3 = (
    Parameter("d", "Upper asymptote of the response (the response at zero dose)", lower=0.0),
    Parameter("e", "Dose giving half of the maximal response (ED50/EC50)", lower=0.0),
    Parameter("b", "Hill slope: steepness of the curve on log-dose axes", lower=0.0),
)
_P_4PL = (
    Parameter("c", "Lower asymptote (the response at infinite dose)"),
    Parameter("d", "Upper asymptote (the response at zero dose)"),
    Parameter("e", "Dose at the inflection point (ED50/EC50)", lower=0.0),
    Parameter("b", "Hill slope: steepness of the curve on log-dose axes", lower=0.0),
)
_P_5PL = (
    Parameter("c", "Lower asymptote (the response at infinite dose)"),
    Parameter("d", "Upper asymptote (the response at zero dose)"),
    Parameter("e", "Dose at the inflection point (ED50/EC50)", lower=0.0),
    Parameter("b", "Hill slope: steepness of the curve on log-dose axes", lower=0.0),
    Parameter("f", "Asymmetry exponent: 1 is symmetric, other values skew the curve", lower=0.0),
)
_P_W1_3 = (
    Parameter("d", "Upper asymptote (the response at zero dose)", lower=0.0),
    Parameter("e", "Scale: the dose at which the response has fallen to 1/e of d", lower=0.0),
    Parameter("b", "Shape exponent of the Weibull curve", lower=0.0),
)
_P_EMAX = (
    Parameter("e0", "Baseline response at zero dose"),
    Parameter("emax", "Maximal effect above the baseline at saturating dose", lower=0.0),
    Parameter("ec50", "Dose giving half of the maximal effect", lower=0.0),
)
_P_INHIBITORY = (
    Parameter("e0", "Baseline response at zero dose"),
    Parameter("imax", "Maximal inhibition below the baseline at saturating dose", lower=0.0),
    Parameter("ic50", "Dose giving half of the maximal inhibition", lower=0.0),
)
_P_BRAIN_COUSENS = (
    Parameter("b", "Hill slope of the decreasing part", lower=0.0),
    Parameter("c", "Lower asymptote (the response at infinite dose)"),
    Parameter("d", "Response at zero dose"),
    Parameter("e", "Dose at the inflection point of the decreasing part", lower=0.0),
    Parameter("f", "Hormesis coefficient: extra response at low doses", lower=0.0),
)
_P_CEDER_GREEN = (
    Parameter("a", "Shape exponent of the hormesis term", lower=0.0),
    Parameter("b", "Hill slope of the decreasing part", lower=0.0),
    Parameter("c", "Lower asymptote (the response at infinite dose)"),
    Parameter("d", "Response at zero dose"),
    Parameter("e", "Dose at the inflection point of the decreasing part", lower=0.0),
    Parameter("f", "Hormesis coefficient: size of the low-dose stimulation", lower=0.0),
)
_P_LOGNORMAL = (
    Parameter("c", "Lower asymptote (the response at zero dose)"),
    Parameter("d", "Upper asymptote (the response at saturating dose)"),
    Parameter("mu", "Mean of the log-dose distribution"),
    Parameter("sigma", "Standard deviation of the log-dose distribution", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="dose_response_ll2",
        name="Log-logistic (two parameters)",
        category="dose_response",
        description="Log-logistic dose-response curve normalised between zero and one.",
        example=ModelExample({"e": 3.0, "b": 1.5}, x_range=(0.1, 20.0)),
        expression="1/(1+(x/e)**b)",
        parameters=_P_LL2,
        initial_guess=_bespoke_guess(_P_LL2, _ll2),
        domain_check=_domain_check(x_positive_for="(x/e)**b"),
        references=(_SEBAUGH, _RITZ),
        assumptions=(
            "x is a strictly positive dose; the response is bounded by 0 and 1 and "
            "decreases with the Hill slope b."
        ),
        aliases=("log_logistic_2", "ll2"),
    ),
    ModelSpec(
        id="dose_response_ll3",
        name="Log-logistic (three parameters)",
        category="dose_response",
        description="Log-logistic dose-response curve with a free upper asymptote.",
        example=ModelExample({"d": 10.0, "e": 3.0, "b": 1.5}, x_range=(0.1, 20.0)),
        expression="d/(1+(x/e)**b)",
        parameters=_P_LL3,
        initial_guess=_bespoke_guess(_P_LL3, _ll3),
        domain_check=_domain_check(x_positive_for="(x/e)**b"),
        references=(_SEBAUGH, _RITZ),
        assumptions=(
            "x is a strictly positive dose; the response falls from d towards zero and the "
            "lower asymptote is fixed at zero."
        ),
        aliases=("log_logistic_3", "ll3"),
    ),
    ModelSpec(
        id="dose_response_4pl",
        name="Four-parameter log-logistic (4PL)",
        category="dose_response",
        description="Symmetric dose-response S-curve between two free asymptotes.",
        example=ModelExample({"c": 1.0, "d": 10.0, "e": 3.0, "b": 1.4}, x_range=(0.2, 20.0)),
        expression="c+(d-c)/(1+(x/e)**b)",
        parameters=_P_4PL,
        initial_guess=_bespoke_guess(_P_4PL, _four_pl),
        domain_check=_domain_check(x_positive_for="(x/e)**b"),
        references=(_FINNEY, _RITZ),
        assumptions=(
            "x is a strictly positive dose; the log-dose curve is logistic and symmetric, "
            "so the Hill slope b fixes the steepness at ED50."
        ),
        aliases=("four_parameter_logistic", "ll4", "log_logistic_4"),
    ),
    ModelSpec(
        id="dose_response_5pl",
        name="Five-parameter log-logistic (5PL)",
        category="dose_response",
        description="Asymmetric dose-response S-curve between two free asymptotes.",
        example=ModelExample(
            {"c": 1.0, "d": 10.0, "e": 3.0, "b": 1.2, "f": 0.8}, x_range=(0.2, 20.0)
        ),
        expression="c+(d-c)/(1+(x/e)**b)**f",
        parameters=_P_5PL,
        initial_guess=_bespoke_guess(_P_5PL, _five_pl),
        domain_check=_domain_check(x_positive_for="(x/e)**b"),
        references=(_FINNEY, _RITZ),
        assumptions=(
            "x is a strictly positive dose; the asymmetry exponent f lets the two tails of "
            "the log-dose curve differ (f = 1 is the four-parameter logistic)."
        ),
        aliases=("five_parameter_logistic", "ll5", "log_logistic_5"),
    ),
    ModelSpec(
        id="dose_response_weibull1_3",
        name="Weibull type 1 (three parameters)",
        category="dose_response",
        description="Weibull survival curve falling from an upper asymptote to zero.",
        example=ModelExample({"d": 10.0, "e": 3.0, "b": 1.5}, x_range=(0.1, 20.0)),
        expression="d*exp(-(x/e)**b)",
        parameters=_P_W1_3,
        initial_guess=_bespoke_guess(_P_W1_3, _weibull1_3),
        domain_check=_domain_check(x_positive_for="(x/e)**b"),
        references=(_WEIBULL, _RITZ),
        assumptions=(
            "x is a strictly positive dose; the response falls from d to zero with the shape "
            "of a Weibull distribution function."
        ),
        aliases=("weibull_type1_3", "w1_3"),
    ),
    ModelSpec(
        id="dose_response_emax",
        name="Emax (hyperbolic) model",
        category="dose_response",
        description="Hyperbolic dose-effect curve with a baseline and a maximal effect.",
        example=ModelExample({"e0": 1.0, "emax": 8.0, "ec50": 2.0}, x_range=(0.0, 20.0)),
        expression="e0+emax*x/(ec50+x)",
        parameters=_P_EMAX,
        initial_guess=_bespoke_guess(_P_EMAX, _emax),
        domain_check=_domain_check(x_non_negative=True),
        references=(_HOLFORD,),
        assumptions=(
            "x is a non-negative dose and the effect rises hyperbolically from e0 to "
            "e0 + emax; Emax is the effect above the baseline."
        ),
        aliases=("emax_model", "hyperbolic_emax"),
    ),
    ModelSpec(
        id="dose_response_inhibitory_emax",
        name="Inhibitory Emax model",
        category="dose_response",
        description="Hyperbolic inhibition of a baseline response with dose.",
        example=ModelExample({"e0": 10.0, "imax": 8.0, "ic50": 2.0}, x_range=(0.0, 20.0)),
        expression="e0-imax*x/(ic50+x)",
        parameters=_P_INHIBITORY,
        initial_guess=_bespoke_guess(_P_INHIBITORY, _inhibitory_emax),
        domain_check=_domain_check(x_non_negative=True),
        references=(_HOLFORD,),
        assumptions=(
            "x is a non-negative dose and the response falls hyperbolically from e0; imax is "
            "the maximal inhibition below the baseline."
        ),
        aliases=("inhibitory_emax_model", "ic50_model"),
    ),
    ModelSpec(
        id="dose_response_brain_cousens",
        name="Brain-Cousens hormesis model",
        category="dose_response",
        description="Log-logistic dose response with low-dose stimulation.",
        example=ModelExample(
            {"b": 1.5, "c": 1.0, "d": 8.0, "e": 2.0, "f": 0.6}, x_range=(0.05, 20.0)
        ),
        expression="c+(d-c+f*x)/(1+(x/e)**b)",
        parameters=_P_BRAIN_COUSENS,
        initial_guess=_bespoke_guess(_P_BRAIN_COUSENS, _brain_cousens),
        domain_check=_domain_check(x_positive_for="(x/e)**b"),
        references=(_BRAIN_COUSENS, _RITZ),
        assumptions=(
            "x is a strictly positive dose; the term f*x adds a hormetic stimulation at low "
            "doses before the logistic decline takes over."
        ),
        aliases=("brain_cousens_hormesis", "bc5"),
    ),
    ModelSpec(
        id="dose_response_cedergreen",
        name="Cedergreen-Ritz-Streibig hormesis model",
        category="dose_response",
        description="Log-logistic dose response with a smooth low-dose stimulation term.",
        example=ModelExample(
            {"a": 1.5, "b": 1.2, "c": 1.0, "d": 6.0, "e": 2.0, "f": 3.0},
            x_range=(0.2, 20.0),
        ),
        expression="c+(d-c+f*exp(-1/x**a))/(1+(x/e)**b)",
        parameters=_P_CEDER_GREEN,
        initial_guess=_bespoke_guess(_P_CEDER_GREEN, _cedergreen),
        domain_check=_domain_check(x_positive_for="x**a"),
        references=(_CEDER_GREEN, _RITZ),
        assumptions=(
            "x is a strictly positive dose; f*exp(-1/x**a) adds a smooth low-dose "
            "stimulation whose shape is set by a."
        ),
        aliases=("cedergreen_ritz_streibig", "crs"),
    ),
    ModelSpec(
        id="dose_response_lognormal",
        name="Log-normal dose-response",
        category="dose_response",
        description="Probit curve in log dose: the classical tolerance-distribution model.",
        example=ModelExample({"c": 1.0, "d": 9.0, "mu": 1.1, "sigma": 0.8}, x_range=(0.2, 20.0)),
        expression="c+(d-c)*0.5*(1+erf((log(x)-mu)/(sigma*sqrt(2))))",
        parameters=_P_LOGNORMAL,
        initial_guess=_bespoke_guess(_P_LOGNORMAL, _dose_lognormal),
        domain_check=_domain_check(x_positive_for="log(x)"),
        references=(_BLISS, _FINNEY),
        assumptions=(
            "x is a strictly positive dose and the tolerance of individuals is log-normal; "
            "the response is the normal distribution function of log x."
        ),
        aliases=("log_normal_dose_response", "probit_dose_response"),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # (a + b*x)/(1 + c*x) is e0 + emax*x/(ec50 + x): the Emax model in rational form.
    ("dose_response_emax", "rational_1_1"),
    ("dose_response_inhibitory_emax", "rational_1_1"),
    # The Morgan-Mercer-Flodin ratio of power laws is the four-parameter logistic with the
    # asymptotes and the exponent written in its rational form.
    ("dose_response_4pl", "morgan_mercer_flodin"),
)
