"""Growth curves: saturating, S-shaped and monotone growth laws.

Every model is fitted on the original response scale. Where the curve has two linear
parameters once its shape parameters are fixed (an asymptote and an offset, an amplitude and
a rate, a scale and an exponent) the guess solves that linear sub-problem for a grid of shape
values and refines the shapes by golden-section search on the residual sum of squares. The
search is deterministic (no RNG) and cheap: the example data have sixty points.

Models that restate an existing curve under a different parametrisation are kept because each
form is what people search for, but they are declared in ``EQUIVALENT_PAIRS`` so the catalogue
test for accidental duplicates accepts them: Brody's law ``a*(1-b*exp(-c*x))`` is the
exponential-with-offset curve of the core starter set.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]
Basis = Callable[[FloatArray, Mapping[str, float]], list[FloatArray]]
Shapes = Mapping[str, tuple[float, float, float]]

_BAYES_BATES = (
    "Bates D.M., Watts D.G. (1988) Nonlinear Regression Analysis and Its Applications, Wiley"
)
_SEBER_WILD = "Seber G.A.F., Wild C.J. (1989) Nonlinear Regression, Wiley"
_DRAPER_SMITH = "Draper N.R., Smith H. (1998) Applied Regression Analysis, 3rd ed., Wiley"
_GOMPERTZ = (
    "Gompertz B. (1825) On the nature of the function expressive of the law of human mortality, "
    "and on a new mode of determining the value of life contingencies. Philosophical "
    "Transactions of the Royal Society of London 115:513-583"
)
_RICHARDS = (
    "Richards F.J. (1959) A flexible growth function for empirical use. Journal of Experimental "
    "Botany 10(2):290-301"
)
_VON_BERTALANFFY = (
    "von Bertalanffy L. (1938) A quantitative theory of organic growth (inquiries on growth "
    "laws II). Human Biology 10(2):181-213"
)
_BRODY = "Brody S. (1945) Bioenergetics and Growth, Reinhold, New York"
_WEIBULL = (
    "Weibull W. (1951) A statistical distribution function of wide applicability. Journal of "
    "Applied Mechanics 18(3):293-297"
)
_MMF = (
    "Morgan P.H., Mercer L.P., Flodin N.W. (1975) General model for nutritional responses of "
    "higher organisms. Proceedings of the National Academy of Sciences USA 72(11):4327-4331"
)
_BARANYI = (
    "Baranyi J., Roberts T.A. (1994) A dynamic approach to predicting bacterial growth in food. "
    "International Journal of Food Microbiology 23(3-4):277-294"
)
_ZWIETERING = (
    "Zwietering M.H., Jongenburger I., Rombouts F.M., van 't Riet K. (1990) Modeling of the "
    "bacterial growth curve. Applied and Environmental Microbiology 56(6):1875-1881"
)
_SCHUMACHER = (
    "Schumacher F.X. (1939) A new growth curve and its application to timber yield studies. "
    "Journal of Forestry 37:819-820"
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


def _scale(y: FloatArray) -> float:
    """Largest ``|y|``, or 1 when there is no data or it is all zero."""
    return _top(y) or 1.0


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

    Both endpoints must be finite with ``low < high``. A non-finite objective (an overflow or
    an invalid shape) simply loses to a finite competitor; when every value is non-finite the
    returned point is still inside the interval.
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


def _grid_values(low: float, high: float, count: int) -> list[float]:
    """Candidate shape values: geometric over a wide positive range, linear otherwise."""
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high or count < 2:
        return [low]
    if low > 0.0 and high / low >= 20.0:
        ratio = (high / low) ** (1.0 / (count - 1))
        return [low * ratio**i for i in range(count)]
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def _count_for(shapes: int) -> int:
    """Grid points per shape parameter, so the full product stays a few hundred points."""
    if shapes == 1:
        return _GRID_SINGLE
    return _GRID_THREE if shapes >= 3 else _GRID_COARSE


def _grid_search(
    values: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    objective: Callable[[Mapping[str, float]], float],
) -> dict[str, float]:
    """Best point of the full cartesian grid over the shape parameters.

    A full product, not a coordinate sweep: asymptote and rate are strongly correlated, so
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
    """Coordinate grid search over the shape parameters of a linear basis.

    ``basis(x, shapes)`` returns the columns whose linear combination is the model; for fixed
    shapes the coefficients (an asymptote and an offset, say) are solved by least squares and
    the shapes are then optimised on the resulting residual sum of squares. Each parameter is
    scanned over its whole range first (a golden-section search alone can converge to the
    wrong local optimum when the range spans orders of magnitude) and polished by a local
    golden-section search between its two neighbouring grid values.
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

    values = _grid_search(values, bounds, scored)
    values = _polish(values, bounds, scored, rounds=rounds)
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
    """``initial_guess`` from an estimator; every value is moved strictly inside the bounds.

    ``estimate`` always returns finite values for every parameter (its own fallback covers
    degenerate data); :func:`rules_guess` with literal numbers then keeps them strictly
    inside the parameter bounds.
    """
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


def _gompertz_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    exponent = np.exp(-shapes["b"] * (x - shapes["c"]))
    return [np.exp(-exponent)]


def _gompertz(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    b, c = 4.0 / span, _median(x)
    low, high = (x[0] - 2.0 * span, x[-1] + 2.0 * span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _gompertz_basis,
        {"b": (1e-3 / span, 1e3 / span, b), "c": (low, high, c)},
    )
    a = float(coef[0]) if coef is not None else _scale(y)
    return {"a": a, "b": values["b"], "c": values["c"]}


def _gompertz4_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    u = np.exp(-np.exp(-shapes["b"] * (x - shapes["d"])))
    return [1.0 - u, u]


def _gompertz4(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    b = 4.0 / span
    d = _median(x)
    low, high = (x[0] - 2.0 * span, x[-1] + 2.0 * span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _gompertz4_basis,
        {"b": (1e-3 / span, 1e3 / span, b), "d": (low, high, d)},
    )
    if coef is None:
        mean = _ymin(y)
        return {"a": mean + _extent(y), "c": mean, "b": b, "d": d}
    return {"a": float(coef[1]), "c": float(coef[0]), "b": values["b"], "d": values["d"]}


def _richards(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y**-d = a**-d + a**-d*d*exp(b*c)*exp(-b*x)`` is linear in two coefficients.

    So for fixed ``b`` and ``d`` the scale ``a`` and the location ``c`` come from a plain
    least-squares solve of the transformed response, and only two shape parameters remain to
    scan. The residual sum of squares is still evaluated on the original scale.
    """
    span = _span(x)

    def solve(b: float, d: float) -> tuple[float, float] | None:
        with np.errstate(all="ignore"):
            if not np.all(y > 0.0):
                return None
            powered = np.power(y, -d)
            if not np.all(np.isfinite(powered)):
                return None
            coef = _least_squares([np.ones(x.size), np.exp(-b * x)], powered, np.ones(x.size))
        if coef is None or coef[0] <= 0.0 or coef[1] <= 0.0 or d <= 0.0:
            return None
        ratio = float(coef[1]) / float(coef[0]) / d
        if ratio <= 0.0:
            return None
        a = float(np.power(coef[0], -1.0 / d))
        c = math.log(ratio) / b
        return (a, c) if math.isfinite(a) and math.isfinite(c) else None

    def objective(b: float, d: float) -> float:
        solved = solve(b, d)
        if solved is None:
            return math.inf
        a, c = solved
        with np.errstate(all="ignore"):
            model = a * np.power(1.0 + d * np.exp(-b * (x - c)), -1.0 / d)
            if not np.all(np.isfinite(model)):
                return math.inf
            sse = float(np.sum((y - model) ** 2))
        return sse if math.isfinite(sse) else math.inf

    values = _coordinate_scan(
        {"b": 2.0 / span, "d": 0.5},
        {"b": (1e-3 / span, 1e3 / span), "d": (0.05, 50.0)},
        lambda current: objective(current["b"], current["d"]),
    )
    solved = solve(values["b"], values["d"])
    if solved is None:
        return {"a": _ymax(y), "b": values["b"], "c": _median(x), "d": values["d"]}
    a, c = solved
    return {"a": a, "b": values["b"], "c": c, "d": values["d"]}


def _von_bertalanffy_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [np.power(1.0 - np.exp(-shapes["b"] * (x - shapes["c"])), 3.0)]


def _von_bertalanffy(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    b, c = 1.0 / span, x[0] - 0.1 * span if x.size else 0.0
    low, high = (x[0] - 2.0 * span, x[-1] + 2.0 * span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _von_bertalanffy_basis,
        {"b": (1e-3 / span, 1e3 / span, b), "c": (low, high, c)},
    )
    a = float(coef[0]) if coef is not None else _ymax(y)
    return {"a": a, "b": values["b"], "c": values["c"]}


def _brody_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [np.ones_like(x), np.exp(-shapes["c"] * x)]


def _brody(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(x, y, _brody_basis, {"c": (1e-3 / span, 1e3 / span, 1.0 / span)})
    a = float(coef[0]) if coef is not None else _ymax(y)
    b = -float(coef[1]) / a if coef is not None and a != 0.0 else 0.5
    return {"a": a, "b": b, "c": values["c"]}


def _chapman_richards_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    return [np.power(1.0 - np.exp(-shapes["b"] * x), shapes["c"])]


def _chapman_richards(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    b, c = 1.0 / span, 1.5
    values, coef = _refine_shapes(
        x,
        y,
        _chapman_richards_basis,
        {"b": (1e-3 / span, 1e3 / span, b), "c": (0.05, 20.0, c)},
    )
    a = float(coef[0]) if coef is not None else _ymax(y)
    return {"a": a, "b": values["b"], "c": values["c"]}


def _weibull_growth_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        f = 1.0 - np.exp(-np.power(x / shapes["b"], shapes["c"]))
    return [1.0 - f, f]


def _weibull_growth(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span, top = _span(x), _ymax(y)
    b, c = span / 2.0, 1.5
    values, coef = _refine_shapes(
        x,
        y,
        _weibull_growth_basis,
        {"b": (1e-3 * span, 1e3 * span, b), "c": (0.05, 20.0, c)},
    )
    if coef is None:
        return {"a": top + _extent(y), "b": b, "c": c, "d": _ymin(y)}
    return {"a": float(coef[1]), "b": values["b"], "c": values["c"], "d": float(coef[0])}


def _morgan_mercer_flodin(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)

    def solve(d: float, scale: float) -> tuple[float, float] | None:
        with np.errstate(all="ignore"):
            b = float(np.power(scale, d))
            xd = np.power(x, d)
            denominator = b + xd
            if not (math.isfinite(b) and np.all(np.isfinite(denominator))):
                return None
            if np.any(denominator == 0.0):
                return None
            coef = _least_squares([np.full(x.size, b), xd], y * denominator, np.ones(x.size))
        if coef is None:
            return None
        return float(coef[0]), float(coef[1])

    def objective(d: float, scale: float) -> float:
        solved = solve(d, scale)
        if solved is None:
            return math.inf
        a, c = solved
        with np.errstate(all="ignore"):
            model = (a * float(np.power(scale, d)) + c * np.power(x, d)) / (
                float(np.power(scale, d)) + np.power(x, d)
            )
            if not np.all(np.isfinite(model)):
                return math.inf
            sse = float(np.sum((y - model) ** 2))
        return sse if math.isfinite(sse) else math.inf

    def scored(current: Mapping[str, float]) -> float:
        return objective(current["d"], math.exp(current["log_scale"]))

    values = _coordinate_scan(
        {"d": 1.0, "log_scale": math.log(span) if span > 0 else 0.0},
        {"d": (0.05, 10.0), "log_scale": (math.log(span) - 6.0, math.log(span) + 6.0)},
        scored,
        rounds=3,
    )
    d, log_scale = values["d"], values["log_scale"]
    solved = solve(d, math.exp(log_scale))
    if solved is None:
        return {"a": _ymax(y), "b": span, "c": _ymin(y), "d": 1.0}
    a, c = solved
    return {"a": a, "b": math.exp(log_scale) ** d, "c": c, "d": d}


def _baranyi_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    decay = np.exp(-shapes["mu"] * np.maximum(x - shapes["lag"], 0.0))
    return [decay, 1.0 - decay]


def _baranyi_lag(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    low_y, high_y = _ymin(y), _ymax(y)
    threshold = low_y + 0.02 * (high_y - low_y)
    rising = np.nonzero(y > threshold)[0] if y.size else np.array([], dtype=int)
    lag = float(x[rising[0]]) if rising.size else (float(x[0]) if x.size else 0.0)
    low, high = (x[0] - span, x[-1] + span) if x.size else (-span, span)
    values, coef = _refine_shapes(
        x,
        y,
        _baranyi_basis,
        {"mu": (1e-3 / span, 1e3 / span, 2.0 / span), "lag": (low, high, lag)},
    )
    if coef is None:
        return {"y0": low_y, "ymax": high_y, "mu": 1.0 / span, "lag": lag}
    return {"y0": float(coef[0]), "ymax": float(coef[1]), "mu": values["mu"], "lag": values["lag"]}


def _generalized_vb(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)

    def solve(c: float, d: float) -> tuple[float, float] | None:
        with np.errstate(all="ignore"):
            if not np.all(y > 0.0):
                return None
            powered = np.power(y, 1.0 / d)
            if not np.all(np.isfinite(powered)):
                return None
            coef = _least_squares([np.ones(x.size), np.exp(-c * x)], powered, np.ones(x.size))
        if coef is None or coef[0] <= 0.0:
            return None
        b = -coef[1] / coef[0]
        a = float(np.power(coef[0], d))
        if not (math.isfinite(a) and math.isfinite(b)):
            return None
        return a, b

    def objective(c: float, d: float) -> float:
        solved = solve(c, d)
        if solved is None:
            return math.inf
        a, b = solved
        with np.errstate(all="ignore"):
            model = a * np.power(1.0 - b * np.exp(-c * x), d)
            if not np.all(np.isfinite(model)):
                return math.inf
            sse = float(np.sum((y - model) ** 2))
        return sse if math.isfinite(sse) else math.inf

    values = _coordinate_scan(
        {"c": 1.0 / span, "d": 1.0},
        {"c": (1e-3 / span, 1e3 / span), "d": (0.05, 20.0)},
        lambda current: objective(current["c"], current["d"]),
        rounds=3,
    )
    c, d = values["c"], values["d"]
    solved = solve(c, d)
    if solved is None:
        return {"a": _ymax(y), "b": 0.5, "c": c, "d": d}
    a, b = solved
    return {"a": a, "b": b, "c": c, "d": d}


def _schumacher(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    if x.size == 0 or not np.all(y > 0.0):
        return {"a": _scale(y), "b": 1.0, "c": span}
    lower = max(-float(np.min(x)), 0.0) + 1e-3 * span

    def objective(c: float) -> float:
        with np.errstate(all="ignore"):
            if np.any(x + c <= 0.0):
                return math.inf
            coef = _least_squares([np.ones(x.size), 1.0 / (x + c)], np.log(y), y**2)
            if coef is None:
                return math.inf
            model = np.exp(coef[0] + coef[1] / (x + c))
            if not np.all(np.isfinite(model)):
                return math.inf
            sse = float(np.sum((y - model) ** 2))
        return sse if math.isfinite(sse) else math.inf

    values = _coordinate_scan(
        {"c": lower}, {"c": (lower, lower + 10.0 * span)}, lambda current: objective(current["c"])
    )
    c = values["c"]
    with np.errstate(all="ignore"):
        coef = _least_squares([np.ones(x.size), 1.0 / (x + c)], np.log(y), y**2)
    if coef is None:
        return {"a": _ymax(y), "b": 1.0, "c": c}
    a = float(np.exp(coef[0]))
    b = float(-coef[1])
    return {"a": a, "b": b, "c": c}


def _allometric_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        return [np.power(x, shapes["b"]), np.ones_like(x)]


def _allometric_offset(x: FloatArray, y: FloatArray) -> dict[str, float]:
    values, coef = _refine_shapes(x, y, _allometric_basis, {"b": (-5.0, 5.0, 1.0)})
    if coef is None:
        return {"a": _scale(y), "b": 0.0, "c": _ymin(y)}
    return {"a": float(coef[0]), "b": values["b"], "c": float(coef[1])}


def _exponential_linear_basis(x: FloatArray, shapes: Mapping[str, float]) -> list[FloatArray]:
    with np.errstate(all="ignore"):
        return [np.exp(shapes["b"] * x), x]


def _exponential_linear_growth(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    values, coef = _refine_shapes(
        x,
        y,
        _exponential_linear_basis,
        {"b": (-10.0 / span, 10.0 / span, 1.0 / span)},
    )
    if coef is None:
        return {"a": _scale(y), "b": 0.0, "c": 0.0}
    return {"a": float(coef[0]), "b": values["b"], "c": float(coef[1])}


def _exponential_power_growth(x: FloatArray, y: FloatArray) -> dict[str, float]:
    keep = (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(keep)) < 3:
        return {"a": _scale(y), "b": 0.0, "c": 0.0}
    xk, yk = x[keep], y[keep]
    with np.errstate(all="ignore"):
        coef = _least_squares([np.ones(xk.size), xk, np.log(xk)], np.log(yk), yk**2)
    if coef is None or not math.isfinite(float(coef[0])):
        return {"a": _scale(y), "b": 0.0, "c": 0.0}
    return {"a": float(np.exp(min(coef[0], 700.0))), "b": float(coef[1]), "c": float(coef[2])}


# -- the models -------------------------------------------------------------------------

_P_GOMPERTZ = (
    Parameter("a", "Upper asymptote: the plateau y approaches as x grows", lower=0.0),
    Parameter("b", "Rate of the exponential decay of the specific growth rate", lower=0.0),
    Parameter("c", "Inflection point: the x at which growth is fastest"),
)
_P_GOMPERTZ4 = (
    Parameter("a", "Upper asymptote of the growth curve"),
    Parameter("c", "Lower asymptote: the value y starts from"),
    Parameter("b", "Rate at which the curve approaches its upper asymptote", lower=0.0),
    Parameter("d", "Inflection point: the x of fastest growth"),
)
_P_RICHARDS = (
    Parameter("a", "Upper asymptote of the growth curve", lower=0.0),
    Parameter("b", "Rate parameter of the generalised logistic curve", lower=0.0),
    Parameter("c", "Location: the x at which the curve is at its steepest"),
    Parameter("d", "Shape: how far the curve departs from a symmetric logistic", lower=0.05),
)
_P_VON_BERTALANFFY = (
    Parameter("a", "Asymptotic size (the plateau)", lower=0.0),
    Parameter("b", "Growth-rate constant", lower=0.0),
    Parameter("c", "Age at which size would be zero (the x-intercept)"),
)
_P_BRODY = (
    Parameter("a", "Asymptote approached as x grows", lower=0.0),
    Parameter("b", "Fraction of the asymptote still to grow at x = 0", lower=0.0, upper=1.0),
    Parameter("c", "Rate constant of the exponential approach", lower=0.0),
)
_P_CHAPMAN_RICHARDS = (
    Parameter("a", "Asymptote approached as x grows", lower=0.0),
    Parameter("b", "Rate constant of the exponential term", lower=0.0),
    Parameter("c", "Shape exponent: how strongly growth is damped", lower=0.0),
)
_P_WEIBULL_GROWTH = (
    Parameter("a", "Upper asymptote of the growth curve"),
    Parameter("b", "Scale: the x at which growth has completed a fixed fraction", lower=0.0),
    Parameter("c", "Shape exponent of the Weibull response", lower=0.0),
    Parameter("d", "Lower asymptote: the value y starts from"),
)
_P_MMF = (
    Parameter("a", "Value of y at x = 0 (the lower end of the curve)"),
    Parameter("b", "Scale parameter with units of x raised to the shape exponent", lower=0.0),
    Parameter("c", "Upper asymptote approached as x grows"),
    Parameter("d", "Shape exponent controlling the curvature", lower=0.0),
)
_P_BARANYI = (
    Parameter("y0", "Initial level during the lag phase"),
    Parameter("ymax", "Plateau level reached after the growth phase"),
    Parameter("mu", "Specific growth rate once the lag phase is over", lower=0.0),
    Parameter("lag", "Length of the zero-growth lag phase"),
)
_P_GENERALIZED_VB = (
    Parameter("a", "Asymptotic size", lower=0.0),
    Parameter("b", "Scale of the mass-balance term: y(0) = a*(1-b)**d", lower=0.0, upper=1.0),
    Parameter("c", "Rate constant of the exponential term", lower=0.0),
    Parameter("d", "Shape exponent of the generalised von Bertalanffy curve", lower=0.0),
)
_P_SCHUMACHER = (
    Parameter("a", "Asymptotic value as x grows", lower=0.0),
    Parameter("b", "Scale of the reciprocal-x term", lower=0.0),
    Parameter("c", "Horizontal offset; non-negative keeps the exponent finite at x = 0", lower=0.0),
)
_P_ALLOMETRIC = (
    Parameter("a", "Scale coefficient of the power law"),
    Parameter("b", "Allometric exponent: the power applied to x"),
    Parameter("c", "Additive offset (value added to the power law)"),
)
_P_EXP_LINEAR = (
    Parameter("a", "Amplitude of the exponential term", lower=0.0),
    Parameter("b", "Exponential rate per unit x"),
    Parameter("c", "Coefficient of the linear term"),
)
_P_EXP_POWER = (
    Parameter("a", "Scale coefficient of the exponential-power law", lower=0.0),
    Parameter("b", "Exponential rate per unit x"),
    Parameter("c", "Power-law exponent"),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="gompertz",
        name="Gompertz growth",
        category="growth",
        description="S-shaped growth whose rate decays exponentially with age.",
        example=ModelExample({"a": 10.0, "b": 0.8, "c": 3.0}, x_range=(0.0, 10.0)),
        expression="a*exp(-exp(-b*(x-c)))",
        parameters=_P_GOMPERTZ,
        initial_guess=_bespoke_guess(_P_GOMPERTZ, _gompertz),
        domain_check=_domain_check(),
        references=(_GOMPERTZ, _SEBER_WILD),
        assumptions=(
            "Growth is monotone increasing from zero to the plateau a; the specific growth "
            "rate falls exponentially, so the curve is asymmetric about its inflection point."
        ),
        aliases=("gompertz_growth",),
    ),
    ModelSpec(
        id="gompertz4",
        name="Gompertz growth with lower asymptote",
        category="growth",
        description="Gompertz S-curve that starts from a non-zero lower asymptote.",
        example=ModelExample({"a": 10.0, "c": 1.0, "b": 0.9, "d": 4.0}, x_range=(0.0, 12.0)),
        expression="c+(a-c)*exp(-exp(-b*(x-d)))",
        parameters=_P_GOMPERTZ4,
        initial_guess=_bespoke_guess(_P_GOMPERTZ4, _gompertz4),
        domain_check=_domain_check(),
        references=(_GOMPERTZ, _BAYES_BATES),
        assumptions=(
            "The curve is a Gompertz law between two asymptotes c and a; the data span the "
            "inflection point d."
        ),
    ),
    ModelSpec(
        id="richards",
        name="Richards growth",
        category="growth",
        description="Generalised logistic growth with a free shape parameter.",
        example=ModelExample({"a": 10.0, "b": 0.8, "c": 3.0, "d": 0.5}, x_range=(0.0, 12.0)),
        expression="a*(1+d*exp(-b*(x-c)))**(-1/d)",
        parameters=_P_RICHARDS,
        initial_guess=_bespoke_guess(_P_RICHARDS, _richards),
        domain_check=_domain_check(),
        references=(_RICHARDS, _BAYES_BATES),
        assumptions=(
            "The response is bounded by zero and the asymptote a; the shape d is kept above "
            "0.05 so the two asymptotes remain separately identifiable."
        ),
        aliases=("richards_growth",),
    ),
    ModelSpec(
        id="von_bertalanffy",
        name="von Bertalanffy growth",
        category="growth",
        description="Cubic von Bertalanffy growth curve used for length at age.",
        example=ModelExample({"a": 10.0, "b": 0.5, "c": -0.5}, x_range=(0.0, 12.0)),
        expression="a*(1-exp(-b*(x-c)))**3",
        parameters=_P_VON_BERTALANFFY,
        initial_guess=_bespoke_guess(_P_VON_BERTALANFFY, _von_bertalanffy),
        domain_check=_domain_check(),
        references=(_VON_BERTALANFFY,),
        assumptions=(
            "Length grows as the cube of the mass-balance term; the curve is monotone and "
            "approaches the asymptote a."
        ),
    ),
    ModelSpec(
        id="brody",
        name="Brody growth",
        category="growth",
        description="Monomolecular growth: an exponential approach to an asymptote.",
        example=ModelExample({"a": 10.0, "b": 0.8, "c": 0.5}, x_range=(0.0, 8.0)),
        expression="a*(1-b*exp(-c*x))",
        parameters=_P_BRODY,
        initial_guess=_bespoke_guess(_P_BRODY, _brody),
        domain_check=_domain_check(),
        references=(_BRODY, _SEBER_WILD),
        assumptions=(
            "Growth is fastest at x = 0 and slows exponentially; the same curve as an "
            "exponential decay towards an offset, with the amplitude written as -a*b."
        ),
        aliases=("monomolecular", "mitscherlich"),
    ),
    ModelSpec(
        id="chapman_richards",
        name="Chapman-Richards growth",
        category="growth",
        description="Classic forestry growth curve: a powered exponential approach to a plateau.",
        example=ModelExample({"a": 10.0, "b": 0.6, "c": 2.0}, x_range=(0.1, 8.0)),
        expression="a*(1-exp(-b*x))**c",
        parameters=_P_CHAPMAN_RICHARDS,
        initial_guess=_bespoke_guess(_P_CHAPMAN_RICHARDS, _chapman_richards),
        domain_check=_domain_check(x_positive_for="(1-exp(-b*x))**c"),
        references=(_RICHARDS,),
        assumptions=(
            "x is a non-negative age and the curve starts at zero; c is not fixed at its "
            "von Bertalanffy value of 3, so the two forms are not interchangeable."
        ),
        aliases=("chapman_richards_growth",),
    ),
    ModelSpec(
        id="weibull_growth",
        name="Weibull growth",
        category="growth",
        description="Weibull response curve between a lower and an upper asymptote.",
        example=ModelExample({"a": 10.0, "b": 3.0, "c": 2.0, "d": 1.0}, x_range=(0.1, 10.0)),
        expression="d+(a-d)*(1-exp(-(x/b)**c))",
        parameters=_P_WEIBULL_GROWTH,
        initial_guess=_bespoke_guess(_P_WEIBULL_GROWTH, _weibull_growth),
        domain_check=_domain_check(x_positive_for="(x/b)**c"),
        references=(_WEIBULL, _SEBER_WILD),
        assumptions=(
            "x is non-negative; the curve rises from d to a with the shape of a Weibull "
            "distribution function (also called the Janoschek growth function)."
        ),
        aliases=("janoschek", "weibull_growth_4", "weibull_type1_4", "weibull_type2_4"),
    ),
    ModelSpec(
        id="morgan_mercer_flodin",
        name="Morgan-Mercer-Flodin growth",
        category="growth",
        description="Ratio of two power laws: a smooth saturation with a free shape exponent.",
        example=ModelExample({"a": 2.0, "b": 5.0, "c": 10.0, "d": 1.5}, x_range=(0.1, 10.0)),
        expression="(a*b+c*x**d)/(b+x**d)",
        parameters=_P_MMF,
        initial_guess=_bespoke_guess(_P_MMF, _morgan_mercer_flodin),
        domain_check=_domain_check(x_positive_for="x**d"),
        references=(_MMF,),
        assumptions=(
            "x is non-negative; a is the value at x = 0, c the asymptote, b a scale and d the "
            "shape (the Hossfeld IV growth function is the same equation)."
        ),
    ),
    ModelSpec(
        id="baranyi_lag",
        name="Lag, exponential growth and plateau",
        category="growth",
        description="Zero-growth lag phase followed by an exponential approach to a plateau.",
        example=ModelExample({"y0": 1.0, "ymax": 10.0, "mu": 0.5, "lag": 2.0}, x_range=(0.0, 10.0)),
        expression="y0+(ymax-y0)*(1-exp(-mu*max(x-lag,0)))",
        parameters=_P_BARANYI,
        initial_guess=_bespoke_guess(_P_BARANYI, _baranyi_lag),
        domain_check=_domain_check(),
        references=(_BARANYI, _ZWIETERING),
        assumptions=(
            "The curve is flat at y0 until the lag ends, then approaches ymax exponentially; "
            "a simplified one-phase form of the Baranyi-Roberts model."
        ),
        aliases=("baranyi_roberts_style", "lag_exponential_plateau"),
    ),
    ModelSpec(
        id="generalized_von_bertalanffy",
        name="Generalised von Bertalanffy growth",
        category="growth",
        description="von Bertalanffy growth with a free exponent on the mass-balance term.",
        example=ModelExample({"a": 12.0, "b": 0.7, "c": 0.6, "d": 1.5}, x_range=(0.0, 10.0)),
        expression="a*(1-b*exp(-c*x))**d",
        parameters=_P_GENERALIZED_VB,
        initial_guess=_bespoke_guess(_P_GENERALIZED_VB, _generalized_vb),
        domain_check=_domain_check(x_non_negative=True),
        references=(_VON_BERTALANFFY, _BAYES_BATES),
        assumptions=(
            "x is non-negative and b < 1, so the base stays positive; d is often fixed at 3 "
            "for length growth and freed here."
        ),
    ),
    ModelSpec(
        id="schumacher",
        name="Schumacher growth",
        category="growth",
        description="Reciprocal-x exponential growth law used for timber yield.",
        example=ModelExample({"a": 10.0, "b": 1.5, "c": 1.0}, x_range=(0.5, 10.0)),
        expression="a*exp(-b/(x+c))",
        parameters=_P_SCHUMACHER,
        initial_guess=_bespoke_guess(_P_SCHUMACHER, _schumacher),
        domain_check=_domain_check(x_non_negative=True),
        references=(_SCHUMACHER,),
        assumptions=(
            "x is non-negative and c is positive, so the exponent stays finite and the curve "
            "rises monotonically from a*exp(-b/c) to a."
        ),
    ),
    ModelSpec(
        id="allometric_offset",
        name="Allometric growth with offset",
        category="growth",
        description="Power-law growth on top of a constant offset.",
        example=ModelExample({"a": 2.0, "b": 0.8, "c": 1.0}, x_range=(0.5, 10.0)),
        expression="a*x**b+c",
        parameters=_P_ALLOMETRIC,
        initial_guess=_bespoke_guess(_P_ALLOMETRIC, _allometric_offset),
        domain_check=_domain_check(x_positive_for="x**b"),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is strictly positive; y is linear in x**b with constant-variance errors on y "
            "(fitted on the original scale, not by regressing log y on log x)."
        ),
        aliases=("allometric_growth",),
    ),
    ModelSpec(
        id="exponential_linear_growth",
        name="Exponential growth with a linear term",
        category="growth",
        description="Exponential growth plus a straight-line drift.",
        example=ModelExample({"a": 1.5, "b": 0.6, "c": 0.4}, x_range=(0.0, 5.0)),
        expression="a*exp(b*x)+c*x",
        parameters=_P_EXP_LINEAR,
        initial_guess=_bespoke_guess(_P_EXP_LINEAR, _exponential_linear_growth),
        domain_check=_domain_check(),
        references=(_DRAPER_SMITH, _SEBER_WILD),
        assumptions=(
            "The response is the sum of one exponential mode and a linear trend; with b = 0 "
            "the curve reduces to a straight line."
        ),
    ),
    ModelSpec(
        id="exponential_power_growth",
        name="Exponential-power growth",
        category="growth",
        description="Product of an exponential and a power law with an allometric prefactor.",
        example=ModelExample({"a": 1.5, "b": 0.2, "c": 0.5}, x_range=(0.5, 6.0)),
        expression="a*exp(b*x)*x**c",
        parameters=_P_EXP_POWER,
        initial_guess=_bespoke_guess(_P_EXP_POWER, _exponential_power_growth),
        domain_check=_domain_check(x_positive_for="x**c"),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is strictly positive; log y is linear in x and log x, and the parameters are "
            "estimated from that exact linear form before fitting on the original scale."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # a*(1-b*exp(-c*x)) = (-a*b)*exp(-c*x) + a: the exponential decay towards an offset.
    ("brody", "exp_decay_offset"),
    # The same identity written as a + b*exp(c*x) with c = -c and b = -a*b.
    ("brody", "exp_offset"),
)
