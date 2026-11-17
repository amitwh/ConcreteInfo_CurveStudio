"""Exponential models: decay and growth, from one term to ordered sums of three.

Multi-exponential sums are exchangeable as written (swapping two amplitudes and their
rates gives the same curve), so every sum here is *ordered*: the first rate is ``r`` and
each further rate is the previous one plus a non-negative gap (``d``, ``d1``, ``d2``).
That makes the parametrisation one-to-one and the recovery well posed.

Initial guesses are deterministic and NumPy-only. A single rate comes from a log-linear
fit. For a sum, the amplitudes are linear once the rates are known, so the guess scans a
small grid of ordered rates and solves the amplitudes by least squares (variable
projection); the fit then refines both. The offset variants add a constant column, the
lagged models locate the lag from the data and then fit the remainder on a shifted x.
"""

import math
from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]

# References: author, year and title are given; volume and page details are omitted where
# they were not verified.
_BATES_WATTS = (
    "Bates D.M., Watts D.G. (1988) Nonlinear Regression Analysis and Its Applications, Wiley"
)
_SEBER_WILD = "Seber G.A.F., Wild C.J. (1989) Nonlinear Regression, Wiley"
_RATKOWSKY = "Ratkowsky D.A. (1990) Handbook of Nonlinear Regression Models, Marcel Dekker"
_MALTHUS = "Malthus T.R. (1798) An Essay on the Principle of Population"
_WILLIAMS_WATTS = (
    "Williams G., Watts D.C. (1970) Non-symmetrical dielectric relaxation behaviour arising "
    "from a simple empirical decay function, Transactions of the Faraday Society"
)
_WEIBULL = (
    "Weibull W. (1951) A statistical distribution function of wide applicability, Journal of "
    "Applied Mechanics"
)
_VON_BERTALANFFY = (
    "von Bertalanffy L. (1957) Quantitative laws in metabolism and growth, Quarterly Review "
    "of Biology"
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


def _scale(values: FloatArray) -> float:
    """Largest magnitude of ``values``, or 1 when there is none or all are zero."""
    top = float(np.max(np.abs(values))) if values.size else 0.0
    return top if top > 0.0 else 1.0


def _span(x: FloatArray) -> float:
    """Extent of ``x``; its magnitude (or 1) when ``x`` is constant."""
    if x.size and float(np.ptp(x)) > 0.0:
        return float(np.ptp(x))
    return _scale(x)


def _least_squares(
    columns: Sequence[FloatArray], target: FloatArray, weights: FloatArray
) -> FloatArray | None:
    """Weighted linear least squares ``target ~ columns``; ``None`` if not determined."""
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


def _linear(columns: Sequence[FloatArray], y: FloatArray) -> tuple[FloatArray | None, float]:
    """Unweighted fit of ``y ~ columns`` and its sum of squared residuals."""
    coef = _least_squares(columns, y, np.ones(y.size))
    if coef is None:
        return None, math.inf
    resid = y - np.column_stack(columns) @ coef
    sse = float(np.sum(resid * resid))
    return (coef, sse) if math.isfinite(sse) else (None, math.inf)


def _weighted_line(u: FloatArray, v: FloatArray, weights: FloatArray) -> tuple[float, float] | None:
    """Weighted least-squares ``v = intercept + slope * u``; ``(slope, intercept)``."""
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


def _log_line(u: FloatArray, y: FloatArray, sign: float) -> tuple[float, float] | None:
    """Fit ``log(sign*y) = intercept + slope*u`` on points with ``sign*y > 0``."""
    keep = sign * y > 0.0
    if int(np.count_nonzero(keep)) < 2:
        return None
    yk = y[keep]
    return _weighted_line(u[keep], np.log(sign * yk), yk**2)


def _sign_of_peak(y: FloatArray) -> float:
    """+1 or -1: the sign of the largest-magnitude observation (+1 when no data)."""
    if y.size == 0:
        return 1.0
    return -1.0 if y[int(np.argmax(np.abs(y)))] < 0.0 else 1.0


def _log_grid(low: float, high: float, count: int) -> FloatArray:
    return np.asarray(np.geomspace(low, high, count), dtype=np.float64)


def _bespoke_guess(
    parameters: tuple[Parameter, ...], estimate: Estimator
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """``initial_guess`` from a deterministic estimator; values are moved inside the bounds."""
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _clean(x, y)
        with np.errstate(all="ignore"):
            estimates = estimate(xs, ys)
        values = {k: (v if math.isfinite(v) else 0.0) for k, v in estimates.items()}
        return rules_guess(values, bounds)(xs, ys)

    return guess


def _domain_check(
    *, x_positive_for: str | None = None
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, plus ``x > 0`` where a variable exponent needs it."""

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
                f"x must be strictly positive for {x_positive_for}: found "
                f"{_count(n, 'non-positive value')}"
            )
        return reasons

    return check


def _count(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


def _best_rate_fit(
    x: FloatArray,
    y: FloatArray,
    rate_count: int,
    *,
    growth: bool,
    offset: bool,
    grid_size: int,
) -> tuple[tuple[float, ...], FloatArray] | None:
    """Variable projection: scan ordered rates, solve the linear amplitudes exactly.

    ``growth`` uses ``exp(+r*x)`` and a rate grid tied to the data span; otherwise
    ``exp(-r*x)``. ``offset`` appends a constant column, whose coefficient comes last.
    """
    span = _span(x)
    high = 5.0 / span if growth else 30.0 / span
    rates = _log_grid(0.02 / span, high, grid_size)
    best: tuple[tuple[float, ...], FloatArray] | None = None
    best_sse = math.inf
    for chosen in combinations(range(rates.size), rate_count):
        selected = tuple(float(rates[i]) for i in chosen)
        sign = 1.0 if growth else -1.0
        columns = [np.exp(sign * rate * x) for rate in selected]
        if offset:
            columns.append(np.ones(y.size))
        coef, sse = _linear(columns, y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = (selected, coef)
    return best


# -- initial guesses --------------------------------------------------------------------


def _decay_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-linear line ``log(sign*y) = log|a| - r*x``."""
    sign = _sign_of_peak(y)
    line = _log_line(x, y, sign)
    if line is not None:
        slope, intercept = line
        rate = -slope
        amplitude = sign * math.exp(min(intercept, 700.0))
        if rate > 0.0 and math.isfinite(amplitude) and amplitude != 0.0:
            return {"a": amplitude, "r": rate}
    return {"a": _scale(y), "r": 1.0 / _span(x)}


def _decay_double_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {"a": 0.5 * _scale(y), "b": 0.5 * _scale(y), "r": 1.0 / span, "d": 1.0 / span}
    fit = _best_rate_fit(x, y, 2, growth=False, offset=False, grid_size=30)
    if fit is None:
        return fallback
    rates, coef = fit
    return {"a": float(coef[0]), "b": float(coef[1]), "r": rates[0], "d": rates[1] - rates[0]}


def _decay_triple_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    third = _scale(y) / 3.0
    fallback = {
        "a": third,
        "b": third,
        "c": third,
        "r": 1.0 / span,
        "d1": 1.0 / span,
        "d2": 1.0 / span,
    }
    fit = _best_rate_fit(x, y, 3, growth=False, offset=False, grid_size=20)
    if fit is None:
        return fallback
    rates, coef = fit
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": float(coef[2]),
        "r": rates[0],
        "d1": rates[1] - rates[0],
        "d2": rates[2] - rates[1],
    }


def _decay_double_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {
        "a": 0.5 * _scale(y),
        "b": 0.5 * _scale(y),
        "r": 1.0 / span,
        "d": 1.0 / span,
        "c": float(np.min(y)) if y.size else 0.0,
    }
    fit = _best_rate_fit(x, y, 2, growth=False, offset=True, grid_size=26)
    if fit is None:
        return fallback
    rates, coef = fit
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "r": rates[0],
        "d": rates[1] - rates[0],
        "c": float(coef[2]),
    }


def _decay_lag_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    sign = _sign_of_peak(y)
    span = _span(x)
    fallback = {"a": _scale(y), "r": 1.0 / span, "c": 0.0}
    if x.size < 3:
        return fallback
    spread = float(np.ptp(y))
    if spread <= 0.0:
        return fallback
    direction = 1.0 if y[-1] >= y[0] else -1.0
    moved = np.nonzero(direction * (y - y[0]) > 0.05 * spread)[0]
    start = int(moved[0]) if moved.size else x.size // 4
    best = fallback
    best_sse = math.inf
    for offset in (-2, -1, 0, 1, 2):
        index = min(max(start + offset, 0), x.size - 1)
        lag = max(float(x[index]), 0.0)
        keep = x > lag
        if int(np.count_nonzero(keep)) < 2:
            continue
        line = _log_line(x[keep] - lag, y[keep], sign)
        if line is None:
            continue
        slope, intercept = line
        rate = -slope
        amplitude = sign * math.exp(min(intercept, 700.0))
        if rate <= 0.0 or not math.isfinite(amplitude) or amplitude == 0.0:
            continue
        resid = amplitude * np.exp(-rate * np.maximum(x - lag, 0.0)) - y
        sse = float(np.sum(resid * resid))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = {"a": amplitude, "r": rate, "c": lag}
    return best


def _stretched_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the stretch exponent; ``log|y|`` is linear in ``x**beta``."""
    sign = _sign_of_peak(y)
    span = _span(x)
    fallback = {"a": _scale(y), "tau": span, "beta": 1.0}
    best = fallback
    best_sse = math.inf
    for beta in _log_grid(0.2, 4.0, 25):
        powered = x**beta
        line = _log_line(powered, y, sign)
        if line is None:
            continue
        slope, intercept = line
        if slope >= 0.0:
            continue
        tau = (-slope) ** (-1.0 / beta)
        amplitude = sign * math.exp(min(intercept, 700.0))
        if not math.isfinite(tau) or tau <= 0.0:
            continue
        if not math.isfinite(amplitude) or amplitude == 0.0:
            continue
        resid = amplitude * np.exp(-np.power(x / tau, beta)) - y
        sse = float(np.sum(resid * resid))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = {"a": amplitude, "tau": tau, "beta": float(beta)}
    return best


def _stretched_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {"a": _scale(y), "tau": span, "beta": 1.0, "c": 0.0}
    if x.size < 3:
        return fallback
    spread = float(np.ptp(y))
    if spread <= 0.0:
        return fallback
    base = float(np.min(y))
    best = fallback
    best_sse = math.inf
    for fraction in (0.001, 0.005, 0.02, 0.05, 0.1):
        offset = base - fraction * spread
        remainder = y - offset
        for beta in _log_grid(0.2, 4.0, 20):
            powered = x**beta
            line = _log_line(powered, remainder, 1.0)
            if line is None:
                continue
            slope, intercept = line
            if slope >= 0.0:
                continue
            tau = (-slope) ** (-1.0 / beta)
            amplitude = math.exp(min(intercept, 700.0))
            if not math.isfinite(tau) or tau <= 0.0:
                continue
            if not math.isfinite(amplitude) or amplitude == 0.0:
                continue
            resid = offset + amplitude * np.exp(-np.power(x / tau, beta)) - y
            sse = float(np.sum(resid * resid))
            if math.isfinite(sse) and sse < best_sse:
                best_sse = sse
                best = {"a": amplitude, "tau": tau, "beta": float(beta), "c": offset}
    return best


def _monomolecular_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the rate; ``y = A + B*exp(-c*x)`` is linear in ``A`` and ``B`` for fixed c."""
    span = _span(x)
    fallback = {"a": _scale(y), "b": 1.0, "c": 1.0 / span}
    best = fallback
    best_sse = math.inf
    ones = np.ones(y.size)
    for rate in _log_grid(0.02 / span, 30.0 / span, 30):
        column = np.exp(-rate * x)
        coef, sse = _linear([ones, column], y)
        if coef is None or coef[0] <= 0.0:
            continue
        deficit = -coef[1] / coef[0]
        if not (math.isfinite(deficit) and deficit > 0.0):
            continue
        if sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": deficit, "c": float(rate)}
    return best


def _biexponential_difference_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan two ordered rates; the amplitude of the difference is linear."""
    span = _span(x)
    fallback = {"a": _scale(y), "b": 1.0 / span, "d": 1.0 / span}
    rates = _log_grid(0.02 / span, 30.0 / span, 24)
    best = fallback
    best_sse = math.inf
    for first, second in combinations(range(rates.size), 2):
        slow, fast = float(rates[first]), float(rates[second])
        column = np.exp(-slow * x) - np.exp(-fast * x)
        coef, sse = _linear([column], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": slow, "d": fast - slow}
    return best


def _biexponential_difference_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {
        "a": _scale(y),
        "b": 1.0 / span,
        "d": 1.0 / span,
        "c": float(np.min(y)) if y.size else 0.0,
    }
    rates = _log_grid(0.02 / span, 30.0 / span, 22)
    ones = np.ones(y.size)
    best = fallback
    best_sse = math.inf
    for first, second in combinations(range(rates.size), 2):
        slow, fast = float(rates[first]), float(rates[second])
        column = np.exp(-slow * x) - np.exp(-fast * x)
        coef, sse = _linear([ones, column], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[1]), "b": slow, "d": fast - slow, "c": float(coef[0])}
    return best


def _growth_double_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {"a": 0.5 * _scale(y), "b": 0.5 * _scale(y), "r": 1.0 / span, "d": 1.0 / span}
    fit = _best_rate_fit(x, y, 2, growth=True, offset=False, grid_size=20)
    if fit is None:
        return fallback
    rates, coef = fit
    return {"a": float(coef[0]), "b": float(coef[1]), "r": rates[0], "d": rates[1] - rates[0]}


def _growth_double_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {
        "a": 0.5 * _scale(y),
        "b": 0.5 * _scale(y),
        "r": 1.0 / span,
        "d": 1.0 / span,
        "c": float(np.min(y)) if y.size else 0.0,
    }
    fit = _best_rate_fit(x, y, 2, growth=True, offset=True, grid_size=18)
    if fit is None:
        return fallback
    rates, coef = fit
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "r": rates[0],
        "d": rates[1] - rates[0],
        "c": float(coef[2]),
    }


def _growth_lag_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    sign = _sign_of_peak(y)
    span = _span(x)
    fallback = {"a": _scale(y), "b": 1.0 / span, "c": 0.0}
    if x.size < 3:
        return fallback
    spread = float(np.ptp(y))
    if spread <= 0.0:
        return fallback
    direction = 1.0 if y[-1] >= y[0] else -1.0
    moved = np.nonzero(direction * (y - y[0]) > 0.05 * spread)[0]
    start = int(moved[0]) if moved.size else x.size // 4
    best = fallback
    best_sse = math.inf
    for offset in (-2, -1, 0, 1, 2):
        index = min(max(start + offset, 0), x.size - 1)
        lag = max(float(x[index]), 0.0)
        keep = x > lag
        if int(np.count_nonzero(keep)) < 2:
            continue
        line = _log_line(x[keep] - lag, y[keep], sign)
        if line is None:
            continue
        slope, intercept = line
        amplitude = sign * math.exp(min(intercept, 700.0))
        if slope <= 0.0 or not math.isfinite(amplitude) or amplitude == 0.0:
            continue
        resid = amplitude * np.exp(slope * np.maximum(x - lag, 0.0)) - y
        sse = float(np.sum(resid * resid))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = {"a": amplitude, "b": slope, "c": lag}
    return best


def _growth_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the growth rate; ``y = c + a*exp(b*x)`` is linear in ``c`` and ``a``."""
    span = _span(x)
    fallback = {"a": _scale(y), "b": 1.0 / span, "c": float(np.min(y)) if y.size else 0.0}
    best = fallback
    best_sse = math.inf
    ones = np.ones(y.size)
    for rate in _log_grid(0.02 / span, 5.0 / span, 30):
        column = np.exp(rate * x)
        coef, sse = _linear([ones, column], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[1]), "b": float(rate), "c": float(coef[0])}
    return best


# -- the models -------------------------------------------------------------------------

_P_DECAY = (
    Parameter("a", "Value of y at x = 0 (amplitude of the decay)"),
    Parameter("r", "Decay rate per unit x (positive)", lower=0.0),
)
_P_DECAY_DOUBLE = (
    Parameter("a", "Amplitude of the slower component"),
    Parameter("b", "Amplitude of the faster component"),
    Parameter("r", "Rate of the slower component", lower=0.0),
    Parameter("d", "Rate gap: the faster component decays at r + d", lower=0.0),
)
_P_DECAY_TRIPLE = (
    Parameter("a", "Amplitude of the slowest component"),
    Parameter("b", "Amplitude of the middle component"),
    Parameter("c", "Amplitude of the fastest component"),
    Parameter("r", "Rate of the slowest component", lower=0.0),
    Parameter("d1", "Rate gap between the slowest and middle components", lower=0.0),
    Parameter("d2", "Rate gap between the middle and fastest components", lower=0.0),
)
_P_DECAY_DOUBLE_OFFSET = (
    Parameter("a", "Amplitude of the slower component"),
    Parameter("b", "Amplitude of the faster component"),
    Parameter("r", "Rate of the slower component", lower=0.0),
    Parameter("d", "Rate gap: the faster component decays at r + d", lower=0.0),
    Parameter("c", "Offset that y approaches as x grows"),
)
_P_DECAY_LAG = (
    Parameter("a", "Value of y at the start of the decay (amplitude)"),
    Parameter("r", "Decay rate per unit x (positive)", lower=0.0),
    Parameter("c", "Lag: the x below which y has not started decaying", lower=0.0),
)
_P_STRETCHED = (
    Parameter("a", "Value of y at x = 0 (amplitude of the decay)"),
    Parameter(
        "tau", "Time constant: y has fallen by a factor e at x = tau (when beta = 1)", lower=0.0
    ),
    Parameter("beta", "Stretch exponent: 1 is a pure exponential, below 1 a heavy tail", lower=0.0),
)
_P_STRETCHED_OFFSET = (
    Parameter("a", "Amplitude of the decaying part above the offset"),
    Parameter("tau", "Time constant of the stretched decay", lower=0.0),
    Parameter("beta", "Stretch exponent: 1 is a pure exponential", lower=0.0),
    Parameter("c", "Offset that y approaches as x grows"),
)
_P_MONOMOLECULAR = (
    Parameter("a", "Asymptote that y approaches as x grows", lower=0.0),
    Parameter("b", "Relative deficit at x = 0: y starts at a*(1 - b)", lower=0.0),
    Parameter("c", "Rate of approach to the asymptote per unit x", lower=0.0),
)
_P_BIEXPONENTIAL_DIFFERENCE = (
    Parameter("a", "Scale of the rise-and-decay curve"),
    Parameter("b", "Rate of the slower (decaying) exponential", lower=0.0),
    Parameter("d", "Rate gap: the rising term has rate b + d", lower=0.0),
)
_P_BIEXPONENTIAL_DIFFERENCE_OFFSET = (
    Parameter("a", "Scale of the rise-and-decay term"),
    Parameter("b", "Rate of the slower (decaying) exponential", lower=0.0),
    Parameter("d", "Rate gap: the rising term has rate b + d", lower=0.0),
    Parameter("c", "Baseline that y returns to as x grows"),
)
_P_GROWTH_DOUBLE = (
    Parameter("a", "Amplitude of the slower growing component"),
    Parameter("b", "Amplitude of the faster growing component"),
    Parameter("r", "Rate of the slower component", lower=0.0),
    Parameter("d", "Rate gap: the faster component grows at r + d", lower=0.0),
)
_P_GROWTH_DOUBLE_OFFSET = (
    Parameter("a", "Amplitude of the slower growing component"),
    Parameter("b", "Amplitude of the faster growing component"),
    Parameter("r", "Rate of the slower component", lower=0.0),
    Parameter("d", "Rate gap: the faster component grows at r + d", lower=0.0),
    Parameter("c", "Baseline added to both growth terms"),
)
_P_GROWTH_LAG = (
    Parameter("a", "Value of y at the end of the lag (initial value)"),
    Parameter("b", "Growth rate per unit x after the lag (positive)", lower=0.0),
    Parameter("c", "Lag: the x below which y stays at a", lower=0.0),
)
_P_GROWTH_OFFSET = (
    Parameter("a", "Amplitude of the exponential growth above the baseline"),
    Parameter("b", "Growth rate per unit x (positive)", lower=0.0),
    Parameter("c", "Baseline that y approaches as x falls"),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="exponential_decay",
        name="Exponential decay",
        category="exponential",
        description="Decay at a constant proportional rate from an initial value.",
        example=ModelExample({"a": 4.0, "r": 0.8}, x_range=(0.0, 8.0)),
        expression="a*exp(-r*x)",
        parameters=_P_DECAY,
        initial_guess=_bespoke_guess(_P_DECAY, _decay_estimate),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _SEBER_WILD),
        assumptions=(
            "y falls by a constant proportion per unit x and approaches zero; errors are "
            "additive on y (fitted on the original scale)."
        ),
    ),
    ModelSpec(
        id="exponential_decay_double",
        name="Double exponential decay",
        category="exponential",
        description="Sum of two ordered exponentials, for a fast and a slow decay component.",
        example=ModelExample({"a": 3.0, "b": 1.0, "r": 0.4, "d": 1.6}, x_range=(0.0, 10.0)),
        expression="a*exp(-r*x)+b*exp(-(r+d)*x)",
        parameters=_P_DECAY_DOUBLE,
        initial_guess=_bespoke_guess(_P_DECAY_DOUBLE, _decay_double_estimate),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _SEBER_WILD),
        assumptions=(
            "Two independent first-order decay processes add; the rates are ordered "
            "(r + d >= r) so the amplitudes and rates cannot trade places."
        ),
    ),
    ModelSpec(
        id="exponential_decay_triple",
        name="Triple exponential decay",
        category="exponential",
        description="Sum of three ordered exponentials, for slow, middle and fast components.",
        example=ModelExample(
            {"a": 2.5, "b": 1.5, "c": 1.0, "r": 0.3, "d1": 0.9, "d2": 1.8},
            x_range=(0.0, 12.0),
        ),
        expression="a*exp(-r*x)+b*exp(-(r+d1)*x)+c*exp(-(r+d1+d2)*x)",
        parameters=_P_DECAY_TRIPLE,
        initial_guess=_bespoke_guess(_P_DECAY_TRIPLE, _decay_triple_estimate),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _SEBER_WILD),
        assumptions=(
            "Three first-order decay processes add; every rate is the previous one plus a "
            "non-negative gap, which makes the parametrisation unique."
        ),
    ),
    ModelSpec(
        id="exponential_decay_double_offset",
        name="Biexponential decay with offset",
        category="exponential",
        description="Two ordered decay components on top of a constant asymptote.",
        example=ModelExample(
            {"a": 2.5, "b": 1.2, "r": 0.5, "d": 1.5, "c": 1.0}, x_range=(0.0, 10.0)
        ),
        expression="c+a*exp(-r*x)+b*exp(-(r+d)*x)",
        parameters=_P_DECAY_DOUBLE_OFFSET,
        initial_guess=_bespoke_guess(_P_DECAY_DOUBLE_OFFSET, _decay_double_offset_estimate),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _SEBER_WILD),
        assumptions=(
            "y relaxes towards the constant c through two ordered exponential components; "
            "errors are additive on y."
        ),
    ),
    ModelSpec(
        id="exponential_decay_lag",
        name="Exponential decay after a lag",
        category="exponential",
        description="Decay that starts only after a lag has passed.",
        example=ModelExample({"a": 4.0, "r": 0.8, "c": 1.5}, x_range=(0.0, 10.0)),
        expression="a*exp(-r*max(x-c,0))",
        parameters=_P_DECAY_LAG,
        initial_guess=_bespoke_guess(_P_DECAY_LAG, _decay_lag_estimate),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _RATKOWSKY),
        assumptions=(
            "y stays at its initial value until the lag c and then decays exponentially at "
            "rate r; the curve is continuous at the lag."
        ),
    ),
    ModelSpec(
        id="stretched_exponential",
        name="Stretched exponential (Kohlrausch-Williams-Watts)",
        category="exponential",
        description="Decay with a stretched time axis: a pure exponential when beta = 1.",
        example=ModelExample({"a": 4.0, "tau": 2.5, "beta": 1.6}, x_range=(0.2, 10.0)),
        expression="a*exp(-(x/tau)**beta)",
        parameters=_P_STRETCHED,
        initial_guess=_bespoke_guess(_P_STRETCHED, _stretched_estimate),
        domain_check=_domain_check(x_positive_for="(x/tau)**beta"),
        references=(_WILLIAMS_WATTS, _WEIBULL, _RATKOWSKY),
        assumptions=(
            "x is strictly positive (the exponent is fitted, so x**beta needs x > 0); beta = 1 "
            "is a pure exponential decay, beta < 1 gives a heavy tail and beta > 1 a sharper "
            "shoulder."
        ),
        aliases=("kww", "weibull_decay", "stretched_exponential_decay"),
    ),
    ModelSpec(
        id="stretched_exponential_offset",
        name="Stretched exponential with offset",
        category="exponential",
        description="Stretched exponential decay towards a non-zero asymptote.",
        example=ModelExample({"a": 3.0, "tau": 2.0, "beta": 1.4, "c": 1.5}, x_range=(0.2, 10.0)),
        expression="c+a*exp(-(x/tau)**beta)",
        parameters=_P_STRETCHED_OFFSET,
        initial_guess=_bespoke_guess(_P_STRETCHED_OFFSET, _stretched_offset_estimate),
        domain_check=_domain_check(x_positive_for="(x/tau)**beta"),
        references=(_WILLIAMS_WATTS, _RATKOWSKY),
        assumptions=(
            "x is strictly positive (the exponent is fitted); y relaxes from c + a towards c "
            "along a stretched exponential with exponent beta."
        ),
    ),
    ModelSpec(
        id="monomolecular",
        name="Monomolecular growth",
        category="growth",
        description="First-order approach to an asymptote (the monomolecular growth curve).",
        example=ModelExample({"a": 5.0, "b": 0.8, "c": 0.7}, x_range=(0.0, 10.0)),
        expression="a*(1-b*exp(-c*x))",
        parameters=_P_MONOMOLECULAR,
        initial_guess=_bespoke_guess(_P_MONOMOLECULAR, _monomolecular_estimate),
        domain_check=_domain_check(),
        references=(_VON_BERTALANFFY, _BATES_WATTS),
        assumptions=(
            "y approaches the asymptote a at a constant rate c, starting at a*(1 - b); the "
            "same curve as a constant plus one exponential term (offset asymptotic growth)."
        ),
    ),
    ModelSpec(
        id="biexponential_difference",
        name="Difference of exponentials",
        category="exponential",
        description="Rise from zero followed by decay, the difference of two exponentials.",
        example=ModelExample({"a": 5.0, "b": 0.5, "d": 1.5}, x_range=(0.0, 10.0)),
        expression="a*(exp(-b*x)-exp(-(b+d)*x))",
        parameters=_P_BIEXPONENTIAL_DIFFERENCE,
        initial_guess=_bespoke_guess(
            _P_BIEXPONENTIAL_DIFFERENCE, _biexponential_difference_estimate
        ),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _SEBER_WILD),
        assumptions=(
            "y is zero at x = 0, rises while the faster term dominates, and returns to zero "
            "as the slower term decays; the rates are ordered, so b + d > b."
        ),
    ),
    ModelSpec(
        id="biexponential_difference_offset",
        name="Difference of exponentials with offset",
        category="exponential",
        description="Rise and decay above a constant baseline.",
        example=ModelExample({"a": 5.0, "b": 0.5, "d": 1.5, "c": 1.0}, x_range=(0.0, 10.0)),
        expression="c+a*(exp(-b*x)-exp(-(b+d)*x))",
        parameters=_P_BIEXPONENTIAL_DIFFERENCE_OFFSET,
        initial_guess=_bespoke_guess(
            _P_BIEXPONENTIAL_DIFFERENCE_OFFSET, _biexponential_difference_offset_estimate
        ),
        domain_check=_domain_check(),
        references=(_BATES_WATTS, _SEBER_WILD),
        assumptions=(
            "y starts at the baseline c, rises while the faster term dominates, and returns "
            "to c; the rates are ordered, so b + d > b."
        ),
    ),
    ModelSpec(
        id="exponential_growth_double",
        name="Double exponential growth",
        category="exponential",
        description="Sum of two ordered exponential growth terms.",
        example=ModelExample({"a": 1.5, "b": 1.0, "r": 0.2, "d": 0.5}, x_range=(0.0, 5.0)),
        expression="a*exp(r*x)+b*exp((r+d)*x)",
        parameters=_P_GROWTH_DOUBLE,
        initial_guess=_bespoke_guess(_P_GROWTH_DOUBLE, _growth_double_estimate),
        domain_check=_domain_check(),
        references=(_MALTHUS, _SEBER_WILD),
        assumptions=(
            "Two independent exponential growth processes add; the rates are ordered "
            "(r + d >= r) so the amplitudes and rates cannot trade places."
        ),
    ),
    ModelSpec(
        id="exponential_growth_double_offset",
        name="Double exponential growth with offset",
        category="exponential",
        description="Two ordered growth terms above a constant baseline.",
        example=ModelExample(
            {"a": 1.2, "b": 0.8, "r": 0.2, "d": 0.4, "c": 1.0}, x_range=(0.0, 6.0)
        ),
        expression="c+a*exp(r*x)+b*exp((r+d)*x)",
        parameters=_P_GROWTH_DOUBLE_OFFSET,
        initial_guess=_bespoke_guess(_P_GROWTH_DOUBLE_OFFSET, _growth_double_offset_estimate),
        domain_check=_domain_check(),
        references=(_MALTHUS, _SEBER_WILD),
        assumptions=(
            "A constant baseline c is added to two ordered exponential growth terms; the "
            "rates satisfy r + d >= r."
        ),
    ),
    ModelSpec(
        id="exponential_growth_lag",
        name="Exponential growth after a lag",
        category="growth",
        description="Exponential growth that starts only after a lag has passed.",
        example=ModelExample({"a": 1.5, "b": 0.5, "c": 2.0}, x_range=(0.0, 10.0)),
        expression="a*exp(b*max(x-c,0))",
        parameters=_P_GROWTH_LAG,
        initial_guess=_bespoke_guess(_P_GROWTH_LAG, _growth_lag_estimate),
        domain_check=_domain_check(),
        references=(_MALTHUS, _RATKOWSKY),
        assumptions=(
            "y stays at its initial value a until the lag c and then grows exponentially at "
            "rate b; the curve is continuous at the lag."
        ),
    ),
    ModelSpec(
        id="exponential_growth_offset",
        name="Exponential growth with offset",
        category="exponential",
        description="Exponential growth above a constant baseline.",
        example=ModelExample({"a": 1.5, "b": 0.4, "c": 2.0}, x_range=(0.0, 6.0)),
        expression="c+a*exp(b*x)",
        parameters=_P_GROWTH_OFFSET,
        initial_guess=_bespoke_guess(_P_GROWTH_OFFSET, _growth_offset_estimate),
        domain_check=_domain_check(),
        references=(_MALTHUS, _SEBER_WILD),
        assumptions=(
            "y grows exponentially at rate b above a baseline c; the same curve as the core "
            "asymptotic regression model with the sign of the rate restricted to growth."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # a*exp(-r*x) is exp_growth with the sign of the rate folded into a positive r.
    ("exponential_decay", "exp_growth"),
    # a*(1-b*exp(-c*x)) is a + B*exp(C*x): the core asymptotic regressions in a
    # growth-curve parametrisation (asymptote, initial deficit, rate).
    ("monomolecular", "exp_offset"),
    ("monomolecular", "exp_decay_offset"),
    # c + a*exp(b*x) is the core asymptotic regression with a positive rate.
    ("exponential_growth_offset", "exp_offset"),
)
