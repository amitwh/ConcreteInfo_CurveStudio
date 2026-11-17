"""Saturation models: curves that rise from zero (or a baseline) towards a plateau.

The family covers the adsorption isotherms (Hill/Sips, Freundlich, Tóth, Redlich-Peterson,
Radke-Prausnitz), the functional responses of ecology (Holling II and III, Ivlev), the
enzyme kinetics of substrate inhibition (Haldane/Andrews), stock-recruitment curves
(Ricker, and Beverton-Holt as the Michaelis-Menten equation under another name) and the
simple hyperbolic saturations.

Every model is fitted on the original response scale. Initial guesses are deterministic
and NumPy-only: where a linearisation exists it is used (Hanes-Woolf, reciprocal forms);
where the parameter enters nonlinearly (Hill, Tóth) the guess scans a small grid of
exponents and solves the remaining linear parameter exactly, which is the
variable-projection idea behind ``starter.py``'s simpler estimates.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]

# References: author, year and title are given; volume and page details are omitted where
# they were not verified.
_HILL = (
    "Hill A.V. (1910) The possible effects of the aggregation of the molecules of "
    "haemoglobin on its dissociation curves, Journal of Physiology"
)
_SIPS = "Sips R. (1948) On the structure of a catalyst surface, Journal of Chemical Physics"
_FREUNDLICH = (
    "Freundlich H.M.F. (1906) Uber die Adsorption in Losungen, Zeitschrift fur physikalische Chemie"
)
_TOTH = (
    "Toth J. (1971) State equations of the solid-gas interface layers, Acta Chimica Academiae "
    "Scientiarum Hungaricae"
)
_REDLICH_PETERSON = (
    "Redlich O., Peterson D.L. (1959) A useful adsorption isotherm, Journal of Physical Chemistry"
)
_RADKE_PRAUSNITZ = (
    "Radke C.J., Prausnitz J.M. (1972) Adsorption of organic solutes from dilute aqueous "
    "solution on activated carbon, Industrial & Engineering Chemistry Fundamentals"
)
_HOLLING = (
    "Holling C.S. (1959) Some characteristics of simple types of predation and parasitism, "
    "Canadian Entomologist"
)
_IVLEV = "Ivlev V.S. (1961) Experimental Ecology of the Feeding of Fishes, Yale University Press"
_HALDANE = "Haldane J.B.S. (1930) Enzymes, Longmans, Green and Co."
_ANDREWS = (
    "Andrews J.F. (1968) A mathematical model for the continuous culture of microorganisms "
    "utilizing inhibitory substrates, Biotechnology and Bioengineering"
)
_RICKER = (
    "Ricker W.E. (1954) Stock and recruitment, Journal of the Fisheries Research Board of Canada"
)
_BEVERTON_HOLT = (
    "Beverton R.J.H., Holt S.J. (1957) On the Dynamics of Exploited Fish Populations, Fishery "
    "Investigations Series II, HMSO"
)
_RATKOWSKY = "Ratkowsky D.A. (1990) Handbook of Nonlinear Regression Models, Marcel Dekker"
_SEBER_WILD = "Seber G.A.F., Wild C.J. (1989) Nonlinear Regression, Wiley"
_DRAPER_SMITH = "Draper N.R., Smith H. (1998) Applied Regression Analysis, 3rd ed., Wiley"


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
    """Weighted linear least squares ``target ~ columns``; ``None`` if not determined.

    Columns are scaled to unit length first so ``x**n`` and ``1`` do not fight over the
    rank tolerance.
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


def _linear(columns: Sequence[FloatArray], y: FloatArray) -> tuple[FloatArray | None, float]:
    """Unweighted fit of ``y ~ columns`` and its sum of squared residuals."""
    coef = _least_squares(columns, y, np.ones(y.size))
    if coef is None:
        return None, math.inf
    resid = y - np.column_stack(columns) @ coef
    sse = float(np.sum(resid * resid))
    return (coef, sse) if math.isfinite(sse) else (None, math.inf)


def _weighted_line(u: FloatArray, v: FloatArray, weights: FloatArray) -> tuple[float, float] | None:
    """Weighted least-squares ``v = intercept + slope * u``; ``None`` if not determined."""
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


# -- applicability ----------------------------------------------------------------------


def _count(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


def _domain_check(
    *, x_positive_for: str | None = None, x_non_negative: bool = False
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, plus the restriction the curve's domain needs."""

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
        if x_non_negative and (n := int(np.count_nonzero(finite_x < 0.0))) > 0:
            reasons.append(f"x must be non-negative: found {_count(n, 'negative value')}")
        return reasons

    return check


# -- initial guesses --------------------------------------------------------------------


def _hill_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the Hill exponent and half-saturation; ``vmax`` is linear in the rest."""
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    best: dict[str, float] = {"vmax": _scale(y), "k": top / 2.0, "n": 1.0}
    best_sse = math.inf
    for n in _log_grid(0.2, 5.0, 24):
        power_n = x**n
        for k in _log_grid(0.02 * top, 20.0 * top, 24):
            fraction = power_n / (k**n + power_n)
            coef, sse = _linear([fraction], y)
            if coef is not None and sse < best_sse:
                best_sse = sse
                best = {"vmax": float(coef[0]), "k": float(k), "n": float(n)}
    return best


def _hill_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    best: dict[str, float] = {
        "c": float(np.min(y)) if y.size else 0.0,
        "vmax": _scale(y),
        "k": top / 2.0,
        "n": 1.0,
    }
    best_sse = math.inf
    for n in _log_grid(0.2, 5.0, 20):
        power_n = x**n
        for k in _log_grid(0.02 * top, 20.0 * top, 20):
            fraction = power_n / (k**n + power_n)
            coef, sse = _linear([np.ones(y.size), fraction], y)
            if coef is not None and sse < best_sse:
                best_sse = sse
                best = {"c": float(coef[0]), "vmax": float(coef[1]), "k": float(k), "n": float(n)}
    return best


def _exponential_saturation_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    best: dict[str, float] = {"a": _scale(y), "b": 1.0 / span}
    best_sse = math.inf
    for b in _log_grid(0.02 / span, 50.0 / span, 30):
        fraction = 1.0 - np.exp(-b * x)
        coef, sse = _linear([fraction], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": float(b)}
    return best


def _exponential_saturation_lag_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    span = _span(x)
    fallback = {"a": _scale(y), "b": 1.0 / span, "c": 0.0}
    if x.size < 3:
        return fallback
    spread = float(np.ptp(y))
    if spread <= 0.0:
        return fallback
    above = np.nonzero(y > float(np.min(y)) + 0.05 * spread)[0]
    start = int(above[0]) if above.size else x.size // 4
    best = fallback
    best_sse = math.inf
    for offset in (-2, -1, 0, 1, 2):
        index = min(max(start + offset, 0), x.size - 1)
        lag = max(float(x[index]), 0.0)
        delayed = np.maximum(x - lag, 0.0)
        for b in _log_grid(0.02 / span, 50.0 / span, 24):
            fraction = 1.0 - np.exp(-b * delayed)
            coef, sse = _linear([fraction], y)
            if coef is not None and sse < best_sse:
                best_sse = sse
                best = {"a": float(coef[0]), "b": float(b), "c": lag}
    return best


def _freundlich_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-log line: ``log y = log K + (1/n) log x``."""
    keep = x > 0.0
    xs, ys = x[keep], y[keep]
    sign = _sign_of_peak(ys)
    line = _log_line(np.log(xs), ys, sign)
    if line is not None:
        slope, intercept = line
        if slope > 1e-6:
            amplitude = sign * math.exp(min(intercept, 700.0))
            if math.isfinite(amplitude) and amplitude != 0.0:
                n = min(max(1.0 / slope, 1e-3), 1e3)
                return {"k": amplitude, "n": n}
    return {"k": _scale(y), "n": 1.0}


def _toth_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the heterogeneity exponent and scale; the plateau ``a`` is linear in the rest."""
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    best: dict[str, float] = {"a": _scale(y), "b": top / 2.0, "n": 1.0}
    best_sse = math.inf
    for n in _log_grid(0.2, 3.0, 18):
        power_n = x**n
        for b in _log_grid(0.02 * top, 20.0 * top, 18):
            column = x / (b**n + power_n) ** (1.0 / n)
            coef, sse = _linear([column], y)
            if coef is not None and sse < best_sse:
                best_sse = sse
                best = {"a": float(coef[0]), "b": float(b), "n": float(n)}
    return best


def _redlich_peterson_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Equation error ``y = k*x - k*a*x**g*y`` refined by Sanathanan-Koerner reweighting."""
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    best: dict[str, float] = {"k": _scale(y) / top, "a": 1.0 / top, "g": 0.7}
    best_sse = math.inf
    for g in _log_grid(0.05, 1.0, 20):
        power_g = x**g
        weights = np.ones(y.size)
        estimate: tuple[float, float] | None = None
        for _ in range(4):
            coef = _least_squares([x, -power_g * y], y, weights)
            if coef is None or coef[0] <= 0.0:
                break
            k = float(coef[0])
            a = float(coef[1])
            if not (math.isfinite(a) and a > 0.0):
                break
            estimate = (k, a)
            weights = 1.0 / (1.0 + a * power_g) ** 2
        if estimate is None:
            continue
        k, a = estimate
        resid = k * x / (1.0 + a * power_g) - y
        sse = float(np.sum(resid * resid))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = {"k": k, "a": a, "g": float(g)}
    return best


def _holling_ii_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Hanes-Woolf line ``x/y = 1/attack + handling*x`` weighted by ``y**4 / x**2``."""
    keep = (x > 0.0) & (y > 0.0)
    xp, yp = x[keep], y[keep]
    if xp.size >= 2:
        line = _weighted_line(xp, xp / yp, yp**4 / xp**2)
        if line is not None and line[0] > 0.0 and line[1] > 0.0:
            return {"attack": 1.0 / line[1], "handling": line[0]}
    top = _scale(y)
    extent = float(np.max(np.abs(x))) if x.size else 1.0
    extent = extent if extent > 0.0 else 1.0
    return {"attack": top / extent, "handling": 1.0 / top}


def _holling_iii_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Reciprocal line ``1/y = 1/vmax + (k**2/vmax) * 1/x**2`` weighted by ``y**4``."""
    keep = (x > 0.0) & (y > 0.0)
    xp, yp = x[keep], y[keep]
    if xp.size >= 2:
        line = _weighted_line(1.0 / xp**2, 1.0 / yp, yp**4)
        if line is not None and line[0] > 0.0 and line[1] > 0.0:
            vmax = 1.0 / line[1]
            return {"vmax": vmax, "k": math.sqrt(line[0] * vmax)}
    top = _scale(y)
    return {"vmax": top, "k": _span(x)}


def _haldane_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Reciprocal model ``1/y = (km/vmax)/x + 1/vmax + x/(vmax*ki)`` weighted by ``y**4``."""
    keep = (x > 0.0) & (y > 0.0)
    xp, yp = x[keep], y[keep]
    if xp.size >= 3:
        ones = np.ones(xp.size)
        coef = _least_squares([1.0 / xp, ones, xp], 1.0 / yp, yp**4)
        if coef is not None and coef[0] > 0.0 and coef[1] > 0.0 and coef[2] > 0.0:
            vmax = 1.0 / coef[1]
            return {"vmax": vmax, "km": coef[0] * vmax, "ki": coef[1] / coef[2]}
    top = _scale(y)
    return {"vmax": top, "km": _span(x), "ki": _span(x)}


def _ricker_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log line ``log(y/x) = log(a) - b*x`` on the points with ``x > 0``."""
    sign = _sign_of_peak(y)
    keep = x > 0.0
    xs, ys = x[keep], y[keep]
    line = _log_line(xs, ys / xs, sign)
    if line is not None:
        slope, intercept = line
        b = -slope
        amplitude = sign * math.exp(min(intercept, 700.0))
        if b > 0.0 and math.isfinite(amplitude) and amplitude != 0.0:
            return {"a": amplitude, "b": b}
    extent = float(np.max(np.abs(x))) if x.size else 1.0
    extent = extent if extent > 0.0 else 1.0
    return {"a": _scale(y) / extent, "b": 1.0 / _span(x)}


def _ricker_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Try baselines below the observed minimum and keep the best log-line fit."""
    span = _span(x)
    extent = float(np.max(np.abs(x))) if x.size else 1.0
    extent = extent if extent > 0.0 else 1.0
    fallback = {"c": float(np.min(y)) if y.size else 0.0, "a": _scale(y) / extent, "b": 1.0 / span}
    if x.size < 3:
        return fallback
    spread = float(np.ptp(y))
    if spread <= 0.0:
        return fallback
    base = float(np.min(y))
    best = fallback
    best_sse = math.inf
    keep_x = x > 0.0
    for fraction in (0.001, 0.005, 0.02, 0.05, 0.1):
        offset = base - fraction * spread
        remainder = y - offset
        sign = _sign_of_peak(remainder)
        xs, rs = x[keep_x], remainder[keep_x]
        line = _log_line(xs, rs / xs, sign)
        if line is None:
            continue
        slope, intercept = line
        b = -slope
        amplitude = sign * math.exp(min(intercept, 700.0))
        if b <= 0.0 or not math.isfinite(amplitude) or amplitude == 0.0:
            continue
        resid = offset + amplitude * x * np.exp(-b * x) - y
        sse = float(np.sum(resid * resid))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = {"c": offset, "a": amplitude, "b": b}
    return best


def _rectangular_hyperbola_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the half-saturation; the offset and amplitude are linear in the rest."""
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    best: dict[str, float] = {
        "c": float(np.min(y)) if y.size else 0.0,
        "a": _scale(y),
        "b": top / 2.0,
    }
    best_sse = math.inf
    for b in _log_grid(0.02 * top, 20.0 * top, 24):
        fraction = x / (b + x)
        coef, sse = _linear([np.ones(y.size), fraction], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"c": float(coef[0]), "a": float(coef[1]), "b": float(b)}
    return best


def _exponential_saturation_double_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Variable projection over two ordered rates; both amplitudes are linear."""
    span = _span(x)
    fallback = {"a": 0.5 * _scale(y), "b": 0.5 * _scale(y), "r": 1.0 / span, "d": 1.0 / span}
    rates = _log_grid(0.02 / span, 30.0 / span, 24)
    best = fallback
    best_sse = math.inf
    for first in range(rates.size - 1):
        slow = float(rates[first])
        slow_column = 1.0 - np.exp(-slow * x)
        for second in range(first + 1, rates.size):
            fast = float(rates[second])
            fast_column = 1.0 - np.exp(-fast * x)
            coef, sse = _linear([slow_column, fast_column], y)
            if coef is not None and sse < best_sse:
                best_sse = sse
                best = {"a": float(coef[0]), "b": float(coef[1]), "r": slow, "d": fast - slow}
    return best


# -- the models -------------------------------------------------------------------------

_P_HILL = (
    Parameter("vmax", "Saturation level that y approaches as x grows", lower=0.0),
    Parameter("k", "Half-saturation: the x at which y reaches vmax/2", lower=0.0),
    Parameter("n", "Hill exponent: n = 1 is hyperbolic, n > 1 is sigmoidal", lower=0.0),
)
_P_HILL_OFFSET = (
    Parameter("c", "Lower plateau: the value y approaches as x falls towards zero"),
    Parameter("vmax", "Height of the saturation above the lower plateau", lower=0.0),
    Parameter(
        "k", "Half-saturation: the x at which y is halfway between c and c + vmax", lower=0.0
    ),
    Parameter("n", "Hill exponent: n = 1 is hyperbolic, n > 1 is sigmoidal", lower=0.0),
)
_P_EXP_SATURATION = (
    Parameter("a", "Upper plateau that y approaches as x grows", lower=0.0),
    Parameter("b", "Rate: how quickly y approaches the plateau per unit x", lower=0.0),
)
_P_EXP_SATURATION_LAG = (
    Parameter("a", "Upper plateau that y approaches as x grows", lower=0.0),
    Parameter("b", "Rate: how quickly y rises once the lag has passed", lower=0.0),
    Parameter("c", "Lag: the x below which y stays at zero", lower=0.0),
)
_P_FREUNDLICH = (
    Parameter("k", "Capacity coefficient: the value of y at x = 1", lower=0.0),
    Parameter("n", "Heterogeneity exponent: y is a power law of x with exponent 1/n", lower=0.0),
)
_P_TOTH = (
    Parameter("a", "Saturation level that y approaches as x grows", lower=0.0),
    Parameter("b", "Scale of x at which the isotherm is half-saturated", lower=0.0),
    Parameter("n", "Heterogeneity exponent: n = 1 is the hyperbolic Langmuir form", lower=0.0),
)
_P_REDLICH_PETERSON = (
    Parameter("k", "Capacity coefficient: y is about k*x at small x", lower=0.0),
    Parameter("a", "Affinity coefficient: larger a shifts saturation to smaller x", lower=0.0),
    Parameter(
        "g", "Exponent: g = 1 reduces the model to the hyperbolic form", lower=0.0, upper=1.0
    ),
)
_P_HOLLING_II = (
    Parameter("attack", "Attack rate: the slope of y at x = 0", lower=0.0),
    Parameter("handling", "Handling time: the reciprocal of the plateau", lower=0.0),
)
_P_HOLLING_III = (
    Parameter("vmax", "Saturation level that y approaches at large x", lower=0.0),
    Parameter("k", "Half-saturation: the x at which y reaches vmax/2", lower=0.0),
)
_P_HALDANE = (
    Parameter("vmax", "Rate the enzyme would reach without inhibition", lower=0.0),
    Parameter("km", "Half-saturation constant of the uninhibited reaction", lower=0.0),
    Parameter("ki", "Substrate inhibition constant: larger ki means weaker inhibition", lower=0.0),
)
_P_RICKER = (
    Parameter("a", "Productivity: the slope of y at x = 0", lower=0.0),
    Parameter("b", "Density dependence: the peak is at x = 1/b", lower=0.0),
)
_P_RICKER_OFFSET = (
    Parameter("c", "Baseline: the value y approaches as x falls to zero", lower=0.0),
    Parameter("a", "Productivity: the initial slope of the recruitment term", lower=0.0),
    Parameter("b", "Density dependence: the peak of the term is at x = 1/b", lower=0.0),
)
_P_RECTANGULAR_HYPERBOLA_OFFSET = (
    Parameter("c", "Baseline that y approaches as x falls to zero"),
    Parameter("a", "Height of the hyperbolic rise above the baseline", lower=0.0),
    Parameter("b", "Half-saturation: y is halfway between c and c + a at x = b", lower=0.0),
)
_P_EXPONENTIAL_SATURATION_DOUBLE = (
    Parameter("a", "Amplitude of the slower saturating component"),
    Parameter("b", "Amplitude of the faster saturating component"),
    Parameter("r", "Rate of the slower component", lower=0.0),
    Parameter("d", "Rate gap: the faster component saturates at rate r + d", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="hill",
        name="Hill saturation",
        category="saturation",
        description="Generalised hyperbolic saturation whose steepness is set by an exponent.",
        example=ModelExample({"vmax": 10.0, "k": 3.0, "n": 2.0}, x_range=(0.5, 20.0)),
        expression="vmax*x**n/(k**n+x**n)",
        parameters=_P_HILL,
        initial_guess=_bespoke_guess(_P_HILL, _hill_estimate),
        domain_check=_domain_check(x_positive_for="x**n"),
        references=(_HILL, _SIPS, _RATKOWSKY),
        assumptions=(
            "x is strictly positive (the exponent is fitted, so x**n needs x > 0); y rises "
            "from zero to the plateau vmax and n is positive."
        ),
        aliases=("hill_equation", "sips", "langmuir_freundlich", "koble_corrigan"),
    ),
    ModelSpec(
        id="hill_offset",
        name="Hill saturation with a lower plateau",
        category="saturation",
        description="Hill saturation above a non-zero baseline, for responses that start high.",
        example=ModelExample({"c": 2.0, "vmax": 8.0, "k": 4.0, "n": 1.5}, x_range=(0.2, 25.0)),
        expression="c+vmax*x**n/(k**n+x**n)",
        parameters=_P_HILL_OFFSET,
        initial_guess=_bespoke_guess(_P_HILL_OFFSET, _hill_offset_estimate),
        domain_check=_domain_check(x_positive_for="x**n"),
        references=(_HILL, _SIPS, _SEBER_WILD),
        assumptions=("x is strictly positive; y rises from the baseline c to c + vmax, and n > 0."),
    ),
    ModelSpec(
        id="exponential_saturation",
        name="Exponential saturation",
        category="saturation",
        description="Saturation that approaches its plateau exponentially (Ivlev response).",
        example=ModelExample({"a": 8.0, "b": 0.6}, x_range=(0.0, 10.0)),
        expression="a*(1-exp(-b*x))",
        parameters=_P_EXP_SATURATION,
        initial_guess=_bespoke_guess(_P_EXP_SATURATION, _exponential_saturation_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_IVLEV, _RATKOWSKY),
        assumptions=(
            "x is non-negative, y is zero at x = 0 and approaches the plateau a at rate b; "
            "the same curve as the Ivlev and Jovanovic responses."
        ),
        aliases=("ivlev", "jovanovic", "exponential_approach"),
    ),
    ModelSpec(
        id="exponential_saturation_lag",
        name="Exponential saturation with a lag",
        category="saturation",
        description="Exponential saturation that stays at zero until a lag has passed.",
        example=ModelExample({"a": 8.0, "b": 0.6, "c": 1.5}, x_range=(0.0, 12.0)),
        expression="a*(1-exp(-b*max(x-c,0)))",
        parameters=_P_EXP_SATURATION_LAG,
        initial_guess=_bespoke_guess(_P_EXP_SATURATION_LAG, _exponential_saturation_lag_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_IVLEV, _RATKOWSKY),
        assumptions=(
            "x is non-negative; y stays at zero until the lag c, then rises exponentially to "
            "the plateau a at rate b."
        ),
    ),
    ModelSpec(
        id="freundlich",
        name="Freundlich isotherm",
        category="saturation",
        description="Empirical power-law isotherm without a finite plateau.",
        example=ModelExample({"k": 2.0, "n": 1.5}, x_range=(0.25, 16.0)),
        expression="k*x**(1/n)",
        parameters=_P_FREUNDLICH,
        initial_guess=_bespoke_guess(_P_FREUNDLICH, _freundlich_estimate),
        domain_check=_domain_check(x_positive_for="x**(1/n)"),
        references=(_FREUNDLICH, _RATKOWSKY),
        assumptions=(
            "x is strictly positive; y grows as a power of x and never reaches a plateau "
            "(unlike Langmuir-type isotherms)."
        ),
    ),
    ModelSpec(
        id="toth",
        name="Toth isotherm",
        category="saturation",
        description="Saturation with a heterogeneity exponent that bends the approach to the "
        "plateau.",
        example=ModelExample({"a": 6.0, "b": 4.0, "n": 0.8}, x_range=(0.5, 40.0)),
        expression="a*x/(b**n+x**n)**(1/n)",
        parameters=_P_TOTH,
        initial_guess=_bespoke_guess(_P_TOTH, _toth_estimate),
        domain_check=_domain_check(x_positive_for="x**n"),
        references=(_TOTH, _RATKOWSKY),
        assumptions=(
            "x is strictly positive; y rises from zero to the plateau a, and n > 0 controls "
            "how far the isotherm departs from the hyperbolic Langmuir shape."
        ),
    ),
    ModelSpec(
        id="redlich_peterson",
        name="Redlich-Peterson isotherm",
        category="saturation",
        description="Three-parameter isotherm between the hyperbolic and power-law extremes.",
        example=ModelExample({"k": 5.0, "a": 0.3, "g": 0.7}, x_range=(0.05, 15.0)),
        expression="k*x/(1+a*x**g)",
        parameters=_P_REDLICH_PETERSON,
        initial_guess=_bespoke_guess(_P_REDLICH_PETERSON, _redlich_peterson_estimate),
        domain_check=_domain_check(x_positive_for="x**g"),
        references=(_REDLICH_PETERSON, _RADKE_PRAUSNITZ, _RATKOWSKY),
        assumptions=(
            "x is strictly positive; g lies in (0, 1], so y grows without ever crossing over "
            "into a declining branch, and g = 1 is the hyperbolic Langmuir form. The "
            "Radke-Prausnitz isotherm a*x/(b + x**n) is the same equation with a/b = k, "
            "1/b = a and n = g."
        ),
        aliases=("radke_prausnitz", "radke_prausnitz_isotherm"),
    ),
    ModelSpec(
        id="holling_ii",
        name="Holling type II functional response",
        category="saturation",
        description="Saturating feeding rate driven by attack rate and handling time.",
        example=ModelExample({"attack": 2.0, "handling": 0.25}, x_range=(0.0, 20.0)),
        expression="attack*x/(1+attack*handling*x)",
        parameters=_P_HOLLING_II,
        initial_guess=_bespoke_guess(_P_HOLLING_II, _holling_ii_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_HOLLING, _BEVERTON_HOLT),
        assumptions=(
            "x is non-negative; prey handling limits the feeding rate, which approaches "
            "1/handling. The same equation as Michaelis-Menten (and the Beverton-Holt "
            "stock-recruitment curve) in its own parametrisation."
        ),
        aliases=("holling_type_ii", "functional_response_ii", "beverton_holt"),
    ),
    ModelSpec(
        id="holling_iii",
        name="Holling type III functional response",
        category="saturation",
        description="S-shaped feeding response whose initial slope is zero at zero prey.",
        example=ModelExample({"vmax": 6.0, "k": 5.0}, x_range=(0.0, 20.0)),
        expression="vmax*x**2/(k**2+x**2)",
        parameters=_P_HOLLING_III,
        initial_guess=_bespoke_guess(_P_HOLLING_III, _holling_iii_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_HOLLING, _SEBER_WILD),
        assumptions=(
            "x is non-negative and y rises sigmoidally from zero to the plateau vmax; k is the "
            "prey density at which half the plateau is reached."
        ),
    ),
    ModelSpec(
        id="haldane",
        name="Haldane substrate inhibition",
        category="kinetics",
        description="Michaelis-Menten rate reduced by inhibition from excess substrate "
        "(Andrews model).",
        example=ModelExample({"vmax": 8.0, "km": 2.0, "ki": 10.0}, x_range=(0.0, 20.0)),
        expression="vmax*x/(km+x+x**2/ki)",
        parameters=_P_HALDANE,
        initial_guess=_bespoke_guess(_P_HALDANE, _haldane_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_HALDANE, _ANDREWS),
        assumptions=(
            "x is non-negative; the rate rises hyperbolically at low substrate then falls "
            "because excess substrate inhibits the enzyme, with a peak at x = sqrt(km*ki)."
        ),
        aliases=("andrews", "substrate_inhibition"),
    ),
    ModelSpec(
        id="ricker",
        name="Ricker stock-recruitment",
        category="growth",
        description="Hump-shaped recruitment: proportional at low x and declining at high x.",
        example=ModelExample({"a": 3.0, "b": 0.4}, x_range=(0.0, 15.0)),
        expression="a*x*exp(-b*x)",
        parameters=_P_RICKER,
        initial_guess=_bespoke_guess(_P_RICKER, _ricker_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_RICKER, _BEVERTON_HOLT),
        assumptions=(
            "x is non-negative; recruitment rises in proportion to the stock and then falls "
            "as density-dependent mortality sets in, peaking at x = 1/b."
        ),
    ),
    ModelSpec(
        id="ricker_offset",
        name="Ricker stock-recruitment with a baseline",
        category="growth",
        description="Ricker hump added to a constant baseline recruitment.",
        example=ModelExample({"c": 1.5, "a": 2.5, "b": 0.35}, x_range=(0.0, 15.0)),
        expression="c+a*x*exp(-b*x)",
        parameters=_P_RICKER_OFFSET,
        initial_guess=_bespoke_guess(_P_RICKER_OFFSET, _ricker_offset_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_RICKER, _RATKOWSKY),
        assumptions=(
            "x is non-negative; a constant baseline recruitment c is added to the Ricker "
            "hump, which peaks at x = 1/b."
        ),
    ),
    ModelSpec(
        id="rectangular_hyperbola_offset",
        name="Rectangular hyperbola with offset",
        category="saturation",
        description="Hyperbolic saturation above a non-zero baseline.",
        example=ModelExample({"c": 1.5, "a": 5.0, "b": 3.0}, x_range=(0.0, 25.0)),
        expression="c+a*x/(b+x)",
        parameters=_P_RECTANGULAR_HYPERBOLA_OFFSET,
        initial_guess=_bespoke_guess(
            _P_RECTANGULAR_HYPERBOLA_OFFSET, _rectangular_hyperbola_offset_estimate
        ),
        domain_check=_domain_check(x_non_negative=True),
        references=(_DRAPER_SMITH, _SEBER_WILD),
        assumptions=(
            "x is non-negative and y rises from the baseline c towards c + a, reaching half "
            "the rise at x = b."
        ),
    ),
    ModelSpec(
        id="exponential_saturation_double",
        name="Double exponential saturation",
        category="saturation",
        description="Two saturating components with ordered rates, a fast and a slow phase.",
        example=ModelExample({"a": 5.0, "b": 3.0, "r": 0.3, "d": 0.9}, x_range=(0.0, 10.0)),
        expression="a*(1-exp(-r*x))+b*(1-exp(-(r+d)*x))",
        parameters=_P_EXPONENTIAL_SATURATION_DOUBLE,
        initial_guess=_bespoke_guess(
            _P_EXPONENTIAL_SATURATION_DOUBLE, _exponential_saturation_double_estimate
        ),
        domain_check=_domain_check(x_non_negative=True),
        references=(_IVLEV, _RATKOWSKY),
        assumptions=(
            "x is non-negative; two saturating processes add and their rates are ordered "
            "(r + d >= r), so the amplitudes and rates cannot trade places."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # k*x**(1/n) is a*x**b with b = 1/n: the Freundlich isotherm in its own parametrisation.
    ("freundlich", "power"),
    # attack*x/(1+attack*handling*x) is vmax*x/(km+x): Holling II, Michaelis-Menten and the
    # Beverton-Holt stock-recruitment curve are one equation in three parametrisations.
    ("holling_ii", "michaelis_menten"),
    ("holling_ii", "rational_x_over_linear"),
)
