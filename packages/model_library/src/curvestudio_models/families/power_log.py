"""Power-law and logarithmic models: x raised to a fitted power, and functions of log x.

The power models cover the generalised power law (with an offset), the shifted power, the
modified power ``a*x**b*exp(-c*x)`` (a power rise with an exponential cut-off, the gamma-type
peak) and a sum of two ordered power laws. The logarithmic models cover ``a*ln(1+b*x)``,
shifted and polynomial-in-log variants, the linear-log model and the logarithm of a straight
line.

Every exponent is a fitted parameter, so ``x`` must be strictly positive wherever a model
raises x to a variable power; that is enforced by the applicability check. Initial guesses
are deterministic and NumPy-only: each model has a linearisation (log-log, reciprocal, or a
grid over the single nonlinear exponent with the remaining parameters solved exactly).
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
_DRAPER_SMITH = "Draper N.R., Smith H. (1998) Applied Regression Analysis, 3rd ed., Wiley"
_SEBER_WILD = "Seber G.A.F., Wild C.J. (1989) Nonlinear Regression, Wiley"
_RATKOWSKY = "Ratkowsky D.A. (1990) Handbook of Nonlinear Regression Models, Marcel Dekker"


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


def _mean(y: FloatArray) -> float:
    if not y.size:
        return 0.0
    top = _scale(y)
    return float(np.mean(y / top)) * top


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


def _exponent_grid(low: float, high: float, count: int) -> FloatArray:
    """Both signs of a positive exponent range, zero excluded."""
    positive = np.asarray(np.geomspace(low, high, count), dtype=np.float64)
    return np.concatenate((-positive[::-1], positive))


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


def _generalised_power_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the exponent; the offset and scale are linear for a fixed exponent."""
    fallback = {"a": _mean(y), "b": 0.0, "c": 1.0}
    best = fallback
    best_sse = math.inf
    ones = np.ones(y.size)
    for exponent in _exponent_grid(0.05, 3.0, 20):
        coef, sse = _linear([ones, x**exponent], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": float(coef[1]), "c": float(exponent)}
    return best


def _shifted_power_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the shift; ``log|y|`` is linear in ``log(x + c)`` for a fixed shift."""
    span = _span(x)
    sign = _sign_of_peak(y)
    fallback = {"a": _scale(y), "b": 1.0, "c": span}
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    candidates = [0.0, *(float(c) for c in _log_grid(0.005 * top, 20.0 * top, 24))]
    best = fallback
    best_sse = math.inf
    for shift in candidates:
        line = _log_line(np.log(x + shift), y, sign)
        if line is None:
            continue
        slope, intercept = line
        amplitude = sign * math.exp(min(intercept, 700.0))
        if not math.isfinite(amplitude) or amplitude == 0.0:
            continue
        resid = amplitude * np.power(x + shift, slope) - y
        sse = float(np.sum(resid * resid))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = sse
            best = {"a": amplitude, "b": slope, "c": shift}
    return best


def _modified_power_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the power exponent; the scale and cut-off rate then come from a log line."""
    span = _span(x)
    sign = _sign_of_peak(y)
    fallback = {"a": _scale(y), "b": 1.0, "c": 1.0 / span}
    keep = (sign * y > 0.0) & (x > 0.0)
    best = fallback
    best_sse = math.inf
    if int(np.count_nonzero(keep)) >= 2:
        xs, ys = x[keep], y[keep]
        for exponent in _exponent_grid(0.05, 3.0, 18):
            line = _weighted_line(xs, np.log(sign * ys) - exponent * np.log(xs), ys**2)
            if line is None or line[0] >= 0.0:
                continue
            rate = -line[0]
            amplitude = sign * math.exp(min(line[1], 700.0))
            if not math.isfinite(amplitude) or amplitude == 0.0:
                continue
            resid = amplitude * np.power(x, exponent) * np.exp(-rate * x) - y
            sse = float(np.sum(resid * resid))
            if math.isfinite(sse) and sse < best_sse:
                best_sse = sse
                best = {"a": amplitude, "b": float(exponent), "c": rate}
    return best


def _modified_power_offset_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Try baselines just beyond the decaying tail, then scan the exponent (log line)."""
    span = _span(x)
    sign = _sign_of_peak(y)
    fallback = {"a": _scale(y), "b": 1.0, "c": 1.0 / span, "d": float(np.min(y)) if y.size else 0.0}
    if x.size < 3:
        return fallback
    spread = float(np.ptp(y))
    if spread <= 0.0:
        return fallback
    base = float(y[-1])
    best = fallback
    best_sse = math.inf
    keep_x = x > 0.0
    for fraction in (0.001, 0.005, 0.02, 0.05, 0.1, 0.2):
        offset = base - fraction * spread * sign
        remainder = sign * (y - offset)
        keep = keep_x & (remainder > 0.0)
        if int(np.count_nonzero(keep)) < 3:
            continue
        xs, rs = x[keep], remainder[keep]
        for exponent in _exponent_grid(0.05, 3.0, 24):
            line = _weighted_line(xs, np.log(rs) - exponent * np.log(xs), rs**2)
            if line is None or line[0] >= 0.0:
                continue
            rate = -line[0]
            amplitude = sign * math.exp(min(line[1], 700.0))
            if not math.isfinite(amplitude) or amplitude == 0.0:
                continue
            resid = offset + amplitude * np.power(x, exponent) * np.exp(-rate * x) - y
            sse = float(np.sum(resid * resid))
            if math.isfinite(sse) and sse < best_sse:
                best_sse = sse
                best = {"a": amplitude, "b": float(exponent), "c": rate, "d": offset}
    return best


def _power_sum_double_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the first exponent and the exponent gap; both amplitudes are linear."""
    fallback = {"a": 0.5 * _scale(y), "b": 1.0, "c": 0.5 * _scale(y), "e": 1.0}
    best = fallback
    best_sse = math.inf
    for exponent in _exponent_grid(0.05, 2.0, 14):
        first = x**exponent
        for gap in _log_grid(0.1, 3.0, 14):
            coef, sse = _linear([first, x ** (exponent + gap)], y)
            if coef is not None and sse < best_sse:
                best_sse = sse
                best = {
                    "a": float(coef[0]),
                    "b": float(exponent),
                    "c": float(coef[1]),
                    "e": float(gap),
                }
    return best


def _log_one_plus_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the rate of ``ln(1 + b*x)``; the scale is linear."""
    span = _span(x)
    fallback = {"a": _scale(y) / math.log1p(_span(x)), "b": 1.0 / span}
    best = fallback
    best_sse = math.inf
    for rate in _log_grid(0.01 / span, 100.0 / span, 30):
        coef, sse = _linear([np.log1p(rate * x)], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": float(rate)}
    return best


def _log_shifted_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the shift; the intercept and slope in ``ln(x + c)`` are linear."""
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    fallback = {"a": _mean(y), "b": 0.0, "c": 0.0}
    candidates = [0.0, *(float(c) for c in _log_grid(0.005 * top, 20.0 * top, 24))]
    best = fallback
    best_sse = math.inf
    ones = np.ones(y.size)
    for shift in candidates:
        coef, sse = _linear([ones, np.log(x + shift)], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": float(coef[1]), "c": shift}
    return best


def _log_quadratic_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Exact linear fit in the basis ``1, ln x, ln(x)**2``."""
    fallback = {"a": _mean(y), "b": 0.0, "c": 0.0}
    keep = x > 0.0
    u = np.log(x[keep])
    coef = _least_squares([np.ones(u.size), u, u * u], y[keep], np.ones(u.size))
    if coef is None:
        return fallback
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])}


def _log_cubic_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Exact linear fit in the basis ``1, ln x, ln(x)**2, ln(x)**3``."""
    fallback = {"a": _mean(y), "b": 0.0, "c": 0.0, "d": 0.0}
    keep = x > 0.0
    u = np.log(x[keep])
    coef = _least_squares([np.ones(u.size), u, u * u, u * u * u], y[keep], np.ones(u.size))
    if coef is None:
        return fallback
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "d": float(coef[3])}


def _linear_log_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Exact linear fit in the basis ``1, x, ln x``."""
    fallback = {"a": _mean(y), "b": 0.0, "c": 0.0}
    keep = x > 0.0
    xs, ys = x[keep], y[keep]
    coef = _least_squares([np.ones(xs.size), xs, np.log(xs)], ys, np.ones(xs.size))
    if coef is None:
        return fallback
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])}


def _square_root_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Exact linear fit in the basis ``sqrt(x), 1``."""
    fallback = {"a": _scale(y) / math.sqrt(_span(x)), "b": _mean(y)}
    keep = x >= 0.0
    coef = _least_squares(
        [np.sqrt(x[keep]), np.ones(int(np.count_nonzero(keep)))],
        y[keep],
        np.ones(int(np.count_nonzero(keep))),
    )
    if coef is None:
        return fallback
    return {"a": float(coef[0]), "b": float(coef[1])}


def _log_quadratic_shifted_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the shift; ``1, ln(x + c), ln(x + c)**2`` are linear for a fixed shift."""
    top = float(np.max(np.abs(x))) if x.size else 1.0
    top = top if top > 0.0 else 1.0
    fallback = {"a": _mean(y), "b": 0.0, "c": 0.0, "d": 0.0}
    candidates = [0.0, *(float(c) for c in _log_grid(0.005 * top, 20.0 * top, 20))]
    best = fallback
    best_sse = math.inf
    ones = np.ones(y.size)
    for shift in candidates:
        u = np.log(x + shift)
        coef, sse = _linear([ones, u, u * u], y)
        if coef is not None and sse < best_sse:
            best_sse = sse
            best = {"a": float(coef[0]), "b": float(coef[1]), "c": shift, "d": float(coef[2])}
    return best


def _log_of_linear_estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``exp(y)`` is exactly linear in x: regress it on x for the two parameters."""
    fallback = {"a": 1.0, "b": 1.0}
    with np.errstate(over="ignore"):
        target = np.exp(y)
    keep = np.isfinite(target)
    if int(np.count_nonzero(keep)) >= 2:
        coef = _least_squares(
            [np.ones(int(np.count_nonzero(keep))), x[keep]],
            target[keep],
            np.ones(int(np.count_nonzero(keep))),
        )
        if coef is not None and coef[0] > 0.0 and coef[1] > 0.0:
            return {"a": float(coef[0]), "b": float(coef[1])}
    return fallback


# -- the models -------------------------------------------------------------------------

_P_GENERALISED_POWER = (
    Parameter("a", "Offset: the value y approaches as x falls to zero (when c > 0)"),
    Parameter("b", "Scale of the power term"),
    Parameter("c", "Exponent of x: positive grows, negative falls"),
)
_P_SHIFTED_POWER = (
    Parameter("a", "Scale of the curve"),
    Parameter("b", "Exponent of the shifted x"),
    Parameter("c", "Shift: x is replaced by x + c", lower=0.0),
)
_P_MODIFIED_POWER = (
    Parameter("a", "Scale of the power term"),
    Parameter("b", "Exponent of the power rise"),
    Parameter("c", "Exponential cut-off rate per unit x", lower=0.0),
)
_P_MODIFIED_POWER_OFFSET = (
    Parameter("a", "Scale of the power term"),
    Parameter("b", "Exponent of the power rise"),
    Parameter("c", "Exponential cut-off rate per unit x", lower=0.0),
    Parameter("d", "Offset added to the power term"),
)
_P_POWER_SUM_DOUBLE = (
    Parameter("a", "Amplitude of the first power term"),
    Parameter("b", "Exponent of the first power term"),
    Parameter("c", "Amplitude of the second power term"),
    Parameter("e", "Exponent gap: the second term has exponent b + e", lower=0.0),
)
_P_LOG_ONE_PLUS = (
    Parameter("a", "Scale of the logarithm"),
    Parameter("b", "Rate: the logarithm is taken of 1 + b*x", lower=0.0),
)
_P_LOG_SHIFTED = (
    Parameter("a", "Value of y when ln(x + c) = 0"),
    Parameter("b", "Slope: change in y per unit ln(x + c)"),
    Parameter("c", "Shift of x before the logarithm", lower=0.0),
)
_P_LOG_QUADRATIC = (
    Parameter("a", "Intercept in the polynomial of ln(x)"),
    Parameter("b", "Coefficient of ln(x)"),
    Parameter("c", "Coefficient of ln(x)**2"),
)
_P_LOG_CUBIC = (
    Parameter("a", "Intercept in the polynomial of ln(x)"),
    Parameter("b", "Coefficient of ln(x)"),
    Parameter("c", "Coefficient of ln(x)**2"),
    Parameter("d", "Coefficient of ln(x)**3"),
)
_P_LINEAR_LOG = (
    Parameter("a", "Intercept of the model"),
    Parameter("b", "Linear coefficient of x"),
    Parameter("c", "Logarithmic coefficient of ln(x)"),
)
_P_SQUARE_ROOT_OFFSET = (
    Parameter("a", "Scale of sqrt(x)"),
    Parameter("b", "Value of y at x = 0 (the offset)"),
)
_P_LOG_QUADRATIC_SHIFTED = (
    Parameter("a", "Intercept in the polynomial of ln(x + c)"),
    Parameter("b", "Coefficient of ln(x + c)"),
    Parameter("c", "Shift of x before the logarithm", lower=0.0),
    Parameter("d", "Coefficient of ln(x + c)**2"),
)
_P_LOG_OF_LINEAR = (
    Parameter("a", "Constant of the positive linear argument", lower=0.0),
    Parameter("b", "Slope of the linear argument", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="generalised_power",
        name="Generalised power law",
        category="power",
        description="A power law with an additive offset, in either direction of growth.",
        example=ModelExample({"a": 2.0, "b": 3.0, "c": 1.5}, x_range=(0.5, 5.0)),
        expression="a+b*x**c",
        parameters=_P_GENERALISED_POWER,
        initial_guess=_bespoke_guess(_P_GENERALISED_POWER, _generalised_power_estimate),
        domain_check=_domain_check(x_positive_for="x**c"),
        references=(_DRAPER_SMITH, _SEBER_WILD, _RATKOWSKY),
        assumptions=(
            "x is strictly positive; y is an offset plus a power term, so c > 0 rises and "
            "c < 0 falls towards the offset a."
        ),
        aliases=("power_with_offset", "inverse_power_offset"),
    ),
    ModelSpec(
        id="shifted_power",
        name="Shifted power law",
        category="power",
        description="A power law in x plus a non-negative shift of the x axis.",
        example=ModelExample({"a": 2.0, "b": 1.3, "c": 1.5}, x_range=(0.5, 8.0)),
        expression="a*(x+c)**b",
        parameters=_P_SHIFTED_POWER,
        initial_guess=_bespoke_guess(_P_SHIFTED_POWER, _shifted_power_estimate),
        domain_check=_domain_check(x_positive_for="(x+c)**b"),
        references=(_RATKOWSKY, _SEBER_WILD),
        assumptions=(
            "x is strictly positive; the curve is a pure power law of x shifted right by c, "
            "which removes the pole that a negative exponent has at x = 0."
        ),
    ),
    ModelSpec(
        id="modified_power",
        name="Modified power (power with exponential cut-off)",
        category="power",
        description="A power-law rise multiplied by an exponential decay, so y peaks.",
        example=ModelExample({"a": 3.0, "b": 1.2, "c": 0.35}, x_range=(0.5, 15.0)),
        expression="a*x**b*exp(-c*x)",
        parameters=_P_MODIFIED_POWER,
        initial_guess=_bespoke_guess(_P_MODIFIED_POWER, _modified_power_estimate),
        domain_check=_domain_check(x_positive_for="x**b"),
        references=(_RATKOWSKY, _SEBER_WILD),
        assumptions=(
            "x is strictly positive; the power term dominates at small x and the exponential "
            "cut-off at large x, so the curve has a single peak at x = b/c when b and c are "
            "positive."
        ),
        aliases=("gamma_model", "exponential_cutoff_power"),
    ),
    ModelSpec(
        id="modified_power_offset",
        name="Modified power with offset",
        category="power",
        description="A power rise with an exponential cut-off, above a constant baseline.",
        example=ModelExample({"a": 3.0, "b": 1.2, "c": 0.35, "d": 1.0}, x_range=(0.5, 15.0)),
        expression="d+a*x**b*exp(-c*x)",
        parameters=_P_MODIFIED_POWER_OFFSET,
        initial_guess=_bespoke_guess(_P_MODIFIED_POWER_OFFSET, _modified_power_offset_estimate),
        domain_check=_domain_check(x_positive_for="x**b"),
        references=(_RATKOWSKY, _DRAPER_SMITH),
        assumptions=(
            "x is strictly positive; a constant baseline d carries the power rise, which has "
            "a single peak when b and c are positive."
        ),
    ),
    ModelSpec(
        id="power_sum_double",
        name="Sum of two power laws",
        category="power",
        description="Two power-law terms with ordered exponents.",
        example=ModelExample({"a": 3.0, "b": 1.2, "c": 1.0, "e": 0.8}, x_range=(0.5, 10.0)),
        expression="a*x**b+c*x**(b+e)",
        parameters=_P_POWER_SUM_DOUBLE,
        initial_guess=_bespoke_guess(_P_POWER_SUM_DOUBLE, _power_sum_double_estimate),
        domain_check=_domain_check(x_positive_for="x**b"),
        references=(_SEBER_WILD, _RATKOWSKY),
        assumptions=(
            "x is strictly positive; two power laws add, and the second exponent is the first "
            "plus a non-negative gap, so the terms cannot trade places."
        ),
    ),
    ModelSpec(
        id="log_one_plus",
        name="Logarithm of one plus x",
        category="logarithmic",
        description="Logarithmic growth, y = a*ln(1 + b*x), defined at x = 0.",
        example=ModelExample({"a": 3.0, "b": 0.8}, x_range=(0.0, 10.0)),
        expression="a*log(1+b*x)",
        parameters=_P_LOG_ONE_PLUS,
        initial_guess=_bespoke_guess(_P_LOG_ONE_PLUS, _log_one_plus_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_RATKOWSKY, _DRAPER_SMITH),
        assumptions=(
            "x is non-negative and y is zero at x = 0; the curve grows without a finite "
            "plateau, more slowly than any power of x."
        ),
    ),
    ModelSpec(
        id="log_shifted",
        name="Shifted logarithm",
        category="logarithmic",
        description="A straight line in ln(x + c), finite at x = 0 when c > 0.",
        example=ModelExample({"a": 1.0, "b": 2.0, "c": 1.5}, x_range=(0.5, 10.0)),
        expression="a+b*log(x+c)",
        parameters=_P_LOG_SHIFTED,
        initial_guess=_bespoke_guess(_P_LOG_SHIFTED, _log_shifted_estimate),
        domain_check=_domain_check(x_positive_for="log(x+c)"),
        references=(_DRAPER_SMITH, _SEBER_WILD),
        assumptions=(
            "x is strictly positive; y is linear in ln(x + c), and the shift c removes the "
            "pole that a plain logarithm has at x = 0."
        ),
    ),
    ModelSpec(
        id="log_quadratic",
        name="Quadratic in ln(x)",
        category="logarithmic",
        description="A parabola in the logarithm of x, for a single bend on log-linear axes.",
        example=ModelExample({"a": 1.0, "b": 1.5, "c": 0.4}, x_range=(0.5, 20.0)),
        expression="a+b*log(x)+c*log(x)**2",
        parameters=_P_LOG_QUADRATIC,
        initial_guess=_bespoke_guess(_P_LOG_QUADRATIC, _log_quadratic_estimate),
        domain_check=_domain_check(x_positive_for="log(x)"),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is strictly positive and y is a quadratic polynomial in ln(x); the three "
            "parameters are linear, so no iterative estimate is needed."
        ),
    ),
    ModelSpec(
        id="log_cubic",
        name="Cubic in ln(x)",
        category="logarithmic",
        description="A cubic polynomial in the logarithm of x, for two bends on log-linear axes.",
        example=ModelExample({"a": 1.0, "b": 1.0, "c": 0.4, "d": -0.1}, x_range=(0.5, 20.0)),
        expression="a+b*log(x)+c*log(x)**2+d*log(x)**3",
        parameters=_P_LOG_CUBIC,
        initial_guess=_bespoke_guess(_P_LOG_CUBIC, _log_cubic_estimate),
        domain_check=_domain_check(x_positive_for="log(x)"),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is strictly positive and y is a cubic polynomial in ln(x); the parameters are "
            "linear in that basis."
        ),
    ),
    ModelSpec(
        id="linear_log",
        name="Linear-log model",
        category="logarithmic",
        description="A straight line in x plus a logarithmic term.",
        example=ModelExample({"a": 1.0, "b": 0.5, "c": 2.0}, x_range=(0.5, 10.0)),
        expression="a+b*x+c*log(x)",
        parameters=_P_LINEAR_LOG,
        initial_guess=_bespoke_guess(_P_LINEAR_LOG, _linear_log_estimate),
        domain_check=_domain_check(x_positive_for="log(x)"),
        references=(_DRAPER_SMITH, _SEBER_WILD),
        assumptions=(
            "x is strictly positive; y combines a linear trend in x with a logarithmic term, "
            "and all three parameters are linear."
        ),
    ),
    ModelSpec(
        id="square_root_offset",
        name="Square root with offset",
        category="power",
        description="A straight line in sqrt(x), the fixed-exponent counterpart of a power law.",
        example=ModelExample({"a": 3.0, "b": 1.5}, x_range=(0.0, 16.0)),
        expression="a*sqrt(x)+b",
        parameters=_P_SQUARE_ROOT_OFFSET,
        initial_guess=_bespoke_guess(_P_SQUARE_ROOT_OFFSET, _square_root_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is non-negative and y is linear in sqrt(x) with constant-variance errors; b is "
            "the value at x = 0."
        ),
    ),
    ModelSpec(
        id="log_quadratic_shifted",
        name="Quadratic in ln(x + c)",
        category="logarithmic",
        description="A parabola in the logarithm of a shifted x.",
        example=ModelExample({"a": 1.0, "b": 2.0, "c": 0.5, "d": 0.5}, x_range=(0.2, 10.0)),
        expression="a+b*log(x+c)+d*log(x+c)**2",
        parameters=_P_LOG_QUADRATIC_SHIFTED,
        initial_guess=_bespoke_guess(_P_LOG_QUADRATIC_SHIFTED, _log_quadratic_shifted_estimate),
        domain_check=_domain_check(x_positive_for="log(x+c)"),
        references=(_DRAPER_SMITH, _SEBER_WILD),
        assumptions=(
            "x is strictly positive; y is a quadratic in ln(x + c), and the shift c keeps the "
            "logarithm finite near zero."
        ),
    ),
    ModelSpec(
        id="log_of_linear",
        name="Logarithm of a straight line",
        category="logarithmic",
        description="The logarithm of a positive linear function of x.",
        example=ModelExample({"a": 2.0, "b": 1.5}, x_range=(0.0, 10.0)),
        expression="log(a+b*x)",
        parameters=_P_LOG_OF_LINEAR,
        initial_guess=_bespoke_guess(_P_LOG_OF_LINEAR, _log_of_linear_estimate),
        domain_check=_domain_check(x_non_negative=True),
        references=(_RATKOWSKY, _DRAPER_SMITH),
        assumptions=(
            "x is non-negative and a + b*x is strictly positive (guaranteed by a > 0 and "
            "b > 0); exp(y) is exactly linear in x, which makes the estimate exact."
        ),
    ),
)
