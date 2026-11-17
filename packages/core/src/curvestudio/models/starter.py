"""Starter model set: nine classic one-variable curves with data-derived initial guesses.

Why the guesses matter: nonlinear least squares is only as reliable as its starting
point, so each model ships a deterministic (no RNG), NumPy-only estimate obtained by
linearising the model where that is sound (log-log for ``power``, Hanes-Woolf for
``michaelis_menten``, logit for ``logistic3``, ...) or by a shape heuristic otherwise.

Every ``initial_guess`` honours the same contract: it never raises and never returns
NaN or infinity, even for degenerate data (no points, constant ``y``, identical ``x``,
values a logarithm cannot take). In that case it falls back to a finite value derived
from the data scale. Bounded parameters are always returned strictly inside their
bounds, so the engine never starts on a bound. The guess is a starting point, not an
estimate: it is deliberately rough on noisy data and the fit refines it.
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from curvestudio.models.registry import list_models, register
from curvestudio.models.spec import ModelExample, ModelSpec, Parameter

FloatArray = NDArray[np.float64]

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
"""Full width at half maximum of a Gaussian divided by its standard deviation."""

_RATE_RANGE = (1e-3, 1e3)
"""Allowed rate constants in units of ``1 / x-span`` for the exponential guesses."""

_TAIL_MARGIN = 0.02
"""Fraction of the y-range by which the asymptote guess sits beyond the data."""


# -- numeric helpers --------------------------------------------------------------------


def _sorted_finite(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Finite ``(x, y)`` pairs ordered by ``x`` (stable); fresh arrays, inputs untouched."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    xa, ya = xa[:n], ya[:n]
    keep = np.isfinite(xa) & np.isfinite(ya)
    order = np.argsort(xa[keep], kind="stable")
    return xa[keep][order], ya[keep][order]


def _scale(y: FloatArray) -> float:
    """Largest ``|y|``, or 1 when there is no data or it is all zero."""
    top = float(np.max(np.abs(y))) if y.size else 0.0
    return top if top > 0.0 else 1.0


def _span(x: FloatArray) -> float:
    """Extent of ``x``; falls back to the magnitude of ``x`` (or 1) when it is constant."""
    if x.size == 0:
        return 1.0
    extent = float(np.ptp(x))
    if extent > 0.0:
        return extent
    magnitude = float(np.max(np.abs(x)))
    return magnitude if magnitude > 0.0 else 1.0


def _mean_or(values: FloatArray, default: float) -> float:
    return float(np.mean(values)) if values.size else default


def _positive(value: float, fallback: float) -> float:
    """``value`` if it is a finite positive number, else the (positive) ``fallback``."""
    return value if math.isfinite(value) and value > 0.0 else fallback


def _finite(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def _sign_of_peak(y: FloatArray) -> float:
    """+1 or -1: the sign of the largest-magnitude observation (+1 when no data)."""
    if y.size == 0:
        return 1.0
    return -1.0 if y[int(np.argmax(np.abs(y)))] < 0.0 else 1.0


def _weighted_line(u: FloatArray, v: FloatArray, weights: FloatArray) -> tuple[float, float] | None:
    """Weighted least-squares ``v = intercept + slope * u``; ``None`` if not determined.

    Closed form rather than ``numpy.polyfit`` so a degenerate design (one point, repeated
    ``u``) is detected explicitly instead of surfacing as a rank warning.
    """
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


def _log_fit(u: FloatArray, y: FloatArray, sign: float) -> tuple[float, float] | None:
    """Fit ``log(sign*y) = intercept + slope*u`` on points with ``sign*y > 0``.

    The least-squares weight ``y**2`` is the delta-method inverse variance of ``log y``
    when ``y`` carries constant-variance noise, so near-zero points do not dominate.
    """
    keep = sign * y > 0.0
    if int(np.count_nonzero(keep)) < 2:
        return None
    yk = y[keep]
    return _weighted_line(u[keep], np.log(sign * yk), yk**2)


# -- initial guesses --------------------------------------------------------------------


def _guess_exp_growth(x: FloatArray, y: FloatArray) -> dict[str, float]:
    xs, ys = _sorted_finite(x, y)
    sign = _sign_of_peak(ys)
    line = _log_fit(xs, ys, sign)
    if line is not None:
        slope, intercept = line
        with np.errstate(over="ignore"):
            amplitude = sign * math.exp(min(intercept, 700.0))
        if math.isfinite(amplitude) and amplitude != 0.0:
            return {"a": amplitude, "b": slope}
    mean = _mean_or(ys, 0.0)
    return {"a": mean if mean != 0.0 else _scale(ys), "b": 0.0}


def _decay_to_offset(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """``(amplitude, rate, offset)`` with ``y ~ amplitude*exp(-rate*x) + offset``, ``rate > 0``.

    Assumes the curve flattens towards the *right* end of the data (callers reverse ``x``
    for the other direction). The offset is placed just beyond the observed extreme on the
    flattening side so that ``y - offset`` keeps one sign; a weighted log-linear fit of
    that remainder then yields rate and amplitude.
    """
    xs, ys = _sorted_finite(x, y)
    span, scale = _span(xs), _scale(ys)
    slowest, fastest = _RATE_RANGE[0] / span, _RATE_RANGE[1] / span
    rate_default = 1.0 / span
    if xs.size < 3:
        return 0.5 * scale, rate_default, _mean_or(ys, 0.0)
    spread = float(np.ptp(ys))
    if spread <= 1e-12 * scale:
        return 0.5 * scale, rate_default, float(np.mean(ys))
    k = max(1, xs.size // 5)
    sign = 1.0 if float(np.mean(ys[:k])) > float(np.mean(ys[-k:])) else -1.0
    extreme = float(np.min(ys)) if sign > 0.0 else float(np.max(ys))
    offset = extreme - sign * _TAIL_MARGIN * spread
    remainder = sign * (ys - offset)  # strictly positive by construction
    line = _weighted_line(xs, np.log(remainder), remainder**2)
    if line is None or line[0] >= 0.0:
        return sign * float(remainder[0]), rate_default, offset
    rate = float(np.clip(-line[0], slowest, fastest))
    with np.errstate(over="ignore"):
        amplitude = sign * math.exp(min(line[1], 700.0))
    if not math.isfinite(amplitude):
        amplitude = sign * float(remainder[0])
    return amplitude, rate, offset


def _guess_exp_decay_offset(x: FloatArray, y: FloatArray) -> dict[str, float]:
    amplitude, rate, offset = _decay_to_offset(x, y)
    return {"a": amplitude, "b": rate, "c": offset}


def _guess_exp_offset(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``a + b*exp(c*x)``: try the asymptote at either end and keep the closer fit."""
    xs, ys = _sorted_finite(x, y)
    amp_r, rate_r, off_r = _decay_to_offset(xs, ys)  # flattens to the right: c < 0
    amp_l, rate_l, off_l = _decay_to_offset(-xs, ys)  # flattens to the left: c > 0
    right = {"a": off_r, "b": amp_r, "c": -rate_r}
    left = {"a": off_l, "b": amp_l, "c": rate_l}

    def sse(p: dict[str, float]) -> float:
        with np.errstate(all="ignore"):
            total = float(np.sum((p["a"] + p["b"] * np.exp(p["c"] * xs) - ys) ** 2))
        return total if math.isfinite(total) else math.inf

    return left if sse(left) < sse(right) else right


def _guess_log(
    transform: Callable[[FloatArray], FloatArray],
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """Guess for ``a + b*transform(x)`` by ordinary least squares on the ``x > 0`` points."""

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _sorted_finite(x, y)
        keep = xs > 0.0
        line = _weighted_line(transform(xs[keep]), ys[keep], np.ones(int(np.count_nonzero(keep))))
        if line is None:
            return {"a": _mean_or(ys, 0.0), "b": 0.0}
        return {"a": line[1], "b": line[0]}

    return guess


def _guess_power(x: FloatArray, y: FloatArray) -> dict[str, float]:
    xs, ys = _sorted_finite(x, y)
    sign = _sign_of_peak(ys)
    keep = xs > 0.0
    xp, yp = xs[keep], ys[keep]
    line = _log_fit(np.log(xp), yp, sign)
    if line is not None:
        slope, intercept = line
        with np.errstate(over="ignore"):
            amplitude = sign * math.exp(min(intercept, 700.0))
        if math.isfinite(amplitude) and amplitude != 0.0:
            return {"a": amplitude, "b": slope}
    mean = _mean_or(ys, 0.0)
    return {"a": mean if mean != 0.0 else _scale(ys), "b": 1.0}


def _guess_michaelis_menten(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Hanes-Woolf line ``x/y = km/vmax + x/vmax``, falling back to a half-saturation read.

    Hanes-Woolf is preferred over Lineweaver-Burk because it does not blow up the noise of
    the smallest ``y``; the clamps keep a wild noisy slope from yielding an absurd start.
    """
    xs, ys = _sorted_finite(x, y)
    positive = (xs > 0.0) & (ys > 0.0)
    xp, yp = xs[positive], ys[positive]
    y_top = float(np.max(yp)) if yp.size else 0.0
    ceiling = y_top if y_top > 0.0 else _scale(ys)
    x_top = float(np.max(xp)) if xp.size else _span(xs)
    line = _weighted_line(xp, xp / yp, yp**4 / xp**2) if xp.size >= 2 else None
    if line is not None and line[0] > 0.0 and line[1] > 0.0:
        vmax = float(np.clip(1.0 / line[0], 0.5 * ceiling, 1e3 * ceiling))
        km = float(np.clip(line[1] * vmax, 1e-3 * x_top, 1e3 * x_top))
        return {"vmax": vmax, "km": km}
    half = int(np.argmin(np.abs(yp - 0.5 * ceiling))) if yp.size else 0
    km = float(xp[half]) if xp.size else x_top
    return {"vmax": 1.1 * ceiling, "km": _positive(km, x_top)}


def _guess_logistic3(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Logit linearisation ``log(p/(1-p)) = r*(x - x0)`` with ``p = y/k``, ``k`` above max y."""
    xs, ys = _sorted_finite(x, y)
    span, scale = _span(xs), _scale(ys)
    y_max = float(np.max(ys)) if ys.size else 0.0
    spread = float(np.ptp(ys)) if ys.size else 0.0
    k = _positive(y_max + 0.05 * spread if y_max > 0.0 else scale, scale)
    p = ys / k
    band = (p > 0.05) & (p < 0.95)
    line = None
    if int(np.count_nonzero(band)) >= 2:
        pb = p[band]
        line = _weighted_line(xs[band], np.log(pb / (1.0 - pb)), (k * pb * (1.0 - pb)) ** 2)
    if line is not None and line[0] > 0.0:
        rate = float(np.clip(line[0], _RATE_RANGE[0] / span, _RATE_RANGE[1] / span))
        return {"k": k, "r": rate, "x0": _finite(-line[1] / line[0], float(np.median(xs)))}
    middle = float(xs[int(np.argmin(np.abs(ys - 0.5 * k)))]) if xs.size else 0.0
    return {"k": k, "r": 8.0 / span, "x0": middle}


def _guess_gaussian(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Amplitude at the largest ``|y|``; centre and width from the half-maximum crossings."""
    xs, ys = _sorted_finite(x, y)
    if xs.size == 0:
        return {"a": 1.0, "mu": 0.0, "s": 1.0}
    peak = int(np.argmax(np.abs(ys)))
    amplitude = float(ys[peak])
    if amplitude == 0.0:
        return {"a": 1.0, "mu": float(np.mean(xs)), "s": _span(xs) / 6.0}
    v = np.sign(amplitude) * ys
    level = 0.5 * abs(amplitude)
    left = right = peak
    while left > 0 and v[left - 1] >= level:
        left -= 1
    while right < xs.size - 1 and v[right + 1] >= level:
        right += 1
    x_left, x_right = float(xs[left]), float(xs[right])
    if left > 0:  # interpolate the crossing between the last point below and the first above
        x_left = float(np.interp(level, [v[left - 1], v[left]], [xs[left - 1], xs[left]]))
    if right < xs.size - 1:
        x_right = float(np.interp(level, [v[right + 1], v[right]], [xs[right + 1], xs[right]]))
    width = x_right - x_left
    if width <= 0.0:
        return {"a": amplitude, "mu": float(xs[peak]), "s": _span(xs) / 10.0}
    return {
        "a": amplitude,
        "mu": 0.5 * (x_left + x_right),
        "s": _positive(width / _FWHM_PER_SIGMA, _span(xs) / 10.0),
    }


# -- applicability ----------------------------------------------------------------------


def _count(n: int) -> str:
    return f"{n} value" if n == 1 else f"{n} values"


def _domain_check(
    *,
    x_positive_for: str | None = None,
    x_non_negative: bool = False,
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability check: finiteness always, plus an optional restriction on ``x``."""

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
                    f"found {n} non-positive {_count(n).split(' ', 1)[1]}"
                )
        if x_non_negative:
            n = int(np.count_nonzero(finite_x < 0.0))
            if n:
                reasons.append(
                    f"x must be non-negative (a concentration or amount): "
                    f"found {n} negative {_count(n).split(' ', 1)[1]}"
                )
        return reasons

    return check


# -- the starter set --------------------------------------------------------------------

_SEBER_WILD = "Seber G.A.F., Wild C.J. (1989) Nonlinear Regression, Wiley"
_DRAPER_SMITH = "Draper N.R., Smith H. (1998) Applied Regression Analysis, 3rd ed., Wiley"

STARTER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="exp_growth",
        name="Exponential growth",
        category="exponential",
        description="Exponential growth of y with x from a starting value.",
        example=ModelExample({"a": 2.0, "b": 0.5}, x_range=(0.0, 4.0)),
        expression="a*exp(b*x)",
        parameters=(
            Parameter("a", "Value of y at x = 0 (scale of the curve)"),
            Parameter("b", "Growth rate per unit x (negative gives decay towards zero)"),
        ),
        initial_guess=_guess_exp_growth,
        domain_check=_domain_check(),
        references=(
            "Malthus T.R. (1798) An Essay on the Principle of Population",
            _SEBER_WILD,
        ),
        assumptions="y changes by a constant proportion per unit x; the curve has no offset.",
    ),
    ModelSpec(
        id="exp_decay_offset",
        name="Exponential decay with offset",
        category="exponential",
        description="Exponential relaxation of y towards a constant asymptote.",
        example=ModelExample({"a": 3.0, "b": 0.8, "c": 1.0}, x_range=(0.0, 6.0)),
        expression="a*exp(-b*x)+c",
        parameters=(
            Parameter("a", "Amplitude: distance of y from the asymptote at x = 0"),
            Parameter("b", "Decay rate per unit x (positive)", lower=0.0),
            Parameter("c", "Asymptote that y approaches as x grows"),
        ),
        initial_guess=_guess_exp_decay_offset,
        domain_check=_domain_check(),
        references=(
            "Bates D.M., Watts D.G. (1988) Nonlinear Regression Analysis and Its Applications, "
            "Wiley",
            _SEBER_WILD,
        ),
        assumptions=(
            "y relaxes towards a constant asymptote at a single rate; the curve is monotonic."
        ),
    ),
    ModelSpec(
        id="exp_offset",
        name="Exponential with offset (asymptotic regression)",
        category="exponential",
        description="A constant plus one exponential term (an asymptote at one end of the data).",
        example=ModelExample({"a": 5.0, "b": -3.0, "c": -0.7}, x_range=(0.0, 6.0)),
        expression="a+b*exp(c*x)",
        parameters=(
            Parameter(
                "a", "Asymptote: the value y approaches (c < 0 as x grows, c > 0 as x falls)"
            ),
            Parameter("b", "Distance of y from the asymptote at x = 0"),
            Parameter("c", "Exponential rate per unit x (either sign)"),
        ),
        initial_guess=_guess_exp_offset,
        domain_check=_domain_check(),
        references=(
            "Ratkowsky D.A. (1983) Nonlinear Regression Modeling, Marcel Dekker",
            _SEBER_WILD,
        ),
        assumptions=(
            "y is a constant plus one exponential term; the sign of c selects which end of "
            "the data the asymptote lies at."
        ),
    ),
    ModelSpec(
        id="log_natural",
        name="Natural logarithm",
        category="logarithmic",
        description="Straight line in the natural logarithm of x.",
        example=ModelExample({"a": 1.0, "b": 2.0}, x_range=(0.5, 10.0)),
        expression="a+b*log(x)",
        parameters=(
            Parameter("a", "Value of y at x = 1"),
            Parameter("b", "Change in y per unit of ln(x)"),
        ),
        initial_guess=_guess_log(np.log),
        domain_check=_domain_check(x_positive_for="log(x)"),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is strictly positive and y is linear in ln(x) with constant-variance errors."
        ),
    ),
    ModelSpec(
        id="log_base10",
        name="Base-10 logarithm",
        category="logarithmic",
        description="Straight line in the base-10 logarithm of x.",
        example=ModelExample({"a": 1.0, "b": 2.0}, x_range=(0.5, 50.0)),
        expression="a+b*log10(x)",
        parameters=(
            Parameter("a", "Value of y at x = 1"),
            Parameter("b", "Change in y per decade of x (per unit of log10(x))"),
        ),
        initial_guess=_guess_log(np.log10),
        domain_check=_domain_check(x_positive_for="log10(x)"),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "x is strictly positive and y is linear in log10(x) with constant-variance errors."
        ),
    ),
    ModelSpec(
        id="power",
        name="Power law",
        category="power",
        description="Power law: y proportional to x raised to a constant exponent.",
        example=ModelExample({"a": 2.0, "b": 1.5}, x_range=(0.5, 5.0)),
        expression="a*x**b",
        parameters=(
            Parameter("a", "Value of y at x = 1 (scale of the curve)"),
            Parameter("b", "Exponent: the slope of the curve on log-log axes"),
        ),
        initial_guess=_guess_power,
        domain_check=_domain_check(x_positive_for="x**b"),
        references=(_DRAPER_SMITH, _SEBER_WILD),
        assumptions=(
            "x is strictly positive; errors are additive on y (fitted on the original scale, "
            "not by regressing log y on log x)."
        ),
    ),
    ModelSpec(
        id="michaelis_menten",
        name="Michaelis-Menten saturation",
        category="saturation",
        description="Hyperbolic saturation: y rises to a plateau, half of it reached at km.",
        example=ModelExample({"vmax": 10.0, "km": 2.0}, x_range=(0.1, 20.0)),
        expression="vmax*x/(km+x)",
        parameters=(
            Parameter("vmax", "Saturation level that y approaches at large x", lower=0.0),
            Parameter("km", "Value of x at which y reaches half of vmax", lower=0.0),
        ),
        initial_guess=_guess_michaelis_menten,
        domain_check=_domain_check(x_non_negative=True),
        references=(
            "Michaelis L., Menten M.L. (1913) Biochem. Z. 49:333-369",
            _SEBER_WILD,
        ),
        assumptions=(
            "x is non-negative and y rises hyperbolically to a plateau, with y = 0 at x = 0."
        ),
    ),
    ModelSpec(
        id="logistic3",
        name="Three-parameter logistic",
        category="sigmoidal",
        description="Symmetric S-curve rising from zero to an upper plateau.",
        example=ModelExample({"k": 10.0, "r": 1.2, "x0": 3.0}, x_range=(0.0, 8.0)),
        expression="k/(1+exp(-r*(x-x0)))",
        parameters=(
            Parameter("k", "Upper plateau (carrying capacity) that y approaches", lower=0.0),
            Parameter("r", "Steepness: growth rate per unit x at the midpoint", lower=0.0),
            Parameter("x0", "Midpoint: the x at which y equals k/2"),
        ),
        initial_guess=_guess_logistic3,
        domain_check=_domain_check(),
        references=(
            "Verhulst P.F. (1838) Notice sur la loi que la population poursuit dans son "
            "accroissement, Corr. Math. Phys. 10:113-121",
            _SEBER_WILD,
        ),
        assumptions=(
            "y increases from a lower plateau of zero to an upper plateau k along a symmetric "
            "S-curve."
        ),
    ),
    ModelSpec(
        id="gaussian",
        name="Gaussian peak",
        category="peak",
        description="Bell-shaped peak centred on mu with standard deviation s.",
        example=ModelExample({"a": 5.0, "mu": 2.0, "s": 1.2}, x_range=(-2.0, 6.0)),
        expression="a*exp(-(x-mu)**2/(2*s**2))",
        parameters=(
            Parameter("a", "Peak height above a zero baseline (negative for a dip)"),
            Parameter("mu", "Position of the peak centre"),
            Parameter("s", "Standard deviation: width of the peak (positive)", lower=0.0),
        ),
        initial_guess=_guess_gaussian,
        domain_check=_domain_check(),
        references=(
            "Gauss C.F. (1809) Theoria motus corporum coelestium in sectionibus conicis solem "
            "ambientium",
            _SEBER_WILD,
        ),
        assumptions="A single symmetric peak on a zero baseline; the data must span the peak.",
    ),
)


def register_starter_models() -> None:
    """Register every starter model that is not already registered (idempotent).

    A different model already holding one of these ids is a programming error and still
    raises ``ValueError`` from :func:`register` rather than being silently kept.
    """
    existing = {m.id: m for m in list_models()}
    for spec in STARTER_MODELS:
        if existing.get(spec.id) is not spec:
            register(spec)


register_starter_models()
