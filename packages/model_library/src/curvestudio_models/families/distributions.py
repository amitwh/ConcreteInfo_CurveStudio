"""Distribution shapes: the bell and tail curves a measured response often follows.

Each model here is the density (or density kernel) of a named distribution scaled by an
amplitude ``a``, and each is fitted on the original response scale. The amplitude enters
the model linearly, so the initial guess profiles it out by least squares over a coarse
grid of the shape parameters and refines them with a golden-section search; the start is
therefore inside the basin of the global optimum, which a single-start fit on noisy data
needs. Every model documents the ``x`` range it is defined on.

This module repeats the profile-guess helper of the ``peaks`` family on purpose: a family
module is loaded on its own and must not depend on a sibling, so that one family can be
skipped or moved without touching the others.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]

# References are cited only where the work treats the model; no page or chapter is given
# because none was verified.
_NIST = (
    "NIST/SEMATECH e-Handbook of Statistical Methods (2012), National Institute of Standards "
    "and Technology, https://www.itl.nist.gov/div898/handbook/"
)
_WEIBULL = (
    "Weibull W. (1951) A statistical distribution function of wide applicability, Journal of "
    "Applied Mechanics 18(3):293-297"
)
_MEEKER = "Meeker W.Q., Escobar L.A. (1998) Statistical Methods for Reliability Data, Wiley"
_GUMBEL = (
    "Gumbel E.J. (1954) Statistical Theory of Extreme Values and Some Practical Applications, "
    "U.S. Department of Commerce, National Bureau of Standards, Applied Mathematics Series 33"
)
_VERHULST = (
    "Verhulst P.F. (1838) Notice sur la loi que la population poursuit dans son accroissement, "
    "Correspondance Mathématique et Physique 10:113-121"
)
_TWEEDIE = (
    "Tweedie M.C.K. (1957) Statistical properties of inverse Gaussian distributions, Annals "
    "of Mathematical Statistics"
)
_JOHNSON_KOTZ = (
    "Johnson N.L., Kotz S., Balakrishnan N. (1994) Continuous Univariate Distributions, "
    "2nd ed., John Wiley and Sons"
)
_NUMERICAL_RECIPES = (
    "Press W.H., Teukolsky S.A., Vetterling W.T., Flannery B.P. (2007) Numerical Recipes: "
    "The Art of Scientific Computing, 3rd ed., Cambridge University Press "
    "(nonlinear least squares)"
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


def _span(xs: FloatArray) -> float:
    """Extent of ``x``; its magnitude (or 1) when ``x`` is constant."""
    if xs.size and float(np.ptp(xs)) > 0.0:
        return float(np.ptp(xs))
    if xs.size:
        top = float(np.max(np.abs(xs)))
        if top > 0.0:
            return top
    return 1.0


def _log_span(xs: FloatArray) -> float:
    """Extent of ``log(x)`` over the positive ``x``; 1 when it cannot be computed."""
    positive = xs[xs > 0.0]
    if positive.size >= 2:
        extent = float(np.log(positive[-1] / positive[0]))
        if math.isfinite(extent) and extent > 0.0:
            return extent
    return 1.0


def _linear_fit(columns: Sequence[FloatArray], target: FloatArray) -> FloatArray | None:
    """Least squares ``target ~ sum(c_i * column_i)``; ``None`` when it is not determined.

    The columns are scaled to unit length first, so a column of a very different magnitude
    cannot decide the rank on its own. A data set that overflows the solver is reported as
    undetermined, and the caller falls back to a finite default.
    """
    if target.size < len(columns):
        return None
    design = np.column_stack(columns)
    with np.errstate(all="ignore"):
        norms = np.linalg.norm(design, axis=0)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
            return None
        scaled = design / norms
        try:
            coef, _, rank, _ = np.linalg.lstsq(scaled, target, rcond=None)
        except np.linalg.LinAlgError:
            return None  # the normal equations do not converge for this data
    if rank < design.shape[1] or not np.all(np.isfinite(coef)):
        return None
    with np.errstate(all="ignore"):
        return np.asarray(coef / norms, dtype=np.float64)


# -- the profile guess ------------------------------------------------------------------


@dataclass(frozen=True)
class _Axis:
    """One shape parameter of a profile guess: its coarse grid and how it is refined.

    ``grid`` builds the candidate values from the data, so the guess follows the range and
    the resolution of the data it is given. A ``relative`` axis (a scale or a shape
    exponent) is refined on a multiplicative bracket, the others (a location) on an additive
    one.
    """

    name: str
    grid: Callable[[FloatArray], FloatArray]
    relative: bool = False
    lower: float = -math.inf
    upper: float = math.inf


_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0
"""The inverse of the golden ratio: the shrink factor of a golden-section search."""

_BRACKET = 2.5
"""A relative axis is refined on the bracket [value/2.5, value*2.5]."""

_PASSES = 2
"""How often the refinement walks over every axis."""

_ITERATIONS = 16
"""Golden-section iterations per axis; 16 shrink the bracket by a factor of 500."""


def _golden_minimum(
    objective: Callable[[float], float], low: float, high: float, iterations: int = _ITERATIONS
) -> float:
    """The point of ``[low, high]`` where the unimodal ``objective`` is smallest."""
    left, right = low, high
    c = right - _INV_PHI * (right - left)
    d = left + _INV_PHI * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - _INV_PHI * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + _INV_PHI * (right - left)
            fd = objective(d)
    return c if fc <= fd else d


def _axis_objective(
    objective: Callable[[dict[str, float]], float],
    current: dict[str, float],
    key: str,
) -> Callable[[float], float]:
    """The objective of one axis, with the other shape parameters held fixed."""

    def evaluate(candidate: float) -> float:
        return objective({**current, key: candidate})

    return evaluate


def _refine(
    axes: tuple[_Axis, ...],
    values: dict[str, float],
    objective: Callable[[dict[str, float]], float],
    span: float,
) -> dict[str, float]:
    """Refine the shape parameters one axis at a time by a golden-section search.

    Each axis is minimised over a bracket around its current value, then the next axis is
    tried; two passes over the axes let the parameters follow each other.
    """
    current = dict(values)
    for _ in range(_PASSES):
        for axis in axes:
            centre = current[axis.name]
            if axis.relative:
                low, high = centre / _BRACKET, centre * _BRACKET
            else:
                low, high = centre - span / 3.0, centre + span / 3.0
            low = max(low, axis.lower)
            high = min(high, axis.upper)
            if not low < high:
                continue
            current[axis.name] = _golden_minimum(
                _axis_objective(objective, current, axis.name), low, high
            )
    return current


def _profile_guess(
    *,
    parameters: tuple[Parameter, ...],
    linear: tuple[str, ...],
    axes: tuple[_Axis, ...],
    columns: Callable[[FloatArray, Mapping[str, float]], list[FloatArray]],
    start: Callable[[FloatArray, FloatArray], dict[str, float]] | None = None,
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """A guess that profiles the linear amplitude out over the shape parameters.

    ``columns(xs, values)`` returns the design column of the amplitude, in the order of
    ``linear``, for the given shape values. Its least-squares solution is the amplitude
    that fits best for those shape values, so the search runs over the shape parameters
    only: a coarse grid first, then a golden-section refinement of one axis at a time.
    ``start`` replaces the grid with values read from the data (used where the model
    linearises in log y).

    The result is always finite and strictly inside the parameter bounds.
    """
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _clean(x, y)
        span = _span(xs)

        def score(values: dict[str, float]) -> tuple[float, dict[str, float]]:
            """(SSE, parameter values) for the given shape values."""
            cols = columns(xs, values)
            coef = _linear_fit(cols, ys)
            if coef is None:
                return math.inf, {}
            out = {name: float(c) for name, c in zip(linear, coef, strict=True)}
            out.update(values)
            with np.errstate(all="ignore"):
                residual = ys - sum(c * col for c, col in zip(coef, cols, strict=True))
                sse = float(np.sum(residual**2))
            if not math.isfinite(sse):
                return math.inf, {}
            return sse, out

        def objective(values: dict[str, float]) -> float:
            """The SSE for the given shape values."""
            return score(values)[0]

        best_sse = math.inf
        current: dict[str, float] = {}
        if start is not None:
            seed = start(xs, ys)
            current = {
                axis.name: min(
                    max(float(seed.get(axis.name, float(np.median(axis.grid(xs))))), axis.lower),
                    axis.upper,
                )
                for axis in axes
            }
            best_sse, _ = score(current)
        else:
            for combo in product(*(axis.grid(xs) for axis in axes)):
                values = {axis.name: float(value) for axis, value in zip(axes, combo, strict=True)}
                sse, _ = score(values)
                if sse < best_sse:
                    best_sse, current = sse, values
        if not current:
            current = {axis.name: float(np.median(axis.grid(xs))) for axis in axes}

        refined = _refine(axes, current, objective, span)
        sse, _ = score(refined)
        if sse < best_sse:
            current = refined

        _, final = score(current)
        if not final:
            final = {**current, **dict.fromkeys(linear, 0.0)}
        return rules_guess(final, bounds)(xs, ys)

    return guess


def _positions(count: int = 21, positive: bool = False) -> Callable[[FloatArray], FloatArray]:
    """Candidate positions on the ``x`` scale: up to ``count`` of the x values of the data."""

    def grid(xs: FloatArray) -> FloatArray:
        values = xs[xs > 0.0] if positive else xs
        if values.size == 0:
            return np.array([1.0])
        if values.size <= count:
            return values
        index = np.unique(np.linspace(0, values.size - 1, count).round().astype(int))
        return np.asarray(values[index], dtype=np.float64)

    return grid


def _widths(count: int = 7) -> Callable[[FloatArray], FloatArray]:
    """Candidate widths on the ``x`` scale: from a twentieth of the span to five times it."""

    def grid(xs: FloatArray) -> FloatArray:
        span = _span(xs)
        return np.geomspace(span / 20.0, 5.0 * span, count)

    return grid


def _log_widths(count: int = 7) -> Callable[[FloatArray], FloatArray]:
    """Candidate widths on the ``log(x)`` scale: from a fiftieth of the span to the span."""

    def grid(xs: FloatArray) -> FloatArray:
        span = _log_span(xs)
        return np.geomspace(span / 50.0, span, count)

    return grid


def _rates(count: int = 9) -> Callable[[FloatArray], FloatArray]:
    """Candidate rates (the inverse of a length): around the inverse of the median ``x``."""

    def grid(xs: FloatArray) -> FloatArray:
        positive = xs[xs > 0.0]
        middle = float(np.median(positive)) if positive.size else 1.0
        if middle <= 0.0:
            middle = 1.0
        return np.geomspace(0.01 / middle, 1000.0 / middle, count)

    return grid


def _fixed(values: Sequence[float] | FloatArray) -> Callable[[FloatArray], FloatArray]:
    """Candidate values that do not depend on the data."""
    array = np.asarray(values, dtype=np.float64)
    return lambda xs: array


# -- shapes read from a linearisation in log y ------------------------------------------


def _log_linear_fit(xs: FloatArray, ys: FloatArray, basis: list[FloatArray]) -> FloatArray | None:
    """Least squares fit of ``log(y)`` against the given columns, on the points with y > 0."""
    if xs.size == 0:
        return None
    keep = ys > 0.0
    if int(np.count_nonzero(keep)) < len(basis):
        return None
    with np.errstate(all="ignore"):
        target = np.log(ys[keep])
        columns = [column[keep] for column in basis]
    return _linear_fit(columns, target)


def _beta_start(xs: FloatArray, ys: FloatArray) -> dict[str, float]:
    """Shape values of the beta kernel, read from the linearity of ``log(y)``.

    The kernel is ``x**(p-1)*(1-x)**(q-1)``, so ``log(y)`` is linear in ``log(x)``,
    ``log(1-x)`` and a constant; the two slopes give ``p-1`` and ``q-1`` directly.
    """
    with np.errstate(all="ignore"):
        basis = [np.log(xs), np.log1p(-xs), np.ones(xs.size)]
    coef = _log_linear_fit(xs, ys, basis)
    if coef is None:
        return {}
    return {"p": float(coef[0]) + 1.0, "q": float(coef[1]) + 1.0}


def _gamma_start(xs: FloatArray, ys: FloatArray) -> dict[str, float]:
    """Shape values of the gamma kernel, read from the linearity of ``log(y)``.

    The kernel is ``x**(k-1)*exp(-x/t)``, so ``log(y)`` is linear in ``log(x)``, ``x`` and
    a constant; the slope of the ``-x`` column is ``1/t``.
    """
    with np.errstate(all="ignore"):
        basis = [np.log(xs), -xs, np.ones(xs.size)]
    coef = _log_linear_fit(xs, ys, basis)
    if coef is None:
        return {}
    rate = float(coef[1])
    if not (math.isfinite(rate) and rate > 0.0):
        return {}
    return {"k": float(coef[0]) + 1.0, "t": 1.0 / rate}


# -- applicability ----------------------------------------------------------------------


def _domain_check(
    *, x_positive: bool = False, x_non_negative: bool = False, x_in_unit_interval: bool = False
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, plus the ``x`` range the distribution is defined on."""

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        if (n := int(np.count_nonzero(~np.isfinite(xa)))) > 0:
            reasons.append(f"x must be finite: found {n} value(s) that are NaN or infinite")
        if (n := int(np.count_nonzero(~np.isfinite(ya)))) > 0:
            reasons.append(f"y must be finite: found {n} value(s) that are NaN or infinite")
        finite_x = xa[np.isfinite(xa)]
        if x_positive and (n := int(np.count_nonzero(finite_x <= 0.0))) > 0:
            reasons.append(
                f"x must be strictly positive (the distribution has a logarithm or a "
                f"negative power of x): found {n} value(s) that are zero or negative"
            )
        if x_non_negative and (n := int(np.count_nonzero(finite_x < 0.0))) > 0:
            reasons.append(
                f"x must not be negative (the distribution is defined for x >= 0): found {n} "
                "value(s) that are negative"
            )
        if x_in_unit_interval:
            if (n := int(np.count_nonzero(finite_x <= 0.0))) > 0:
                reasons.append(
                    f"x must be strictly positive (the distribution has a negative power of "
                    f"x): found {n} value(s) that are zero or negative"
                )
            if (n := int(np.count_nonzero(finite_x >= 1.0))) > 0:
                reasons.append(
                    f"x must be below 1 (the distribution has a negative power of 1-x): found "
                    f"{n} value(s) that are 1 or more"
                )
        return reasons

    return check


# -- the models -------------------------------------------------------------------------

_P_LOGNORMAL = (
    Parameter(
        "a", "Scale of the density: the area under the curve divided by the width", lower=0.0
    ),
    Parameter("m", "Scale of x: the median of the distribution (positive)", lower=0.0),
    Parameter(
        "s", "Standard deviation of log(x): the relative width of the distribution", lower=0.0
    ),
)
_P_GAMMA = (
    Parameter("a", "Scale of the density kernel", lower=0.0),
    Parameter("k", "Shape of the distribution: the power of x in the kernel", lower=0.0),
    Parameter("t", "Scale of the distribution: the exponential decay length", lower=0.0),
)
_P_WEIBULL = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("k", "Shape of the distribution: the width of the tail", lower=0.0),
    Parameter("t", "Scale of the distribution: the value of x at which the tail drops", lower=0.0),
)
_P_CAUCHY = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Centre of the distribution"),
    Parameter("g", "Half width at half maximum of the density", lower=0.0),
)
_P_LEVY = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("c", "Scale of the distribution: the density peaks at x = c/3", lower=0.0),
)
_P_RAYLEIGH = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter(
        "s", "Scale of the distribution: the value of x at which the density peaks", lower=0.0
    ),
)
_P_MAXWELL = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("s", "Scale of the distribution: the width of the peak", lower=0.0),
)
_P_BETA = (
    Parameter("a", "Scale of the density kernel", lower=0.0),
    Parameter("p", "Shape of the kernel at the left end (the power of x)", lower=0.0),
    Parameter("q", "Shape of the kernel at the right end (the power of 1-x)", lower=0.0),
)
_P_LOGISTIC = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Centre of the distribution"),
    Parameter("s", "Scale of the distribution: the width of the peak", lower=0.0),
)
_P_GUMBEL = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Centre of the distribution"),
    Parameter("t", "Scale of the distribution: the width of the peak", lower=0.0),
)
_P_LAPLACE = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Centre of the distribution"),
    Parameter("t", "Scale of the distribution: the width of the peak", lower=0.0),
)
_P_INVERSE_GAUSSIAN = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Mean of the distribution (positive)", lower=0.0),
    Parameter("lam", "Precision of the distribution: the inverse of its spread", lower=0.0),
)
_P_NORMAL = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Centre of the distribution"),
    Parameter("s", "Standard deviation of the distribution", lower=0.0),
)
_P_LOG_LOGISTIC = (
    Parameter("a", "Scale of the density", lower=0.0),
    Parameter("m", "Scale of x: the median of the distribution (positive)", lower=0.0),
    Parameter("k", "Shape of the distribution: the power of x in the density", lower=0.0),
)


def _lognormal_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the log-normal density."""
    m, s = np.float64(values["m"]), np.float64(values["s"])
    with np.errstate(all="ignore"):
        shape = np.exp(-((np.log(xs / m)) ** 2) / (2.0 * s**2))
        return [shape / (xs * s * math.sqrt(2.0 * math.pi))]


def _gamma_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the gamma density kernel.

    ``x**(k-1)*exp(-x/t)`` is evaluated as ``exp((k-1)*log(x)-x/t)`` so that a huge ``x``,
    where ``x**(k-1)`` overflows to inf while ``exp(-x/t)`` underflows to 0, gives the true
    value 0 instead of the NaN of the product inf*0.
    """
    k, t = np.float64(values["k"]), np.float64(values["t"])
    with np.errstate(all="ignore"):
        return [np.exp((k - 1.0) * np.log(xs) - xs / t)]


def _weibull_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Weibull density."""
    k, t = np.float64(values["k"]), np.float64(values["t"])
    with np.errstate(all="ignore"):
        return [(k / t) * (xs / t) ** (k - 1.0) * np.exp(-((xs / t) ** k))]


def _cauchy_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Cauchy density."""
    m, g = np.float64(values["m"]), np.float64(values["g"])
    with np.errstate(all="ignore"):
        return [1.0 / (math.pi * g * (1.0 + ((xs - m) / g) ** 2))]


def _levy_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Levy density."""
    c = np.float64(values["c"])
    with np.errstate(all="ignore"):
        return [np.sqrt(c / (2.0 * math.pi)) * np.exp(-c / (2.0 * xs)) / xs**1.5]


def _rayleigh_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Rayleigh density."""
    s = np.float64(values["s"])
    with np.errstate(all="ignore"):
        return [xs / s**2 * np.exp(-(xs**2) / (2.0 * s**2))]


def _maxwell_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Maxwell-Boltzmann density.

    The factor ``x**2`` is evaluated as ``exp(2*log(x))`` so that a huge ``x``, where
    ``x**2`` overflows to inf while the Gaussian factor underflows to 0, gives the true
    value 0 instead of the NaN of the product inf*0.
    """
    s = np.float64(values["s"])
    with np.errstate(all="ignore"):
        return [math.sqrt(2.0 / math.pi) * np.exp(2.0 * np.log(xs) - xs**2 / (2.0 * s**2)) / s**3]


def _beta_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the beta density kernel."""
    p, q = np.float64(values["p"]), np.float64(values["q"])
    with np.errstate(all="ignore"):
        return [xs ** (p - 1.0) * (1.0 - xs) ** (q - 1.0)]


def _logistic_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the logistic density.

    The density is written in the symmetric form ``1/(s*(2+e**z+e**(-z)))`` with
    ``z = (x-m)/s`` (an exact rewrite of ``e**(-z)/(s*(1+e**(-z))**2)``): deep in either
    tail the denominator overflows to inf and the value is a clean 0, while the product
    form pairs an underflowed 0 with an inf denominator and yields NaN.
    """
    m, s = np.float64(values["m"]), np.float64(values["s"])
    with np.errstate(all="ignore"):
        z = (xs - m) / s
        return [1.0 / (s * (2.0 + np.exp(z) + np.exp(-z)))]


def _gumbel_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Gumbel (type I extreme value) density."""
    m, t = np.float64(values["m"]), np.float64(values["t"])
    with np.errstate(all="ignore"):
        z = (xs - m) / t
        return [np.exp(-(z + np.exp(-z))) / t]


def _laplace_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Laplace (double exponential) density."""
    m, t = np.float64(values["m"]), np.float64(values["t"])
    with np.errstate(all="ignore"):
        return [np.exp(-np.abs(xs - m) / t) / (2.0 * t)]


def _inverse_gaussian_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the inverse Gaussian (Wald) density."""
    m, lam = np.float64(values["m"]), np.float64(values["lam"])
    with np.errstate(all="ignore"):
        return [
            np.sqrt(lam / (2.0 * math.pi * xs**3))
            * np.exp(-lam * (xs - m) ** 2 / (2.0 * m**2 * xs))
        ]


def _normal_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the normal density."""
    m, s = np.float64(values["m"]), np.float64(values["s"])
    with np.errstate(all="ignore"):
        return [np.exp(-((xs - m) ** 2) / (2.0 * s**2)) / (s * math.sqrt(2.0 * math.pi))]


def _log_logistic_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the log-logistic (Fisk) density."""
    m, k = np.float64(values["m"]), np.float64(values["k"])
    with np.errstate(all="ignore"):
        ratio = xs / m
        return [(k / m) * ratio ** (k - 1.0) / (1.0 + ratio**k) ** 2]


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="distribution_lognormal",
        name="Log-normal distribution",
        category="distribution",
        description="The log-normal density: a normal distribution on the log of x.",
        example=ModelExample({"a": 2.0, "m": 3.0, "s": 0.6}, x_range=(0.4, 12.0)),
        expression="a*exp(-(log(x/m))**2/(2*s**2))/(x*s*sqrt(2*pi))",
        parameters=_P_LOGNORMAL,
        initial_guess=_profile_guess(
            parameters=_P_LOGNORMAL,
            linear=("a",),
            axes=(
                _Axis("m", _positions(13, positive=True), relative=True, lower=0.0),
                _Axis("s", _log_widths(7), relative=True, lower=0.0),
            ),
            columns=_lognormal_columns,
        ),
        domain_check=_domain_check(x_positive=True),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "x is strictly positive; the distribution is right-skewed and its median is m. "
            "Errors are additive on y."
        ),
    ),
    ModelSpec(
        id="distribution_gamma",
        name="Gamma distribution",
        category="distribution",
        description="The gamma density kernel x**(k-1)*exp(-x/t), scaled by an amplitude.",
        example=ModelExample({"a": 2.0, "k": 2.5, "t": 3.0}, x_range=(0.2, 14.0)),
        expression="a*exp((k-1)*log(x)-x/t)",
        parameters=_P_GAMMA,
        initial_guess=_profile_guess(
            parameters=_P_GAMMA,
            linear=("a",),
            axes=(
                _Axis(
                    "k",
                    _fixed([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]),
                    relative=True,
                    lower=0.0,
                ),
                _Axis("t", _widths(7), relative=True, lower=0.0),
            ),
            columns=_gamma_columns,
            start=_gamma_start,
        ),
        domain_check=_domain_check(x_positive=True),
        references=(_NIST, _MEEKER),
        assumptions=(
            "x is strictly positive and the shape k is larger than one for the curve to have "
            "a maximum inside the data; the normalising factor 1/(Gamma(k)*t**k) is absorbed "
            "into the amplitude, so the curve is a scaled gamma density, not a normalised one."
        ),
    ),
    ModelSpec(
        id="distribution_weibull",
        name="Weibull distribution",
        category="distribution",
        description="The Weibull density: a stretched exponential fall-off.",
        example=ModelExample({"a": 2.0, "k": 2.0, "t": 3.0}, x_range=(0.2, 14.0)),
        expression="a*(k/t)*(x/t)**(k-1)*exp(-(x/t)**k)",
        parameters=_P_WEIBULL,
        initial_guess=_profile_guess(
            parameters=_P_WEIBULL,
            linear=("a",),
            axes=(
                _Axis(
                    "k",
                    _fixed([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]),
                    relative=True,
                    lower=0.0,
                ),
                _Axis("t", _widths(7), relative=True, lower=0.0),
            ),
            columns=_weibull_columns,
        ),
        domain_check=_domain_check(x_positive=True),
        references=(_WEIBULL, _MEEKER, _NIST),
        assumptions=(
            "x is strictly positive; k below one gives a density that rises without bound at "
            "x = 0, k = 1 the exponential density and larger k a peak followed by a heavy tail."
        ),
    ),
    ModelSpec(
        id="distribution_cauchy",
        name="Cauchy distribution",
        category="distribution",
        description="The Cauchy density: a Lorentzian peak with heavy tails.",
        example=ModelExample({"a": 2.0, "m": 0.7, "g": 0.9}, x_range=(-4.0, 5.0)),
        expression="a/(pi*g*(1+((x-m)/g)**2))",
        parameters=_P_CAUCHY,
        initial_guess=_profile_guess(
            parameters=_P_CAUCHY,
            linear=("a",),
            axes=(
                _Axis("m", _positions()),
                _Axis("g", _widths(), relative=True, lower=0.0),
            ),
            columns=_cauchy_columns,
        ),
        domain_check=_domain_check(),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "The density is centred on m and its tails fall off like 1/(x-m)**2; it has no "
            "mean and no variance."
        ),
    ),
    ModelSpec(
        id="distribution_levy",
        name="Levy distribution",
        category="distribution",
        description="The Levy density: a sharp peak at x = c/3 and a long tail to the right.",
        example=ModelExample({"a": 2.0, "c": 2.0}, x_range=(0.1, 10.0)),
        expression="a*sqrt(c/(2*pi))*exp(-c/(2*x))/x**1.5",
        parameters=_P_LEVY,
        initial_guess=_profile_guess(
            parameters=_P_LEVY,
            linear=("a",),
            axes=(_Axis("c", _widths(9), relative=True, lower=0.0),),
            columns=_levy_columns,
        ),
        domain_check=_domain_check(x_positive=True),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "x is strictly positive and the location of the distribution is zero, as in the "
            "standard Levy density; the density peaks at x = c/3 and falls off like "
            "x**(-3/2)."
        ),
    ),
    ModelSpec(
        id="distribution_rayleigh",
        name="Rayleigh distribution",
        category="distribution",
        description="The Rayleigh density: a peak at x = s and a decaying tail.",
        example=ModelExample({"a": 2.0, "s": 3.0}, x_range=(0.1, 14.0)),
        expression="a*x/s**2*exp(-x**2/(2*s**2))",
        parameters=_P_RAYLEIGH,
        initial_guess=_profile_guess(
            parameters=_P_RAYLEIGH,
            linear=("a",),
            axes=(_Axis("s", _widths(9), relative=True, lower=0.0),),
            columns=_rayleigh_columns,
        ),
        domain_check=_domain_check(x_non_negative=True),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "x is not negative; the density is zero at x = 0, peaks at x = s and falls off "
            "like exp(-x**2/(2*s**2))."
        ),
    ),
    ModelSpec(
        id="distribution_maxwell_boltzmann",
        name="Maxwell-Boltzmann distribution",
        category="distribution",
        description="The Maxwell-Boltzmann density: the speed distribution of a gas.",
        example=ModelExample({"a": 2.0, "s": 3.0}, x_range=(0.1, 14.0)),
        expression="a*sqrt(2/pi)*x**2/s**3*exp(-x**2/(2*s**2))",
        parameters=_P_MAXWELL,
        initial_guess=_profile_guess(
            parameters=_P_MAXWELL,
            linear=("a",),
            axes=(_Axis("s", _widths(9), relative=True, lower=0.0),),
            columns=_maxwell_columns,
        ),
        domain_check=_domain_check(x_non_negative=True),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "x is not negative; the density is zero at x = 0, peaks at x = s*sqrt(2) and "
            "falls off like exp(-x**2/(2*s**2))."
        ),
    ),
    ModelSpec(
        id="distribution_beta",
        name="Beta-shaped distribution",
        category="distribution",
        description="The beta density kernel x**(p-1)*(1-x)**(q-1), scaled by an amplitude.",
        example=ModelExample({"a": 2.0, "p": 2.5, "q": 1.8}, x_range=(0.05, 0.95)),
        expression="a*x**(p-1)*(1-x)**(q-1)",
        parameters=_P_BETA,
        initial_guess=_profile_guess(
            parameters=_P_BETA,
            linear=("a",),
            axes=(
                _Axis(
                    "p",
                    _fixed([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
                    relative=True,
                    lower=0.0,
                ),
                _Axis(
                    "q",
                    _fixed([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
                    relative=True,
                    lower=0.0,
                ),
            ),
            columns=_beta_columns,
            start=_beta_start,
        ),
        domain_check=_domain_check(x_in_unit_interval=True),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "x lies strictly between 0 and 1; the normalising factor 1/B(p,q) is absorbed "
            "into the amplitude, so the curve is a scaled beta density, not a normalised one."
        ),
    ),
    ModelSpec(
        id="distribution_logistic",
        name="Logistic distribution",
        category="distribution",
        description="The logistic density: a symmetric peak with exponential tails.",
        example=ModelExample({"a": 2.0, "m": 0.5, "s": 0.8}, x_range=(-4.0, 5.0)),
        expression="a/(s*(2+exp((x-m)/s)+exp(-(x-m)/s)))",
        parameters=_P_LOGISTIC,
        initial_guess=_profile_guess(
            parameters=_P_LOGISTIC,
            linear=("a",),
            axes=(
                _Axis("m", _positions()),
                _Axis("s", _widths(), relative=True, lower=0.0),
            ),
            columns=_logistic_columns,
        ),
        domain_check=_domain_check(),
        references=(_NIST, _VERHULST),
        assumptions=(
            "The density is centred on m, peaks there and falls off like exp(-|x-m|/s); it "
            "has a mean of m and a standard deviation of s*sqrt(3)/pi."
        ),
    ),
    ModelSpec(
        id="distribution_gumbel",
        name="Gumbel distribution",
        category="distribution",
        description="The Gumbel (type I extreme value) density: the distribution of a maximum.",
        example=ModelExample({"a": 2.0, "m": 0.5, "t": 1.2}, x_range=(-4.0, 5.0)),
        expression="a*exp(-((x-m)/t+exp(-(x-m)/t)))/t",
        parameters=_P_GUMBEL,
        initial_guess=_profile_guess(
            parameters=_P_GUMBEL,
            linear=("a",),
            axes=(
                _Axis("m", _positions()),
                _Axis("t", _widths(), relative=True, lower=0.0),
            ),
            columns=_gumbel_columns,
        ),
        domain_check=_domain_check(),
        references=(_GUMBEL, _NIST),
        assumptions=(
            "The density is the one of the largest of many samples: it is right-skewed, with "
            "a long tail to the right and a short one to the left."
        ),
    ),
    ModelSpec(
        id="distribution_laplace",
        name="Laplace distribution",
        category="distribution",
        description="The Laplace (double exponential) density: a sharp peak and straight tails.",
        example=ModelExample({"a": 2.0, "m": 0.5, "t": 1.2}, x_range=(-4.0, 5.0)),
        expression="a*exp(-abs(x-m)/t)/(2*t)",
        parameters=_P_LAPLACE,
        initial_guess=_profile_guess(
            parameters=_P_LAPLACE,
            linear=("a",),
            axes=(
                _Axis("m", _positions()),
                _Axis("t", _widths(), relative=True, lower=0.0),
            ),
            columns=_laplace_columns,
        ),
        domain_check=_domain_check(),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=("The density is centred on m and falls off like exp(-|x-m|/t) on both sides."),
    ),
    ModelSpec(
        id="distribution_inverse_gaussian",
        name="Inverse Gaussian distribution",
        category="distribution",
        description=(
            "The inverse Gaussian (Wald) density: the first passage time of a random walk."
        ),
        example=ModelExample({"a": 2.0, "m": 2.0, "lam": 20.0}, x_range=(0.1, 10.0)),
        expression="a*sqrt(lam/(2*pi*x**3))*exp(-lam*(x-m)**2/(2*m**2*x))",
        parameters=_P_INVERSE_GAUSSIAN,
        initial_guess=_profile_guess(
            parameters=_P_INVERSE_GAUSSIAN,
            linear=("a",),
            axes=(
                _Axis("m", _positions(13, positive=True), lower=0.0),
                _Axis("lam", _rates(9), relative=True, lower=0.0),
            ),
            columns=_inverse_gaussian_columns,
        ),
        domain_check=_domain_check(x_positive=True),
        references=(_TWEEDIE, _NIST),
        assumptions=(
            "x is strictly positive; the distribution has mean m and variance m**3/lam, so "
            "the peak sits below the mean and the tail to the right is long."
        ),
    ),
    ModelSpec(
        id="distribution_normal",
        name="Normal distribution",
        category="distribution",
        description="The normal (Gaussian) density.",
        example=ModelExample({"a": 2.0, "m": 0.7, "s": 1.3}, x_range=(-4.0, 6.0)),
        expression="a/(s*sqrt(2*pi))*exp(-(x-m)**2/(2*s**2))",
        parameters=_P_NORMAL,
        initial_guess=_profile_guess(
            parameters=_P_NORMAL,
            linear=("a",),
            axes=(
                _Axis("m", _positions()),
                _Axis("s", _widths(), relative=True, lower=0.0),
            ),
            columns=_normal_columns,
        ),
        domain_check=_domain_check(),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "The density is centred on m with standard deviation s; it is the same curve as "
            "the Gaussian peak of the core starter set, with a different amplitude meaning."
        ),
    ),
    ModelSpec(
        id="distribution_log_logistic",
        name="Log-logistic distribution",
        category="distribution",
        description="The log-logistic (Fisk) density: the logistic distribution on a log scale.",
        example=ModelExample({"a": 2.0, "m": 3.0, "k": 2.0}, x_range=(0.2, 14.0)),
        expression="a*(k/m)*(x/m)**(k-1)/(1+(x/m)**k)**2",
        parameters=_P_LOG_LOGISTIC,
        initial_guess=_profile_guess(
            parameters=_P_LOG_LOGISTIC,
            linear=("a",),
            axes=(
                _Axis("m", _positions(13, positive=True), relative=True, lower=0.0),
                _Axis(
                    "k",
                    _fixed([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
                    relative=True,
                    lower=0.0,
                ),
            ),
            columns=_log_logistic_columns,
        ),
        domain_check=_domain_check(x_positive=True),
        references=(_NIST, _JOHNSON_KOTZ),
        assumptions=(
            "x is strictly positive; the distribution is right-skewed for k above one and "
            "its tail falls off like x**(-k-1)."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # The normal density and the Gaussian peak of the core starter set are the same curve;
    # only the meaning of the amplitude differs (the density carries 1/(s*sqrt(2*pi))).
    ("distribution_normal", "gaussian"),
    # A Lorentzian peak of height a and a Cauchy density of the same scale describe the
    # same curve; they differ only in what the amplitude means (the density carries the
    # factor 1/(pi*g)).
    ("peak_lorentzian", "distribution_cauchy"),
    # The centred Cauchy density and the quadratic inverse polynomial are the same curve:
    # 1/(a0+a1*x+a2*x**2) is a Lorentzian of centre -a1/(2*a2) and half width
    # sqrt(a0/a2-(a1/(2*a2))**2).
    ("distribution_cauchy", "invpoly_2"),
    # The gamma density kernel x**(k-1)*exp(-x/t) and the exponential-power growth curve
    # a*exp(b*x)*x**c are the same family (k-1 = c, b = -1/t); only the amplitude and the
    # domain differ (the density lives on positive x).
    ("distribution_gamma", "exponential_power_growth"),
    # The Weibull density of shape k = 2 (the example) is exp(a+b*ln(x)+c*x**2), the
    # generated three-term chain tc3_exp_ln_sqr: one over t**2, b = 1, c = -1/t**2.
    ("distribution_weibull", "tc3_exp_ln_sqr"),
)
