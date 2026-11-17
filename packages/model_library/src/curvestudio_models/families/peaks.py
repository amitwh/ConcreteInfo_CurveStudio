"""Peak profiles: the bell-shaped curves used to describe a maximum in a measurement.

Every model here is fitted on the original response scale and carries an amplitude, so a
peak may stand above a baseline (``peak_gaussian_offset``) or sit on the zero line. Widths
are positive, the peak position is free, and a sum of peaks orders its components by
position (``d >= 0``), so the fit cannot swap two peaks of the same shape.

The initial guesses are not rules of thumb: in every model the amplitude (and the
baseline, where there is one) enters *linearly*, so the guess profiles it out by least
squares over a coarse grid of the shape parameters and then refines them with a pattern
search. The result is a start point inside the basin of the global optimum, which is what
a single-start fit on noisy data needs. Sums of peaks start from the peaks found in the
data and are refined the same way.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import scipy.special as sps
from numpy.typing import NDArray

from curvestudio.expressions.special import voigt_value
from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]

# References are cited only where the work treats the model; no page or chapter is given
# because none was verified.
_GAUSS = (
    "Gauss C.F. (1809) Theoria motus corporum coelestium in sectionibus conicis solem ambientium"
)
_SNEDDON = "Sneddon I.N. (1951) Fourier Transforms, McGraw-Hill, New York"
_VOIGT = (
    "Voigt W. (1912) Über das Gesetz der Intensitätsverteilung innerhalb der Linien eines "
    "Gasspektrums, Sitzungsberichte der Königlichen Bayerischen Akademie der Wissenschaften"
)
_OLIVERO = (
    "Olivero J.J., Longbothum R.L. (1977) Empirical fits to the Voigt line width: A brief "
    "review, Journal of Quantitative Spectroscopy and Radiative Transfer 17:233-236"
)
_IDA = (
    "Ida T., Ando M., Toraya H. (2000) Extended pseudo-Voigt function for approximating the "
    "Voigt profile, Journal of Applied Crystallography 33:1311-1316"
)
_THOMPSON = (
    "Thompson P., Cox D.E., Hastings J.B. (1987) Rietveld refinement of Debye-Scherrer "
    "synchrotron X-ray data from Al2O3, Journal of Applied Crystallography 20:79-83"
)
_KALAMBET = (
    "Kalambet Y., Kozmin Y., Mikhailova K., Naulaeva I., Kalambet Y. (2011) Reconstruction "
    "of chromatographic peaks using the exponentially modified Gaussian function, Journal "
    "of Chemometrics"
)
_AZZALINI = (
    "Azzalini S. (1985) A class of distributions which includes the normal ones, "
    "Scandinavian Journal of Statistics 12:171-178"
)
_PEARSON = (
    "Pearson K. (1895) Contributions to the Mathematical Theory of Evolution II. Skew "
    "Variation, Philosophical Transactions of the Royal Society of London A 186:343-414"
)
_BARD = "Bard Y. (1974) Nonlinear Parameter Estimation, Academic Press, New York"
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
    the resolution of the data it is given. A ``relative`` axis (a width or a shape
    exponent) is refined on a multiplicative bracket, the others (a peak position) on an
    additive one.
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
    """A guess that profiles the linear parameters out over the shape parameters.

    ``columns(xs, values)`` returns the design columns of the linear parameters, in the
    order of ``linear``, for the given shape values. Their least-squares solution is the
    amplitude (and baseline) that fits best for those shape values, so the search runs
    over the shape parameters only: a coarse grid first, then a golden-section refinement
    of one axis at a time. ``start`` replaces the grid with values read from the data
    (used by the sums of peaks).

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


def _positions(count: int = 21) -> Callable[[FloatArray], FloatArray]:
    """Candidate positions of a peak: up to ``count`` of the x values of the data."""

    def grid(xs: FloatArray) -> FloatArray:
        if xs.size == 0:
            return np.array([0.0])
        if xs.size <= count:
            return xs
        index = np.unique(np.linspace(0, xs.size - 1, count).round().astype(int))
        return np.asarray(xs[index], dtype=np.float64)

    return grid


def _widths(count: int = 7) -> Callable[[FloatArray], FloatArray]:
    """Candidate widths: from a twentieth of the ``x`` span to five times the span."""

    def grid(xs: FloatArray) -> FloatArray:
        span = _span(xs)
        return np.geomspace(span / 20.0, 5.0 * span, count)

    return grid


def _fixed(values: Sequence[float] | FloatArray) -> Callable[[FloatArray], FloatArray]:
    """Candidate values that do not depend on the data."""
    array = np.asarray(values, dtype=np.float64)
    return lambda xs: array


# -- peaks read from the data ------------------------------------------------------------


def _parabola_peak(xs: FloatArray, ys: FloatArray, index: int) -> tuple[float, float, float]:
    """(amplitude, centre, width) of the peak at ``index``, from a parabola through log(y).

    A parabola through ``log(y)`` around a maximum is a parabola through the Gaussian peak
    itself, so the coefficients give the centre and the standard deviation directly.
    """
    height = float(ys[index])
    span = _span(xs)
    if height <= 0.0:
        return 0.0, float(xs[index]), 0.1 * span
    threshold = 0.3 * height
    left = index
    while left > 0 and float(ys[left]) > threshold:
        left -= 1
    right = index
    while right < ys.size - 1 and float(ys[right]) > threshold:
        right += 1
    low, high = left + 1, right
    if high - low < 3:
        low, high = max(0, index - 1), min(ys.size, index + 2)
    window_x = xs[low:high] - float(xs[index])
    window_y = ys[low:high]
    with np.errstate(all="ignore"):
        coef = _linear_fit([window_x**2, window_x, np.ones(window_x.size)], np.log(window_y))
    if coef is None or coef[0] >= 0.0:
        return height, float(xs[index]), 0.1 * span
    quadratic, linear, constant = (float(c) for c in coef)
    centre = float(xs[index]) - linear / (2.0 * quadratic)
    width = math.sqrt(-1.0 / (2.0 * quadratic)) if quadratic < 0.0 else math.nan
    with np.errstate(all="ignore"):
        amplitude = float(np.exp(constant - linear * linear / (4.0 * quadratic)))
    if not (math.isfinite(amplitude) and amplitude > 0.0):
        amplitude = height
    if not (math.isfinite(width) and width > 0.0):
        width = 0.1 * span
    if not math.isfinite(centre):
        centre = float(xs[index])
    return amplitude, centre, width


def _peaks_in(xs: FloatArray, ys: FloatArray, count: int) -> list[tuple[float, float, float]]:
    """Up to ``count`` peaks as (amplitude, centre, width), the tallest first."""
    if xs.size == 0:
        return []
    interior = np.arange(1, max(ys.size - 1, 1))
    local = interior[(ys[interior] >= ys[interior - 1]) & (ys[interior] >= ys[interior + 1])]
    order = local[np.argsort(ys[local])[::-1]] if local.size else np.array([], dtype=int)
    chosen: list[tuple[float, float, float]] = []
    for index in order:
        peak = _parabola_peak(xs, ys, int(index))
        if peak[0] <= 0.0:
            continue
        if any(abs(peak[1] - other[1]) < 0.5 * max(peak[2], other[2]) for other in chosen):
            continue
        chosen.append(peak)
        if len(chosen) == count:
            break
    while len(chosen) < count and xs.size:
        # Not enough local maxima: fall back to the tallest point not yet used.
        remaining = [
            i
            for i in range(xs.size)
            if all(abs(float(xs[i]) - other[1]) > 0.5 * other[2] for other in chosen)
        ]
        if not remaining:
            break
        tallest = max(remaining, key=lambda i: float(ys[i]))
        chosen.append(_parabola_peak(xs, ys, tallest))
    return chosen


def _two_peak_start(xs: FloatArray, ys: FloatArray) -> dict[str, float]:
    """Starting shape values for the sum of two Gaussians."""
    peaks = _peaks_in(xs, ys, 2)
    span = _span(xs)
    if len(peaks) < 2:
        return {"mu1": 0.0, "s1": 0.1 * span, "d": 0.5 * span, "s2": 0.1 * span}
    first, second = sorted(peaks, key=lambda peak: peak[1])
    mu1, s1 = first[1], first[2]
    mu2, s2 = second[1], second[2]
    return {"mu1": mu1, "s1": s1, "d": max(mu2 - mu1, 1e-3 * span), "s2": s2}


def _three_peak_start(xs: FloatArray, ys: FloatArray) -> dict[str, float]:
    """Starting shape values for the sum of three Gaussians."""
    peaks = _peaks_in(xs, ys, 3)
    span = _span(xs)
    if len(peaks) < 3:
        return {
            "mu1": 0.0,
            "s1": 0.1 * span,
            "d2": 0.3 * span,
            "s2": 0.1 * span,
            "d3": 0.3 * span,
            "s3": 0.1 * span,
        }
    first, second, third = sorted(peaks, key=lambda peak: peak[1])
    return {
        "mu1": first[1],
        "s1": first[2],
        "d2": max(second[1] - first[1], 1e-3 * span),
        "s2": second[2],
        "d3": max(third[1] - second[1], 1e-3 * span),
        "s3": third[2],
    }


# -- applicability ----------------------------------------------------------------------


def _domain_check() -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: the data must be finite; a peak is defined on every real x."""

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        if (n := int(np.count_nonzero(~np.isfinite(xa)))) > 0:
            reasons.append(f"x must be finite: found {n} value(s) that are NaN or infinite")
        if (n := int(np.count_nonzero(~np.isfinite(ya)))) > 0:
            reasons.append(f"y must be finite: found {n} value(s) that are NaN or infinite")
        return reasons

    return check


# -- the models -------------------------------------------------------------------------

_P_GAUSSIAN_OFFSET = (
    Parameter("c", "Baseline: the level y approaches away from the peak"),
    Parameter("a", "Height of the peak above the baseline (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter("s", "Standard deviation of the Gaussian: the width of the peak", lower=0.0),
)
_P_LORENTZIAN_CENTERED = (
    Parameter("a", "Height of the peak at x = 0 (negative for a dip)"),
    Parameter("g", "Half width at half maximum of the peak", lower=0.0),
)
_P_LORENTZIAN = (
    Parameter("a", "Height of the peak (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter("g", "Half width at half maximum of the peak", lower=0.0),
)
_P_LORENTZIAN_OFFSET = (
    Parameter("c", "Baseline: the level y approaches away from the peak"),
    Parameter("a", "Height of the peak above the baseline (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter("g", "Half width at half maximum of the peak", lower=0.0),
)
_P_PEARSON_VII = (
    Parameter("a", "Height of the peak (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter(
        "g",
        "Width scale of the peak: the half width at half maximum is g*sqrt(2**(1/nu)-1), "
        "which equals g for the Lorentzian nu = 1",
        lower=0.0,
    ),
    Parameter("nu", "Exponent of the tail: the peak falls off like a power law", lower=0.0),
)
_P_PSEUDO_VOIGT = (
    Parameter("a", "Height of the peak (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter("w", "Full width at half maximum shared by the two components", lower=0.0),
    Parameter("eta", "Lorentzian fraction of the peak, between 0 and 1", lower=0.0, upper=1.0),
)
_P_VOIGT = (
    Parameter("a", "Area of the peak: the integral of the curve over all x"),
    Parameter("mu", "Position of the peak centre"),
    Parameter("sigma", "Standard deviation of the Gaussian component", lower=0.0),
    Parameter("gamma", "Half width at half maximum of the Lorentzian component", lower=0.0),
)
_TINY = 1e-12
"""Lower bound of a parameter the expression divides by (as in the rational family)."""

_P_EMG = (
    Parameter("a", "Area of the peak: the integral of the curve over all x"),
    Parameter("mu", "Centre of the Gaussian component"),
    Parameter("sigma", "Standard deviation of the Gaussian component", lower=0.0),
    Parameter("tau", "Mean of the exponential component: the length of the tail", lower=_TINY),
)
_P_SKEW_NORMAL = (
    Parameter("a", "Scale of the peak (the density of the skew-normal for one unit of area)"),
    Parameter("mu", "Position of the Gaussian component"),
    Parameter("s", "Standard deviation of the Gaussian component", lower=0.0),
    Parameter("alpha", "Skewness: positive tilts the peak to the right, negative to the left"),
)
_P_SPLIT_GAUSSIAN = (
    Parameter("a", "Height of the peak (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter("s1", "Standard deviation on the left of the centre", lower=0.0),
    Parameter("s2", "Standard deviation on the right of the centre", lower=0.0),
)
_P_POWER_PEAK = (
    Parameter("a", "Height of the peak (negative for a dip)"),
    Parameter("mu", "Position of the peak centre"),
    Parameter(
        "w",
        "Width scale of the peak: at the distance w from the centre the height has fallen "
        "to a*2**(-b) (halved when b = 1)",
        lower=0.0,
    ),
    Parameter("b", "Exponent of the tail: the peak falls off like a power law", lower=0.0),
)
_P_TWO_GAUSSIAN = (
    Parameter("a1", "Height of the first peak"),
    Parameter("mu1", "Position of the first peak"),
    Parameter("s1", "Standard deviation of the first peak", lower=0.0),
    Parameter("a2", "Height of the second peak"),
    Parameter(
        "d", "Distance between the two centres (the second peak is the later one)", lower=0.0
    ),
    Parameter("s2", "Standard deviation of the second peak", lower=0.0),
)
_P_THREE_GAUSSIAN = (
    Parameter("a1", "Height of the first peak"),
    Parameter("mu1", "Position of the first peak"),
    Parameter("s1", "Standard deviation of the first peak", lower=0.0),
    Parameter("a2", "Height of the second peak"),
    Parameter("d2", "Distance from the first to the second centre", lower=0.0),
    Parameter("s2", "Standard deviation of the second peak", lower=0.0),
    Parameter("a3", "Height of the third peak"),
    Parameter("d3", "Distance from the second to the third centre", lower=0.0),
    Parameter("s3", "Standard deviation of the third peak", lower=0.0),
)


def _gaussian_offset_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Columns of the baseline (1) and of the Gaussian peak."""
    mu, s = np.float64(values["mu"]), np.float64(values["s"])
    with np.errstate(all="ignore"):
        shape = np.exp(-((xs - mu) ** 2) / (2.0 * s**2))
    return [np.ones(xs.size), shape]


def _lorentzian_centered_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Lorentzian peak centred on x = 0."""
    g = np.float64(values["g"])
    with np.errstate(all="ignore"):
        return [g**2 / (xs**2 + g**2)]


def _lorentzian_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Lorentzian peak of any position."""
    mu, g = np.float64(values["mu"]), np.float64(values["g"])
    with np.errstate(all="ignore"):
        return [g**2 / ((xs - mu) ** 2 + g**2)]


def _lorentzian_offset_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Columns of the baseline (1) and of the Lorentzian peak."""
    mu, g = np.float64(values["mu"]), np.float64(values["g"])
    with np.errstate(all="ignore"):
        return [np.ones(xs.size), g**2 / ((xs - mu) ** 2 + g**2)]


def _pearson_vii_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Pearson VII peak."""
    mu, g, nu = (np.float64(values[name]) for name in ("mu", "g", "nu"))
    with np.errstate(all="ignore"):
        return [1.0 / (1.0 + ((xs - mu) / g) ** 2) ** nu]


def _pseudo_voigt_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the pseudo-Voigt peak: a Gaussian and a Lorentzian of the same width."""
    mu, w, eta = (np.float64(values[name]) for name in ("mu", "w", "eta"))
    with np.errstate(all="ignore"):
        gauss = np.exp(-4.0 * math.log(2.0) * (xs - mu) ** 2 / w**2)
        lorentz = (w / 2.0) ** 2 / ((xs - mu) ** 2 + (w / 2.0) ** 2)
        return [(1.0 - eta) * gauss + eta * lorentz]


def _voigt_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the Voigt profile (area-normalised, so the amplitude is the area)."""
    mu, sigma, gamma = (np.float64(values[name]) for name in ("mu", "sigma", "gamma"))
    with np.errstate(all="ignore"):
        return [np.asarray(voigt_value(xs - mu, sigma, gamma), dtype=np.float64)]


def _emg_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the exponentially modified Gaussian peak.

    The closed form of the convolution of a Gaussian with an exponential of mean ``tau`` is
    ``exp(-(x-mu)/tau + sigma**2/(2*tau**2)) * erfc((mu + sigma**2/tau - x)/(sqrt(2)*sigma))
    / (2*tau)``. It is evaluated with the erfc factor in the exponent
    (``exp(E + log(erfc(u)))``, an exact rewrite of ``exp(E)*erfc(u)``): far from the peak
    one factor overflows to inf and the other underflows to 0, whose product is NaN, while
    ``log(erfc(u))`` degenerates to -inf and the whole form to a clean 0.
    """
    mu, sigma, tau = (np.float64(values[name]) for name in ("mu", "sigma", "tau"))
    with np.errstate(all="ignore"):
        exponent = -(xs - mu) / tau + sigma**2 / (2.0 * tau**2)
        tail = np.log(
            sps.erfc((mu + sigma**2 / tau - xs) / (math.sqrt(2.0) * sigma)).astype(np.float64)
        )
        return [np.exp(exponent + tail) / (2.0 * tau)]


def _skew_normal_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the skew-normal peak: a Gaussian tilted by the normal cumulative function."""
    mu, s, alpha = (np.float64(values[name]) for name in ("mu", "s", "alpha"))
    with np.errstate(all="ignore"):
        shape = np.exp(-0.5 * ((xs - mu) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))
        tilt = 1.0 + np.asarray(sps.erf(alpha * (xs - mu) / (math.sqrt(2.0) * s)), dtype=np.float64)
        return [shape * tilt]


def _split_gaussian_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the split Gaussian peak: a different width on each side of the centre."""
    mu, s1, s2 = (np.float64(values[name]) for name in ("mu", "s1", "s2"))
    with np.errstate(all="ignore"):
        left = np.minimum(xs - mu, 0.0)
        right = np.maximum(xs - mu, 0.0)
        return [np.exp(-(left**2) / (2.0 * s1**2) - (right**2) / (2.0 * s2**2))]


def _power_peak_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Column of the power-law peak."""
    mu, w, b = (np.float64(values[name]) for name in ("mu", "w", "b"))
    with np.errstate(all="ignore"):
        return [(1.0 + np.abs((xs - mu) / w)) ** (-b)]


def _two_gaussian_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Columns of the two Gaussian peaks; the second one stands at ``mu1 + d``."""
    mu1, s1, d, s2 = (np.float64(values[name]) for name in ("mu1", "s1", "d", "s2"))
    with np.errstate(all="ignore"):
        first = np.exp(-((xs - mu1) ** 2) / (2.0 * s1**2))
        second = np.exp(-((xs - (mu1 + d)) ** 2) / (2.0 * s2**2))
    return [first, second]


def _three_gaussian_columns(xs: FloatArray, values: Mapping[str, float]) -> list[FloatArray]:
    """Columns of the three Gaussian peaks, at ``mu1``, ``mu1 + d2`` and ``mu1 + d2 + d3``."""
    mu1, s1, d2, s2, d3, s3 = (
        np.float64(values[name]) for name in ("mu1", "s1", "d2", "s2", "d3", "s3")
    )
    with np.errstate(all="ignore"):
        first = np.exp(-((xs - mu1) ** 2) / (2.0 * s1**2))
        second = np.exp(-((xs - (mu1 + d2)) ** 2) / (2.0 * s2**2))
        third = np.exp(-((xs - (mu1 + d2 + d3)) ** 2) / (2.0 * s3**2))
    return [first, second, third]


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="peak_gaussian_offset",
        name="Gaussian peak on a baseline",
        category="peak",
        description="A Gaussian peak standing on a constant baseline.",
        example=ModelExample({"c": 0.5, "a": 4.0, "mu": 1.0, "s": 0.8}, x_range=(-3.0, 5.0)),
        expression="c+a*exp(-(x-mu)**2/(2*s**2))",
        parameters=_P_GAUSSIAN_OFFSET,
        initial_guess=_profile_guess(
            parameters=_P_GAUSSIAN_OFFSET,
            linear=("c", "a"),
            axes=(
                _Axis("mu", _positions()),
                _Axis("s", _widths(), relative=True, lower=0.0),
            ),
            columns=_gaussian_offset_columns,
        ),
        domain_check=_domain_check(),
        references=(_GAUSS, _BARD),
        assumptions=(
            "The peak is symmetric about mu and the baseline is flat; errors are additive on "
            "y, so the baseline is fitted with the peak rather than read off the data."
        ),
    ),
    ModelSpec(
        id="lorentzian_centered",
        name="Lorentzian peak (centred)",
        category="peak",
        description="A Lorentzian peak centred on x = 0, with heavy tails.",
        example=ModelExample({"a": 3.0, "g": 0.9}, x_range=(-4.0, 4.0)),
        expression="a*g**2/(x**2+g**2)",
        parameters=_P_LORENTZIAN_CENTERED,
        initial_guess=_profile_guess(
            parameters=_P_LORENTZIAN_CENTERED,
            linear=("a",),
            axes=(_Axis("g", _widths(), relative=True, lower=0.0),),
            columns=_lorentzian_centered_columns,
        ),
        domain_check=_domain_check(),
        references=(_SNEDDON, _BARD),
        assumptions=(
            "The peak is centred on x = 0; shift the data first if the peak is elsewhere. The "
            "tails fall off like 1/x**2."
        ),
    ),
    ModelSpec(
        id="peak_lorentzian",
        name="Lorentzian peak",
        category="peak",
        description="A Lorentzian peak of any position, with heavy tails.",
        example=ModelExample({"a": 3.0, "mu": 0.7, "g": 0.9}, x_range=(-3.0, 5.0)),
        expression="a*g**2/((x-mu)**2+g**2)",
        parameters=_P_LORENTZIAN,
        initial_guess=_profile_guess(
            parameters=_P_LORENTZIAN,
            linear=("a",),
            axes=(
                _Axis("mu", _positions()),
                _Axis("g", _widths(), relative=True, lower=0.0),
            ),
            columns=_lorentzian_columns,
        ),
        domain_check=_domain_check(),
        references=(_SNEDDON, _BARD),
        assumptions=(
            "The peak is symmetric about mu and its tails fall off like 1/(x-mu)**2; errors "
            "are additive on y."
        ),
    ),
    ModelSpec(
        id="peak_lorentzian_offset",
        name="Lorentzian peak on a baseline",
        category="peak",
        description="A Lorentzian peak standing on a constant baseline.",
        example=ModelExample({"c": 0.4, "a": 3.0, "mu": 0.7, "g": 0.9}, x_range=(-3.0, 5.0)),
        expression="c+a*g**2/((x-mu)**2+g**2)",
        parameters=_P_LORENTZIAN_OFFSET,
        initial_guess=_profile_guess(
            parameters=_P_LORENTZIAN_OFFSET,
            linear=("c", "a"),
            axes=(
                _Axis("mu", _positions()),
                _Axis("g", _widths(), relative=True, lower=0.0),
            ),
            columns=_lorentzian_offset_columns,
        ),
        domain_check=_domain_check(),
        references=(_SNEDDON, _BARD),
        assumptions=(
            "The peak is symmetric about mu, the baseline is flat and the tails fall off like "
            "1/(x-mu)**2; errors are additive on y."
        ),
    ),
    ModelSpec(
        id="peak_pearson_vii",
        name="Pearson VII peak",
        category="peak",
        description="A peak whose height falls off like a power law: the generalised Lorentzian.",
        example=ModelExample({"a": 3.0, "mu": 0.7, "g": 0.9, "nu": 1.5}, x_range=(-3.0, 5.0)),
        expression="a/(1+((x-mu)/g)**2)**nu",
        parameters=_P_PEARSON_VII,
        initial_guess=_profile_guess(
            parameters=_P_PEARSON_VII,
            linear=("a",),
            axes=(
                _Axis("mu", _positions()),
                _Axis("g", _widths(), relative=True, lower=0.0),
                _Axis(
                    "nu",
                    _fixed([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]),
                    relative=True,
                    lower=0.0,
                ),
            ),
            columns=_pearson_vii_columns,
        ),
        domain_check=_domain_check(),
        references=(_PEARSON, _BARD),
        assumptions=(
            "The peak is symmetric about mu; nu = 1 gives a Lorentzian and larger nu gives "
            "lighter tails (the height falls off like |x-mu|**(-2*nu)). Errors are additive "
            "on y."
        ),
    ),
    ModelSpec(
        id="peak_pseudo_voigt",
        name="Pseudo-Voigt peak",
        category="peak",
        description="A Gaussian and a Lorentzian of the same width, mixed by a fraction eta.",
        example=ModelExample({"a": 3.0, "mu": 0.7, "w": 1.6, "eta": 0.4}, x_range=(-3.0, 5.0)),
        expression=("a*((1-eta)*exp(-4*log(2)*(x-mu)**2/w**2)+eta*(w/2)**2/((x-mu)**2+(w/2)**2))"),
        parameters=_P_PSEUDO_VOIGT,
        initial_guess=_profile_guess(
            parameters=_P_PSEUDO_VOIGT,
            linear=("a",),
            axes=(
                _Axis("mu", _positions()),
                _Axis("w", _widths(), relative=True, lower=0.0),
                _Axis(
                    "eta",
                    _fixed([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
                    lower=0.0,
                    upper=1.0,
                ),
            ),
            columns=_pseudo_voigt_columns,
        ),
        domain_check=_domain_check(),
        references=(_THOMPSON, _OLIVERO, _IDA),
        assumptions=(
            "Both components share the full width at half maximum w; eta = 0 gives a Gaussian "
            "and eta = 1 a Lorentzian. The curve is an approximation of the true Voigt "
            "profile, not the convolution itself."
        ),
    ),
    ModelSpec(
        id="peak_voigt",
        name="Voigt peak",
        category="peak",
        description="The true Voigt profile: a Gaussian convolved with a Lorentzian.",
        example=ModelExample(
            {"a": 2.0, "mu": 0.7, "sigma": 0.6, "gamma": 0.35}, x_range=(-3.0, 5.0)
        ),
        expression="a*voigt(x-mu, sigma, gamma)",
        parameters=_P_VOIGT,
        initial_guess=_profile_guess(
            parameters=_P_VOIGT,
            linear=("a",),
            axes=(
                _Axis("mu", _positions(13)),
                _Axis("sigma", _widths(5), relative=True, lower=0.0),
                _Axis("gamma", _widths(5), relative=True, lower=0.0),
            ),
            columns=_voigt_columns,
        ),
        domain_check=_domain_check(),
        references=(_VOIGT, _OLIVERO, _IDA),
        assumptions=(
            "The profile is the convolution of a Gaussian of standard deviation sigma with a "
            "Lorentzian of half width gamma; it is area-normalised, so a is the area of the "
            "peak."
        ),
    ),
    ModelSpec(
        id="peak_emg",
        name="Exponentially modified Gaussian peak",
        category="peak",
        description="A Gaussian peak with an exponential tail on the right.",
        example=ModelExample({"a": 2.0, "mu": 0.5, "sigma": 0.7, "tau": 0.6}, x_range=(-2.0, 4.0)),
        expression=(
            "a/(2*tau)*exp(-(x-mu)/tau+sigma**2/(2*tau**2))"
            "*erfc((mu+sigma**2/tau-x)/(sqrt(2)*sigma))"
        ),
        parameters=_P_EMG,
        initial_guess=_profile_guess(
            parameters=_P_EMG,
            linear=("a",),
            axes=(
                _Axis("mu", _positions(13)),
                _Axis("sigma", _widths(5), relative=True, lower=0.0),
                _Axis("tau", _widths(5), relative=True, lower=_TINY),
            ),
            columns=_emg_columns,
        ),
        domain_check=_domain_check(),
        references=(_KALAMBET, _BARD),
        assumptions=(
            "The peak is a Gaussian of standard deviation sigma convolved with an exponential "
            "of mean tau, so it has a longer right-hand tail than a Gaussian. For sigma/tau "
            "above about 38 the closed form is no longer finite around the peak; fit a "
            "Gaussian peak when the exponential tail is that much narrower than the Gaussian."
        ),
    ),
    ModelSpec(
        id="peak_skew_normal",
        name="Skew-normal peak",
        category="peak",
        description="A Gaussian peak tilted to one side by the normal cumulative function.",
        example=ModelExample({"a": 2.0, "mu": 0.0, "s": 1.0, "alpha": 4.0}, x_range=(-3.0, 4.0)),
        expression=("a*exp(-(x-mu)**2/(2*s**2))*(1+erf(alpha*(x-mu)/(sqrt(2)*s)))/(s*sqrt(2*pi))"),
        parameters=_P_SKEW_NORMAL,
        initial_guess=_profile_guess(
            parameters=_P_SKEW_NORMAL,
            linear=("a",),
            axes=(
                _Axis("mu", _positions(13)),
                _Axis("s", _widths(5), relative=True, lower=0.0),
                _Axis(
                    "alpha",
                    _fixed([-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0]),
                    relative=True,
                ),
            ),
            columns=_skew_normal_columns,
        ),
        domain_check=_domain_check(),
        references=(_AZZALINI, _BARD),
        assumptions=(
            "The peak is the skew-normal density of Azzalini: a Gaussian tilted by the normal "
            "cumulative function. alpha = 0 gives a symmetric peak."
        ),
    ),
    ModelSpec(
        id="peak_split_gaussian",
        name="Split Gaussian peak",
        category="peak",
        description="A Gaussian peak with a different width on each side of its centre.",
        example=ModelExample({"a": 3.0, "mu": 0.5, "s1": 0.9, "s2": 0.4}, x_range=(-3.0, 4.0)),
        expression="a*exp(-min(x-mu,0)**2/(2*s1**2)-max(x-mu,0)**2/(2*s2**2))",
        parameters=_P_SPLIT_GAUSSIAN,
        initial_guess=_profile_guess(
            parameters=_P_SPLIT_GAUSSIAN,
            linear=("a",),
            axes=(
                _Axis("mu", _positions()),
                _Axis("s1", _widths(), relative=True, lower=0.0),
                _Axis("s2", _widths(), relative=True, lower=0.0),
            ),
            columns=_split_gaussian_columns,
        ),
        domain_check=_domain_check(),
        references=(_BARD, _NUMERICAL_RECIPES),
        assumptions=(
            "The peak has one standard deviation below mu and another above it; the two "
            "halves join with the same height and the same slope at mu."
        ),
    ),
    ModelSpec(
        id="peak_power_tail",
        name="Power-law peak",
        category="peak",
        description="A peak whose height falls off as a power of the distance from its centre.",
        example=ModelExample({"a": 3.0, "mu": 0.5, "w": 1.2, "b": 2.5}, x_range=(-4.0, 5.0)),
        expression="a*(1+abs((x-mu)/w))**(-b)",
        parameters=_P_POWER_PEAK,
        initial_guess=_profile_guess(
            parameters=_P_POWER_PEAK,
            linear=("a",),
            axes=(
                _Axis("mu", _positions()),
                _Axis("w", _widths(), relative=True, lower=0.0),
                _Axis(
                    "b",
                    _fixed([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
                    relative=True,
                    lower=0.0,
                ),
            ),
            columns=_power_peak_columns,
        ),
        domain_check=_domain_check(),
        references=(_BARD, _NUMERICAL_RECIPES),
        assumptions=(
            "The peak is symmetric about mu and its height falls off like the power b; b = 2 "
            "gives a Lorentzian shape."
        ),
    ),
    ModelSpec(
        id="peak_two_gaussian",
        name="Two Gaussian peaks",
        category="peak",
        description="Two Gaussian peaks on a common baseline, the second one to the right.",
        example=ModelExample(
            {"a1": 3.0, "mu1": 0.0, "s1": 0.55, "a2": 2.0, "d": 5.0, "s2": 0.8},
            x_range=(-2.0, 8.0),
            n=120,
        ),
        expression="a1*exp(-(x-mu1)**2/(2*s1**2))+a2*exp(-(x-(mu1+d))**2/(2*s2**2))",
        parameters=_P_TWO_GAUSSIAN,
        initial_guess=_profile_guess(
            parameters=_P_TWO_GAUSSIAN,
            linear=("a1", "a2"),
            axes=(
                _Axis("mu1", _positions()),
                _Axis("s1", _widths(), relative=True, lower=0.0),
                _Axis(
                    "d",
                    _fixed(np.geomspace(0.5, 10.0, 9)),
                    relative=True,
                    lower=0.0,
                ),
                _Axis("s2", _widths(), relative=True, lower=0.0),
            ),
            columns=_two_gaussian_columns,
            start=_two_peak_start,
        ),
        domain_check=_domain_check(),
        references=(_GAUSS, _BARD),
        assumptions=(
            "The two peaks are on the same baseline and the second one is the later of the "
            "two (d >= 0), so the fit cannot swap them; the data must show both peaks."
        ),
    ),
    ModelSpec(
        id="peak_three_gaussian",
        name="Three Gaussian peaks",
        category="peak",
        description="Three Gaussian peaks on a common baseline, in order of position.",
        example=ModelExample(
            {
                "a1": 3.0,
                "mu1": 0.0,
                "s1": 0.5,
                "a2": 2.0,
                "d2": 4.0,
                "s2": 0.6,
                "a3": 1.5,
                "d3": 4.0,
                "s3": 0.7,
            },
            x_range=(-2.0, 11.0),
            n=140,
        ),
        expression=(
            "a1*exp(-(x-mu1)**2/(2*s1**2))+a2*exp(-(x-(mu1+d2))**2/(2*s2**2))"
            "+a3*exp(-(x-(mu1+d2+d3))**2/(2*s3**2))"
        ),
        parameters=_P_THREE_GAUSSIAN,
        initial_guess=_profile_guess(
            parameters=_P_THREE_GAUSSIAN,
            linear=("a1", "a2", "a3"),
            axes=(
                _Axis("mu1", _positions()),
                _Axis("s1", _widths(), relative=True, lower=0.0),
                _Axis(
                    "d2",
                    _fixed(np.geomspace(0.5, 10.0, 9)),
                    relative=True,
                    lower=0.0,
                ),
                _Axis("s2", _widths(), relative=True, lower=0.0),
                _Axis(
                    "d3",
                    _fixed(np.geomspace(0.5, 10.0, 9)),
                    relative=True,
                    lower=0.0,
                ),
                _Axis("s3", _widths(), relative=True, lower=0.0),
            ),
            columns=_three_gaussian_columns,
            start=_three_peak_start,
        ),
        domain_check=_domain_check(),
        references=(_GAUSS, _BARD),
        assumptions=(
            "The three peaks are on the same baseline and ordered by position (d2, d3 >= 0), "
            "so the fit cannot permute them; the data must show all three peaks."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # A Gaussian peak on a baseline and a Weibull growth curve of shape c = 2 describe the
    # same curve: d+(a-d)*(1-exp(-(x/b)**2)) is a Gaussian of centre 0 and width b/sqrt(2)
    # standing on the baseline a (the Weibull example uses c = 2).
    ("peak_gaussian_offset", "weibull_growth"),
    # The quadratic inverse polynomial 1/(a0+a1*x+a2*x**2) is a Lorentzian of any centre:
    # the peak and the (centred) Cauchy density are the same curve in other parameters.
    ("peak_lorentzian", "invpoly_2"),
)
