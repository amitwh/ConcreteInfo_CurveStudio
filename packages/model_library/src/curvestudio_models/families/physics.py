"""Physics models: transport laws, spectroscopy, thermal physics and solid-state curves.

Most of these models are linearisable, and the initial guess uses that
linearisation: an Arrhenius rate is a straight line in ``ln k`` against ``1/T``,
a Clausius-Clapeyron vapour pressure is a straight line in ``ln P`` against
``1/T``, a Sellmeier dispersion is a straight line in ``1/(n**2 - 1)`` against
``1/lambda**2``. Where the model has one shape parameter that no algebra
linearises (the pole of an Antoine equation, the centre of a Lorentzian), the guess
searches that parameter and reads the rest off a line.

The physical constants that the models carry (the gas constant, the Planck
constant) are folded into the fitted parameters, so a curve is fitted as it is
measured rather than in a particular system of units.
"""

import math

import numpy as np

from curvestudio.models import ModelExample, ModelSpec, Parameter
from curvestudio_models.families._kit import (
    FloatArray,
    bespoke_guess,
    best_of_grid,
    clean,
    domain_check,
    lstsq,
    pattern_minimum,
    scale,
    scan_refine,
    span,
)

# References are cited only where the work treats the model. Author, year and title
# are given; a volume or page is added only where it is certain.
_ARRHENIUS = "Arrhenius S. (1889) Ueber den Einfluss der Temperatur auf die Elektrolyse"
_EYRING = "Eyring H. (1935) The activated complex in chemical reactions, J. Chem. Phys. 3:107-115"
_PROPERTIES = (
    "Reid R.C., Prausnitz J.M., Poling B.E. (1987) The Properties of Gases and Liquids, "
    "4th ed., McGraw-Hill"
)
_SELLMEIER = "Sellmeier W. (1872), Annalen der Physik und Chemie"
_OPTICS = "Born M., Wolf E. (1999) Principles of Optics, 7th ed., Cambridge University Press"
_WIEN = (
    "Wien W. (1893) Eine neue Beziehung der Strahlung schwarzer Körper zum zweiten "
    "Hauptsatz der Wärmelehre"
)
_STEFAN = "Stefan J. (1879), Sitzungsberichte der Kaiserlichen Akademie der Wissenschaften (Wien)"
_PLANCK = "Planck M. (1900) Zur Theorie des Gesetzes der Energieverteilung im Normalspectrum"
_VOGEL = "Vogel H. (1921), Physikalische Zeitschrift"
_TAMMANN = "Tammann G., Hesse G. (1926), Zeitschrift für anorganische und allgemeine Chemie"
_EINSTEIN = (
    "Einstein A. (1907) Die Plancksche Theorie der Strahlung und die Theorie der "
    "spezifischen Wärme, Annalen der Physik"
)
_DEBYE_POLAR = "Debye P. (1929) Polar Molecules, Chemical Catalog Company"
_KITTEL = "Kittel C. (2004) Introduction to Solid State Physics, 8th ed., Wiley"
_MAXWELL = "Maxwell J.C. (1860) Illustrations of the dynamical theory of gases"
_BOLTZMANN = "Boltzmann L. (1868) Studien über das Gleichgewicht der lebendigen Kraft"
_BREIT_WIGNER = "Breit G., Wigner E. (1936) Capture of slow neutrons, Phys. Rev. 49:519-531"
_FERMI = "Fermi E. (1926) Zur Quantelung des idealen einatomigen Gases, Z. Phys. 36:902"
_BOSE = "Bose S.N. (1924) Plancks Gesetz und Lichtquantenhypothese, Z. Phys. 26:178-181"
_MOSELEY = "Moseley H.G.J. (1913) High-frequency spectra, Nature 92:103-104"
_TREATISE = "Maxwell J.C. (1873) A Treatise on Electricity and Magnetism, Clarendon Press"
_ELECTRODYNAMICS = "Zangwill A. (2012) Modern Electrodynamics, Cambridge University Press"


# -- small helpers ----------------------------------------------------------------------


def _fit(
    columns: list[FloatArray], target: FloatArray, weights: FloatArray | None = None
) -> FloatArray | None:
    """Least-squares coefficients of ``target`` on ``columns``; ``None`` when not determined."""
    w = np.ones(target.size, dtype=np.float64) if weights is None else weights
    return lstsq(columns, target, w)


def _fallback(values: dict[str, float], defaults: dict[str, float]) -> dict[str, float]:
    """``values`` with every missing or non-finite entry replaced by the default."""
    out: dict[str, float] = {}
    for name, default in defaults.items():
        value = values.get(name, default)
        out[name] = float(value) if math.isfinite(value) else default
    return out


def _sse(curve: FloatArray, y: FloatArray) -> float:
    """Sum of squared residuals; not finite when the curve left the domain."""
    with np.errstate(all="ignore"):
        residual = np.asarray(curve, dtype=np.float64) - y
        total = float(np.sum(residual**2))
    return total if math.isfinite(total) else math.inf


# -- guesses that read a straight line ----------------------------------------------------


def _log_line(x: FloatArray, y: FloatArray, column: FloatArray) -> FloatArray | None:
    """Coefficients of ``ln y`` against a constant and ``column``: ``(intercept, slope)``.

    The column is centred first: a column that hardly varies (the reciprocal
    temperature, say) is nearly parallel to the constant, and the split between
    the two coefficients would then be decided by rounding. The intercept that is
    returned is the value at ``column = 0``.
    """
    keep = np.isfinite(y) & (y > 0.0) & np.isfinite(column)
    if int(np.count_nonzero(keep)) < 2:
        return None
    centred = column[keep]
    middle = float(np.mean(centred))
    coef = _fit([np.ones(int(np.count_nonzero(keep))), centred - middle], np.log(y[keep]), None)
    if coef is None:
        return None
    return np.array([coef[0] - coef[1] * middle, coef[1]])


def _arrhenius(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*exp(-b/x)``: ``ln y`` is a straight line in ``1/x``."""
    xs, ys = clean(x, y)
    defaults = {"a": scale(ys), "b": span(xs)}
    coef = _log_line(xs, ys, 1.0 / xs)
    if coef is None:
        return defaults
    return _fallback(
        {"a": math.exp(min(coef[0], 700.0)), "b": -coef[1]},
        defaults,
    )


def _eyring(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*x*exp(-b/x)``: ``ln(y/x)`` is a straight line in ``1/x``."""
    xs, ys = clean(x, y)
    defaults = {"a": scale(ys), "b": span(xs)}
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 2:
        return defaults
    coef = _log_line(xs[keep], ys[keep] / xs[keep], 1.0 / xs[keep])
    if coef is None:
        return defaults
    return _fallback(
        {"a": math.exp(min(coef[0], 700.0)), "b": -coef[1]},
        defaults,
    )


def _einstein(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*(b/x)**2*exp(-b/x)/(1-exp(-b/x))**2``: linear in ``a``; ``b`` is searched.

    The overflow-safe form of the law is used here as well: ``exp(-b/x)`` stays in
    ``(0, 1)`` for every positive temperature, so no intermediate overflows.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": 0.5 * extent}
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    xs, ys = xs[keep], ys[keep]

    def line(theta: float) -> tuple[float, float]:
        with np.errstate(all="ignore"):
            ratio = theta / xs
            column = ratio**2 * np.exp(-ratio) / (1.0 - np.exp(-ratio)) ** 2
        coef = _fit([column], ys)
        if coef is None or coef[0] <= 0.0:
            return math.nan, math.inf
        with np.errstate(all="ignore"):
            return float(coef[0]), _sse(coef[0] * column, ys)

    def error(theta: float) -> float:
        if theta <= 0.0:
            return math.inf
        return line(theta)[1]

    best, _ = scan_refine(error, 1e-3 * extent, 1e3 * extent, count=80)
    amplitude, _ = line(best)
    if not math.isfinite(amplitude):
        return defaults
    return _fallback({"a": amplitude, "b": best}, defaults)


def _antoine(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = exp(a - b/(c+x))``.

    For a fixed ``c`` the model is a straight line in ``ln y`` against ``1/(c+x)``;
    ``c`` is searched and the two coefficients are read off the line.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": 0.0, "b": extent, "c": extent}
    keep = ys > 0.0
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    xs, ys = xs[keep], ys[keep]

    def line(constant: float) -> tuple[float, float, float]:
        with np.errstate(all="ignore"):
            column = 1.0 / (constant + xs)
        coef = _log_line(xs, ys, column)
        if coef is None:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            curve = np.exp(coef[0] + coef[1] * column)
        return coef[0], coef[1], _sse(curve, ys)

    def error(constant: float) -> float:
        if constant + float(xs[0]) <= 0.0:
            return math.inf
        return line(constant)[2]

    best, _ = scan_refine(error, 1e-3, 1e4, count=80)
    intercept, slope, _ = line(best)
    if not math.isfinite(intercept):
        return defaults
    return _fallback({"a": intercept, "b": -slope, "c": best}, defaults)


def _sellmeier(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = sqrt(1 + b*x**2/(x**2 - c))``.

    For a fixed resonance ``c`` the model is ``y**2 - 1 = b*x**2/(x**2 - c)``, a
    straight line through the origin, so the strength follows from the line and
    the resonance is searched on the original scale.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"b": 1.0, "c": 0.01 * extent**2}
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    xs, ys = xs[keep], ys[keep]
    lowest = float(np.min(xs**2))

    def line(resonance: float) -> tuple[float, float]:
        if resonance >= lowest:
            return math.nan, math.inf
        with np.errstate(all="ignore"):
            column = xs**2 / (xs**2 - resonance)
            strength = float(np.sum(column * (ys**2 - 1.0)) / np.sum(column**2))
        if not math.isfinite(strength) or strength <= 0.0:
            return math.nan, math.inf
        with np.errstate(all="ignore"):
            curve = np.sqrt(1.0 + strength * column)
        return strength, _sse(curve, ys)

    def error(resonance: float) -> float:
        return line(resonance)[1]

    best, _ = scan_refine(error, 1e-9 * lowest, lowest * 0.999, count=60, geometric=False)
    strength, _ = line(best)
    if not math.isfinite(strength):
        return defaults
    return _fallback({"b": strength, "c": best}, defaults)


def _cauchy(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b/x**2 + c/x**4``: a straight line in ``1/x**2`` and ``1/x**4``."""
    xs, ys = clean(x, y)
    defaults = {"a": float(np.mean(ys)) if ys.size else 0.0, "b": 0.0, "c": 0.0}
    if xs.size < 4:
        return defaults
    coef = _fit([np.ones(xs.size), 1.0 / xs**2, 1.0 / xs**4], ys)
    if coef is None:
        return defaults
    return _fallback({"a": coef[0], "b": coef[1], "c": coef[2]}, defaults)


def _wien(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = b/x``: the product of the two is the constant of the law."""
    xs, ys = clean(x, y)
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 1:
        return {"b": scale(ys)}
    return {"b": float(np.median(ys[keep] * xs[keep]))}


def _stefan_boltzmann(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*x**4``: the ratio of the two is the constant of the law."""
    xs, ys = clean(x, y)
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 1:
        return {"a": scale(ys)}
    with np.errstate(all="ignore"):
        ratio = ys[keep] / xs[keep] ** 4
    return {"a": float(np.median(ratio))}


def _coulomb(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a/x**2``: the product of the two is the constant of the law."""
    xs, ys = clean(x, y)
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 1:
        return {"a": scale(ys)}
    with np.errstate(all="ignore"):
        ratio = ys[keep] * xs[keep] ** 2
    return {"a": float(np.median(ratio))}


def _planck(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a/(exp(b/x) - 1)``.

    For a fixed ``a`` the model gives ``ln(1 + a/y) = b/x``, a straight line
    through the origin in ``1/x``; ``a`` is searched and the slope gives ``b``.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": extent}
    keep = ys > 0.0
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    xs, ys = xs[keep], ys[keep]

    def line(prefactor: float) -> tuple[float, float, float]:
        if prefactor <= 0.0:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            inner = 1.0 + prefactor / ys
        good = inner > 0.0
        if int(np.count_nonzero(good)) < 3:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            slope = float(
                np.sum((1.0 / xs[good]) * np.log(inner[good])) / np.sum((1.0 / xs[good]) ** 2)
            )
        if not math.isfinite(slope) or slope <= 0.0:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            curve = prefactor / np.expm1(slope / xs)
        return prefactor, slope, _sse(curve, ys)

    def error(prefactor: float) -> float:
        return line(prefactor)[2]

    best, _ = scan_refine(error, extent * 1e-6, extent * 1e4, count=80)
    prefactor, slope, _ = line(best)
    if not math.isfinite(prefactor):
        return defaults
    return _fallback({"a": prefactor, "b": slope}, defaults)


def _vogel_fulcher_tammann(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = exp(a + b/(x - c))``.

    For a fixed ``c`` the model gives ``ln y = a + b/(x - c)``, a straight line; the
    Vogel temperature ``c`` is searched below the data and the line gives the rest.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": math.log(max(scale(ys), 1e-12)), "b": extent, "c": 0.0}
    keep = ys > 0.0
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    xs, ys = xs[keep], ys[keep]
    lowest = float(xs[0])

    def line(constant: float) -> tuple[float, float, float]:
        with np.errstate(all="ignore"):
            column = 1.0 / (xs - constant)
        coef = _log_line(xs, ys, column)
        if coef is None:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            curve = np.exp(np.clip(coef[0] + coef[1] * column, -700.0, 700.0))
        return coef[0], coef[1], _sse(curve, ys)

    def error(constant: float) -> float:
        if constant >= lowest:
            return math.inf
        return line(constant)[2]

    best, _ = scan_refine(error, lowest - extent, lowest - 1e-9 * extent, count=80)
    intercept, slope, _ = line(best)
    if not math.isfinite(intercept):
        return defaults
    return _fallback({"a": intercept, "b": slope, "c": best}, defaults)


def _debye_dielectric(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b/(x**2 + c**2)``: for a fixed ``c`` the model is a straight line."""
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": float(np.mean(ys)) if ys.size else 0.0, "b": 0.0, "c": 0.0}
    if xs.size < 4:
        return defaults

    def line(constant: float) -> tuple[float, float, float]:
        with np.errstate(all="ignore"):
            column = 1.0 / (xs**2 + constant**2)
        coef = _fit([np.ones(xs.size), column], ys)
        if coef is None:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            curve = coef[0] + coef[1] * column
        return coef[0], coef[1], _sse(curve, ys)

    def error(constant: float) -> float:
        if constant < 0.0:
            return math.inf
        return line(constant)[2]

    best, _ = scan_refine(error, 1e-6 * extent, 10.0 * extent, count=60)
    intercept, slope, _ = line(best)
    if not math.isfinite(intercept):
        return defaults
    return _fallback({"a": intercept, "b": slope, "c": best}, defaults)


def _maxwell_boltzmann(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*x**2*exp(-b*x**2)``: ``ln(y/x**2)`` is a straight line in ``x**2``."""
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": 1.0 / extent**2}
    keep = (xs > 0.0) & (ys > 0.0)
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    with np.errstate(all="ignore"):
        reduced = ys[keep] / xs[keep] ** 2
    coef = _log_line(xs[keep], reduced, xs[keep] ** 2)
    if coef is None:
        return defaults
    return _fallback(
        {"a": math.exp(min(coef[0], 700.0)), "b": -coef[1]},
        defaults,
    )


def _lorentzian(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a/(1 + (x - c)**2/b**2)``.

    Reciprocating gives ``1/y = (1/a)(1 + (x - c)**2/b**2)``, a straight line in
    ``x**2`` and ``x``; the weight ``y**4`` keeps the noisy tails from taking over.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": extent, "c": float(np.median(xs)) if xs.size else 0.0}
    keep = ys > 0.0
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    xs, ys = xs[keep], ys[keep]
    with np.errstate(all="ignore"):
        weights = (ys / scale(ys)) ** 4
    coef = _fit(
        [np.ones(xs.size), xs, xs**2],
        1.0 / ys,
        weights,
    )
    if coef is None or coef[0] <= 0.0:
        return defaults
    # 1/y = (b**2 + c**2 - 2*c*x + x**2)/(a*b**2): the three coefficients give
    # c from their ratio and b**2 from the constant term.
    centre = -coef[1] / (2.0 * coef[2])
    squared = coef[0] / coef[2] - centre**2
    if squared <= 0.0 or coef[2] <= 0.0:
        return defaults
    a = 1.0 / (coef[2] * squared)
    return _fallback({"a": a, "b": math.sqrt(squared), "c": centre}, defaults)


def _sellmeier_two(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = sqrt(1 + b*x**2/(x**2 - c) + d*x**2/(x**2 - c - g))``.

    For fixed resonances the squared model minus one is linear in the two strengths,
    so the two poles (ordered, both below the data) are searched over a grid and the
    strengths follow from one least-squares solve.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"b": 1.0, "c": 0.01 * extent**2, "d": 0.5, "g": 0.01 * extent**2}
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 5:
        return defaults
    xs, ys = xs[keep], ys[keep]
    lowest = float(np.min(xs**2))

    def line(first: float, second: float) -> tuple[float, float, float]:
        with np.errstate(all="ignore"):
            columns = [xs**2 / (xs**2 - first), xs**2 / (xs**2 - second)]
            coef = _fit(columns, ys**2 - 1.0)
            if coef is None:
                return math.nan, math.nan, math.inf
            curve = np.sqrt(np.maximum(1.0 + coef[0] * columns[0] + coef[1] * columns[1], 0.0))
        return float(coef[0]), float(coef[1]), _sse(curve, ys)

    def error(point: FloatArray) -> float:
        first, gap = float(point[0]), float(point[1])
        if first <= 0.0 or gap <= 0.0 or first + gap >= lowest:
            return math.inf
        return line(first, first + gap)[2]

    grid = np.geomspace(1e-4 * lowest, 0.4 * lowest, 12)
    gaps = np.geomspace(1e-4 * lowest, 0.8 * lowest, 12)
    scored: list[tuple[float, FloatArray]] = []
    for first in grid:
        for gap in gaps:
            value = error(np.array([first, gap]))
            if math.isfinite(value):
                scored.append((value, np.array([first, gap])))
    if not scored:
        return defaults
    scored.sort(key=lambda item: item[0])
    best_value, best_point = math.inf, scored[0][1]
    for _, start in scored[:3]:
        point, value = pattern_minimum(error, start, 0.05 * lowest)
        point, value = pattern_minimum(error, point, 0.005 * lowest)
        if value < best_value:
            best_value, best_point = value, point
    point = best_point
    pole, pole_gap = float(point[0]), float(point[1])
    strength, second_strength, _ = line(pole, pole + pole_gap)
    if not (math.isfinite(strength) and math.isfinite(second_strength)):
        return defaults
    return _fallback(
        {"b": strength, "c": pole, "d": second_strength, "g": pole_gap},
        defaults,
    )


def _fermi_dirac(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = 1/(exp((x - c)/d) + 1)``.

    On the band where the occupation is between 5 % and 95 % the model gives
    ``ln(1/y - 1) = (x - c)/d``, a straight line; ``d`` is searched and the line
    is read on the band.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"c": float(np.median(xs)) if xs.size else 0.0, "d": 0.1 * extent}
    keep = (ys > 0.02) & (ys < 0.98)
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    xs, ys = xs[keep], ys[keep]

    def line(width: float) -> tuple[float, float, float]:
        if width <= 0.0:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            reduced = np.log(1.0 / ys - 1.0)
        coef = _fit([np.ones(xs.size), xs], reduced)
        if coef is None or coef[1] <= 0.0:
            return math.nan, math.nan, math.inf
        slope = 1.0 / width
        centre = -coef[0] / slope
        with np.errstate(all="ignore"):
            curve = 1.0 / (np.exp((xs - centre) / width) + 1.0)
        return centre, width, _sse(curve, ys)

    def error(width: float) -> float:
        return line(width)[2]

    best, _ = scan_refine(error, 1e-6 * extent, 10.0 * extent, count=60)
    centre, width, _ = line(best)
    if not math.isfinite(centre):
        return defaults
    return _fallback({"c": centre, "d": width}, defaults)


def _bose_einstein(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = 1/(exp((x - c)/d) - 1)``.

    On the band where the occupation is between 0.1 and 10 the model gives
    ``ln(1/y + 1) = (x - c)/d``, a straight line; ``d`` is searched.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"c": float(np.median(xs)) if xs.size else 0.0, "d": 0.1 * extent}
    keep = (ys > 0.1) & (ys < 10.0)
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    xs, ys = xs[keep], ys[keep]

    def line(width: float) -> tuple[float, float, float]:
        if width <= 0.0:
            return math.nan, math.nan, math.inf
        with np.errstate(all="ignore"):
            reduced = np.log(1.0 / ys + 1.0)
        coef = _fit([np.ones(xs.size), xs], reduced)
        if coef is None or coef[1] <= 0.0:
            return math.nan, math.nan, math.inf
        slope = 1.0 / width
        centre = -coef[0] / slope
        with np.errstate(all="ignore"):
            curve = 1.0 / np.expm1((xs - centre) / width)
        return centre, width, _sse(curve, ys)

    def error(width: float) -> float:
        return line(width)[2]

    best, _ = scan_refine(error, 1e-6 * extent, 10.0 * extent, count=60)
    centre, width, _ = line(best)
    if not math.isfinite(centre):
        return defaults
    return _fallback({"c": centre, "d": width}, defaults)


def _moseley(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*(x - c)**2``: linear in ``a`` for a fixed vertex ``c``; ``c`` is searched.

    The vertex is the screening constant and lies at or below the lightest element
    measured, so the search keeps it below the data and the curve rises.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys) / max(extent, 1e-12) ** 2, "c": 0.0}
    if xs.size < 3:
        return defaults

    def line(vertex: float) -> tuple[float, float]:
        with np.errstate(all="ignore"):
            column = (xs - vertex) ** 2
        coef = _fit([column], ys)
        if coef is None or coef[0] <= 0.0:
            return math.nan, math.inf
        with np.errstate(all="ignore"):
            return float(coef[0]), _sse(coef[0] * column, ys)

    def error(vertex: float) -> float:
        return line(vertex)[1]

    best, _ = scan_refine(error, float(xs[0]) - extent, float(xs[0]), count=60, geometric=False)
    amplitude, _ = line(best)
    if not math.isfinite(amplitude):
        return defaults
    return _fallback({"a": amplitude, "c": best}, defaults)


def _van_der_waals(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a/(x - b) - c/x**2``.

    The curve is linear in ``a``, so the excluded volume and the attraction are
    searched over a grid and the scale follows from the line.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys) * extent, "b": 0.0, "c": scale(ys) * extent**2}
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    xs, ys = xs[keep], ys[keep]

    def error(point: FloatArray) -> float:
        excluded, attraction = float(point[0]), float(point[1])
        if excluded <= 0.0 or excluded >= float(xs[0]):
            return math.inf
        with np.errstate(all="ignore"):
            basis = 1.0 / (xs - excluded)
        coef = _fit([basis], ys + attraction / xs**2)
        if coef is None:
            return math.inf
        with np.errstate(all="ignore"):
            curve = coef[0] * basis - attraction / xs**2
        return _sse(curve, ys)

    grid = np.geomspace(1e-4 * extent, 0.9 * float(xs[0]), 14)
    attraction = np.geomspace(1e-6 * extent**2, 1e4 * extent**2, 14)
    point, _ = best_of_grid(error, grid, attraction)
    point, _ = pattern_minimum(error, point, 0.05 * extent)
    excluded, attraction_value = float(point[0]), float(point[1])
    with np.errstate(all="ignore"):
        basis = 1.0 / (xs - excluded)
    coef = _fit([basis], ys + attraction_value / xs**2)
    if coef is None:
        return defaults
    return _fallback({"a": float(coef[0]), "b": excluded, "c": attraction_value}, defaults)


def _lorentz_oscillator(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b/(c**2 - x**2)``.

    For a fixed ``a`` the model gives ``1/(y - a)`` as a straight line in ``x**2``;
    ``a`` is searched and the resonance frequency and the strength are read off it.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": float(np.min(ys)) if ys.size else 0.0, "b": 0.0, "c": 2.0 * extent}
    if xs.size < 4:
        return defaults

    def line(background: float) -> tuple[float, float, float]:
        with np.errstate(all="ignore"):
            reduced = 1.0 / (ys - background)
        coef = _fit([np.ones(xs.size), xs**2], reduced)
        if coef is None or coef[1] >= 0.0:
            return math.nan, math.nan, math.inf
        strength = -1.0 / coef[1]
        resonance = math.sqrt(max(-coef[0] / coef[1], 0.0))
        with np.errstate(all="ignore"):
            curve = background + strength / (resonance**2 - xs**2)
        return strength, resonance, _sse(curve, ys)

    def error(background: float) -> float:
        return line(background)[2]

    lowest = float(np.min(ys))
    best, _ = scan_refine(error, lowest - 10.0 * scale(ys), lowest - 1e-9 * scale(ys), count=60)
    strength, resonance, _ = line(best)
    if not math.isfinite(strength):
        return defaults
    return _fallback({"a": best, "b": strength, "c": resonance}, defaults)


# -- the models --------------------------------------------------------------------------


_P_ARRHENIUS = (
    Parameter(
        "a",
        "Pre-exponential factor: the rate the Arrhenius law approaches at high temperature",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Activation temperature: the energy of activation divided by the gas constant",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_EYRING = (
    Parameter(
        "a",
        "Pre-exponential factor of the transition-state expression",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Activation temperature: the enthalpy of activation divided by the gas constant",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_ANTONIE = (
    Parameter("a", "Logarithm of the pre-exponential pressure", lower=-math.inf, upper=math.inf),
    Parameter(
        "b",
        "Pressure coefficient: the energy term divided by the gas constant",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("c", "Temperature offset of the Antoine equation", lower=-math.inf, upper=math.inf),
)
_P_SELLMEIER = (
    Parameter("b", "Oscillator strength of the absorption band", lower=0.0, upper=math.inf),
    Parameter(
        "c",
        "Resonance wavelength squared of the absorption band",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_SELLMEIER_TWO = (
    Parameter("b", "Oscillator strength of the first absorption band", lower=0.0, upper=math.inf),
    Parameter(
        "c",
        "Resonance wavelength squared of the first band",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("d", "Oscillator strength of the second absorption band", lower=0.0, upper=math.inf),
    Parameter(
        "g",
        "Gap to the resonance wavelength squared of the second band (ordered c <= c + g)",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_CAUCHY = (
    Parameter("a", "Refractive index at infinite wavelength", lower=-math.inf, upper=math.inf),
    Parameter(
        "b",
        "Coefficient of the wavelength dispersion (1/lambda**2)",
        lower=-math.inf,
        upper=math.inf,
    ),
    Parameter(
        "c",
        "Coefficient of the second dispersion term (1/lambda**4)",
        lower=-math.inf,
        upper=math.inf,
    ),
)
_P_WIEN = (
    Parameter(
        "b",
        "Constant of the displacement law: the product of the peak wavelength and the temperature",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_STEFAN = (
    Parameter("a", "Emissive power constant of the black body", lower=0.0, upper=math.inf),
)
_P_COULOMB = (
    Parameter(
        "a",
        "Constant of the inverse-square law: the product of the field and the squared distance",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_PLANCK = (
    Parameter("a", "Prefactor of the spectral radiance", lower=0.0, upper=math.inf),
    Parameter(
        "b",
        "Second radiation constant: the product of the photon energy and the temperature",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_VFT = (
    Parameter(
        "a",
        "Logarithm of the viscosity at infinite temperature",
        lower=-math.inf,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Pseudo-activation temperature of the viscosity (the B of the law)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "c",
        "Vogel temperature: the point the data must stay above, where the fit diverges",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_DEBYE_DIELECTRIC = (
    Parameter("a", "Permittivity at infinite frequency", lower=-math.inf, upper=math.inf),
    Parameter(
        "b",
        "Strength of the dipolar relaxation (the dielectric increment is b/c**2)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "c",
        "Characteristic relaxation rate: the reciprocal of the relaxation time",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_MAXWELL = (
    Parameter("a", "Prefactor of the speed distribution", lower=0.0, upper=math.inf),
    Parameter(
        "b",
        "Inverse of the temperature parameter of the distribution",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_LORENTZIAN = (
    Parameter("a", "Height of the resonance", lower=0.0, upper=math.inf),
    Parameter("b", "Half width at half height of the resonance", lower=0.0, upper=math.inf),
    Parameter("c", "Position of the resonance", lower=-math.inf, upper=math.inf),
)
_P_FERMI = (
    Parameter("c", "Energy at which the occupation is one half", lower=-math.inf, upper=math.inf),
    Parameter(
        "d",
        "Width of the transition: the thermal energy of the particles",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_BOSE = (
    Parameter("c", "Energy at which the occupation is one half", lower=-math.inf, upper=math.inf),
    Parameter(
        "d",
        "Width of the transition: the thermal energy of the particles",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_MOSELEY = (
    Parameter(
        "a",
        "Constant of the law: the slope of the square root of the frequency against x",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "c",
        "Screening constant: the x at which the fitted frequency falls to zero",
        lower=-math.inf,
        upper=math.inf,
    ),
)
_P_VAN_DER_WAALS = (
    Parameter(
        "a",
        "Thermal scale: the product of the amount of gas and the gas constant",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("b", "Excluded volume per mole", lower=0.0, upper=math.inf),
    Parameter("c", "Attraction constant of the gas", lower=0.0, upper=math.inf),
)
_P_EINSTEIN = (
    Parameter(
        "a",
        "Dulong-Petit limit of the heat capacity: the value the curve saturates towards",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("b", "Einstein temperature of the solid", lower=0.0, upper=math.inf),
)
_P_LORENTZ_OSCILLATOR = (
    Parameter("a", "Permittivity without the resonance", lower=-math.inf, upper=math.inf),
    Parameter("b", "Strength of the resonance", lower=0.0, upper=math.inf),
    Parameter("c", "Resonance frequency of the oscillator", lower=0.0, upper=math.inf),
)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="physics_arrhenius",
        name="Arrhenius rate",
        category="physics",
        description=(
            "Rate constant that grows exponentially with temperature: a straight line in ln(k) "
            "against 1/T."
        ),
        example=ModelExample({"a": 1000.0, "b": 1500.0}, x_range=(300.0, 600.0)),
        expression="a*exp(-b/x)",
        parameters=_P_ARRHENIUS,
        initial_guess=bespoke_guess(_P_ARRHENIUS, _arrhenius),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_ARRHENIUS,),
        assumptions=(
            "The temperature is positive and in kelvin; the rate constant follows the "
            "Arrhenius law with the activation energy divided by the gas constant."
        ),
    ),
    ModelSpec(
        id="physics_eyring",
        name="Eyring rate",
        category="physics",
        description="Transition-state rate law: a straight line in ln(k/T) against 1/T.",
        example=ModelExample({"a": 2.0, "b": 1500.0}, x_range=(300.0, 600.0)),
        expression="a*x*exp(-b/x)",
        parameters=_P_EYRING,
        initial_guess=bespoke_guess(_P_EYRING, _eyring),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_EYRING,),
        assumptions=(
            "The temperature is positive and in kelvin; the rate constant follows the "
            "transition-state expression of Eyring."
        ),
    ),
    ModelSpec(
        id="physics_einstein_heat_capacity",
        name="Einstein heat capacity",
        category="physics",
        description=(
            "Heat capacity of a solid in the Einstein model: it rises steeply below the Einstein "
            "temperature and saturates at the Dulong-Petit value."
        ),
        example=ModelExample({"a": 25.0, "b": 300.0}, x_range=(50.0, 1000.0)),
        expression="a*(b/x)**2*exp(-b/x)/(1-exp(-b/x))**2",
        parameters=_P_EINSTEIN,
        initial_guess=bespoke_guess(_P_EINSTEIN, _einstein),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_EINSTEIN, _KITTEL),
        assumptions=(
            "The temperature is positive and in kelvin; the solid is a set of independent "
            "harmonic oscillators of one frequency, so the heat capacity approaches the "
            "Dulong-Petit limit a at high temperature."
        ),
    ),
    ModelSpec(
        id="physics_antoine",
        name="Antoine vapour pressure",
        category="physics",
        description="Vapour pressure of a liquid: a straight line in ln(P) against 1/(T + c).",
        example=ModelExample({"a": 10.0, "b": 2000.0, "c": 100.0}, x_range=(300.0, 600.0)),
        expression="exp(a-b/(c+x))",
        parameters=_P_ANTONIE,
        initial_guess=bespoke_guess(_P_ANTONIE, _antoine),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_PROPERTIES,),
        assumptions=(
            "The temperature is positive and in kelvin; the constants are fitted to the "
            "range of measurement, not to the whole curve."
        ),
    ),
    ModelSpec(
        id="physics_sellmeier_1term",
        name="Sellmeier dispersion (one term)",
        category="physics",
        description="Refractive index of a transparent material with one absorption band.",
        example=ModelExample({"b": 2.0, "c": 0.04}, x_range=(0.25, 1.5)),
        expression="sqrt(1+b*x**2/(x**2-c))",
        parameters=_P_SELLMEIER,
        initial_guess=bespoke_guess(_P_SELLMEIER, _sellmeier),
        domain_check=domain_check(x_positive_for="the wavelength"),
        references=(_SELLMEIER, _OPTICS),
        assumptions=(
            "The wavelength is positive and the resonance of the band lies below the "
            "measured range; the index is real over the whole range."
        ),
    ),
    ModelSpec(
        id="physics_cauchy_dispersion",
        name="Cauchy dispersion",
        category="physics",
        description=(
            "Refractive index that falls with the wavelength as a sum of 1/lambda**2 and "
            "1/lambda**4."
        ),
        example=ModelExample({"a": 1.4, "b": 0.02, "c": 0.005}, x_range=(0.3, 1.2)),
        expression="a+b/x**2+c/x**4",
        parameters=_P_CAUCHY,
        initial_guess=bespoke_guess(_P_CAUCHY, _cauchy),
        domain_check=domain_check(x_positive_for="the wavelength"),
        references=(_OPTICS,),
        assumptions=(
            "The wavelength is positive; the normal dispersion of the material is "
            "described by the two terms of the Cauchy series."
        ),
    ),
    ModelSpec(
        id="physics_wien_displacement",
        name="Wien displacement law",
        category="physics",
        description=(
            "Wavelength of the peak of the thermal radiation of a body: it is inversely "
            "proportional to the temperature."
        ),
        example=ModelExample({"b": 2.9e6}, x_range=(500.0, 3000.0)),
        expression="b/x",
        parameters=_P_WIEN,
        initial_guess=bespoke_guess(_P_WIEN, _wien),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_WIEN,),
        assumptions=(
            "The temperature is positive and in kelvin; the peak wavelength follows the "
            "displacement law over the measured range."
        ),
    ),
    ModelSpec(
        id="physics_stefan_boltzmann",
        name="Stefan-Boltzmann law",
        category="physics",
        description=(
            "Emissive power of a black body: it grows with the fourth power of the temperature."
        ),
        example=ModelExample({"a": 5.67e-8}, x_range=(200.0, 1500.0)),
        expression="a*x**4",
        parameters=_P_STEFAN,
        initial_guess=bespoke_guess(_P_STEFAN, _stefan_boltzmann),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_STEFAN,),
        assumptions=(
            "The temperature is positive and in kelvin; the body radiates as a black "
            "body over the measured range."
        ),
    ),
    ModelSpec(
        id="physics_planck_radiance",
        name="Planck spectral radiance",
        category="physics",
        description=(
            "Spectral radiance of a black body at a fixed wavelength as a function of temperature."
        ),
        example=ModelExample({"a": 1000.0, "b": 300.0}, x_range=(100.0, 800.0)),
        expression="a/(exp(b/x)-1)",
        parameters=_P_PLANCK,
        initial_guess=bespoke_guess(_P_PLANCK, _planck),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_PLANCK,),
        assumptions=(
            "The temperature is positive and in kelvin; the wavelength is fixed and the "
            "radiance follows Planck's law."
        ),
    ),
    ModelSpec(
        id="physics_vogel_fulcher_tammann",
        name="Vogel-Fulcher-Tammann law",
        category="physics",
        description=(
            "Viscosity of a supercooled liquid: the logarithm of the viscosity is linear in the "
            "reciprocal distance to the Vogel temperature."
        ),
        example=ModelExample({"a": 0.0, "b": 300.0, "c": 200.0}, x_range=(250.0, 800.0)),
        expression="exp(a+b/(x-c))",
        parameters=_P_VFT,
        initial_guess=bespoke_guess(_P_VFT, _vogel_fulcher_tammann),
        domain_check=domain_check(x_positive_for="the temperature"),
        references=(_VOGEL, _TAMMANN),
        assumptions=(
            "The temperature lies above the Vogel temperature c, where the viscosity diverges; "
            "the logarithm of the viscosity follows the Vogel-Fulcher-Tammann line over the "
            "measured range."
        ),
    ),
    ModelSpec(
        id="physics_debye_dielectric",
        name="Debye dielectric relaxation",
        category="physics",
        description=(
            "Real part of the permittivity of a polar material as a function of the frequency: "
            "it falls from its static value across the relaxation."
        ),
        example=ModelExample({"a": 2.5, "b": 2.0e6, "c": 200.0}, x_range=(0.0, 800.0)),
        expression="a+b/(x**2+c**2)",
        parameters=_P_DEBYE_DIELECTRIC,
        initial_guess=bespoke_guess(_P_DEBYE_DIELECTRIC, _debye_dielectric),
        domain_check=domain_check(x_non_negative=True),
        references=(_DEBYE_POLAR, _KITTEL),
        assumptions=(
            "The frequency is non-negative and a single relaxation time describes the dipoles; "
            "the permittivity falls from the static value a + b/c**2 towards a as the "
            "frequency grows."
        ),
    ),
    ModelSpec(
        id="physics_maxwell_boltzmann_speed",
        name="Maxwell-Boltzmann speed distribution",
        category="physics",
        description="Number of particles per unit speed in a gas at equilibrium.",
        example=ModelExample({"a": 2.0, "b": 0.5}, x_range=(0.0, 4.0)),
        expression="a*x**2*exp(-b*x**2)",
        parameters=_P_MAXWELL,
        initial_guess=bespoke_guess(_P_MAXWELL, _maxwell_boltzmann),
        domain_check=domain_check(x_non_negative=True),
        references=(_MAXWELL, _BOLTZMANN),
        assumptions=(
            "The speed is non-negative and the gas is in equilibrium; the distribution "
            "is the Maxwell-Boltzmann one."
        ),
    ),
    ModelSpec(
        id="physics_lorentzian_resonance",
        name="Lorentzian resonance",
        category="physics",
        description="Absorption line of a damped oscillator: a Lorentzian peak on a zero baseline.",
        example=ModelExample({"a": 10.0, "b": 0.5, "c": 2.0}, x_range=(0.0, 6.0)),
        expression="a/(1+(x-c)**2/b**2)",
        parameters=_P_LORENTZIAN,
        initial_guess=bespoke_guess(_P_LORENTZIAN, _lorentzian),
        domain_check=domain_check(),
        references=(_BREIT_WIGNER,),
        assumptions=(
            "The line is a single damped oscillator on a zero baseline; the measured "
            "range covers the peak."
        ),
    ),
    ModelSpec(
        id="physics_sellmeier_2term",
        name="Sellmeier dispersion (two terms)",
        category="physics",
        description=(
            "Refractive index of a transparent material with two absorption bands below the "
            "measured range."
        ),
        example=ModelExample({"b": 1.5, "c": 0.04, "d": 0.5, "g": 0.16}, x_range=(0.5, 1.2)),
        expression="sqrt(1+b*x**2/(x**2-c)+d*x**2/(x**2-c-g))",
        parameters=_P_SELLMEIER_TWO,
        initial_guess=bespoke_guess(_P_SELLMEIER_TWO, _sellmeier_two),
        domain_check=domain_check(x_positive_for="the wavelength"),
        references=(_SELLMEIER, _OPTICS),
        assumptions=(
            "The wavelength is positive and both resonances of the bands lie below the "
            "measured range; the index is real over the whole range."
        ),
    ),
    ModelSpec(
        id="physics_fermi_dirac_occupation",
        name="Fermi-Dirac occupation",
        category="physics",
        description="Occupation number of a fermionic state as a function of its energy.",
        example=ModelExample({"c": 0.0, "d": 1.0}, x_range=(-6.0, 6.0)),
        expression="1/(exp((x-c)/d)+1)",
        parameters=_P_FERMI,
        initial_guess=bespoke_guess(_P_FERMI, _fermi_dirac),
        domain_check=domain_check(),
        references=(_FERMI,),
        assumptions=(
            "The energy is measured relative to the chemical potential; the states are "
            "fermions at a fixed temperature."
        ),
    ),
    ModelSpec(
        id="physics_bose_einstein_occupation",
        name="Bose-Einstein occupation",
        category="physics",
        description="Occupation number of a bosonic state as a function of its energy.",
        example=ModelExample({"c": 0.0, "d": 1.0}, x_range=(0.2, 6.0)),
        expression="1/(exp((x-c)/d)-1)",
        parameters=_P_BOSE,
        initial_guess=bespoke_guess(_P_BOSE, _bose_einstein),
        domain_check=domain_check(x_positive_for="the energy above the ground state"),
        references=(_BOSE,),
        assumptions=(
            "The energy lies above the ground state; the states are bosons at a fixed temperature."
        ),
    ),
    ModelSpec(
        id="physics_moseley_law",
        name="Moseley law",
        category="physics",
        description=(
            "Frequency of a K-alpha x-ray line: it grows with the square of the atomic number "
            "less a screening constant."
        ),
        example=ModelExample({"a": 0.25, "c": 1.0}, x_range=(3.0, 20.0)),
        expression="a*(x-c)**2",
        parameters=_P_MOSELEY,
        initial_guess=bespoke_guess(_P_MOSELEY, _moseley),
        domain_check=domain_check(),
        references=(_MOSELEY,),
        assumptions=(
            "The x axis is the atomic number; the screening constant c places the vertex of "
            "the parabola at or below the lightest element, so the frequency grows as "
            "a*(x-c)**2 over the measured elements."
        ),
    ),
    ModelSpec(
        id="physics_van_der_waals",
        name="Van der Waals pressure",
        category="physics",
        description=(
            "Pressure of a real gas: the ideal gas law with an excluded volume and an attraction."
        ),
        example=ModelExample({"a": 12.0, "b": 0.5, "c": 2.0}, x_range=(2.0, 12.0)),
        expression="a/(x-b)-c/x**2",
        parameters=_P_VAN_DER_WAALS,
        initial_guess=bespoke_guess(_P_VAN_DER_WAALS, _van_der_waals),
        domain_check=domain_check(x_positive_for="the volume"),
        references=(_PROPERTIES,),
        assumptions=(
            "The volume is positive and larger than the excluded volume; the gas is "
            "described by one mole of a van der Waals gas."
        ),
    ),
    ModelSpec(
        id="physics_coulomb_field",
        name="Coulomb inverse-square field",
        category="physics",
        description="Strength of the electric (or gravitational) field of a point source.",
        example=ModelExample({"a": 100.0}, x_range=(1.0, 10.0)),
        expression="a/x**2",
        parameters=_P_COULOMB,
        initial_guess=bespoke_guess(_P_COULOMB, _coulomb),
        domain_check=domain_check(x_positive_for="the distance"),
        references=(_TREATISE,),
        assumptions=(
            "The distance is positive; the field of a point source falls with the "
            "square of the distance."
        ),
    ),
    ModelSpec(
        id="physics_lorentz_oscillator",
        name="Lorentz oscillator permittivity",
        category="physics",
        description="Permittivity of a material with a single resonance in the measured range.",
        example=ModelExample({"a": 2.0, "b": 300.0, "c": 12.0}, x_range=(2.0, 10.0)),
        expression="a+b/(c**2-x**2)",
        parameters=_P_LORENTZ_OSCILLATOR,
        initial_guess=bespoke_guess(_P_LORENTZ_OSCILLATOR, _lorentz_oscillator),
        domain_check=domain_check(x_positive_for="the frequency"),
        references=(_ELECTRODYNAMICS,),
        assumptions=(
            "The frequency is positive and the resonance lies outside the measured "
            "range, so the permittivity stays finite."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # exp(a - b/(c + x)) is a*exp(-b/(x + c)) with the prefactor on a logarithmic scale.
    ("physics_antoine", "schumacher"),
    # a*exp(-b/x) is exp(a + b/x) with b of the other sign; the generated default keeps
    # both signs, the physics model only the activated one.
    ("physics_arrhenius", "tc_exp_inv"),
    # a*(x - c)**2 is (a' + b'*x)**2 with the vertex of the parabola named directly.
    ("physics_moseley_law", "tc_sqr_x"),
    # a/(1 + (x - c)**2/b**2) is 1/(a0 + a1*x + a2*x**2) with a complex-conjugate pair
    # of poles; the generated default spans the real-pole cases too.
    ("physics_lorentzian_resonance", "invpoly_2"),
)
"""No two physics models of this family describe the same curve.

The pairs declared above name models of other families that are the same curve on
purpose: the Antoine equation and the Schumacher growth curve are one family with
the prefactor scaled or logged, and the Moseley, Arrhenius and Lorentzian laws are
the physics names of curves the generated families also ship as bare algebra. Each
pair is declared so the duplicate hunt accepts it and Auto-fit can tie the twins.

The Clausius-Clapeyron law is the Antoine equation with ``c = 0`` and Newton's
law of cooling is the offset exponential decay of the core starter set, so
neither is shipped a second time. The Vogel-Fulcher-Tammann viscosity is the
mirror of the Schumacher curve (``exp(-b/(x+c))`` there, ``exp(b/(x-c))`` here);
the bounds keep the two apart, one rising to its plateau and the other diverging
at the Vogel temperature, so no fit can turn either into the other.
"""
