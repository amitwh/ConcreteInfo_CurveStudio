"""Kinetics models: reaction time courses, rate laws and surface-reaction forms.

Every model here is fitted on the original response scale. Where a model can be
linearised, the initial guess uses that linearisation, because it is exact for
noiseless data and a sound start for noisy data: a rate that is reciprocal in the
substrate concentration, a film thickness that grows as the square root of time, a
conversion that is exponential in time.

The models are *not* the Michaelis-Menten curve again: the catalogue already ships it
(as ``michaelis_menten``), and a second model with the same curve is forbidden. The
inhibition and saturation forms that would reproduce it are therefore absent; what is
here is the chemistry that has a shape of its own.

Two parameters that can be exchanged (the two rates of a sum of exponentials, the two
Langmuir constants of a surface reaction) use an ordered parametrisation, so the
parameters are told apart and the fit is well posed.
"""

import math

import numpy as np

from curvestudio.models import ModelExample, ModelSpec, Parameter
from curvestudio_models.families._kit import (
    FloatArray,
    bespoke_guess,
    clean,
    domain_check,
    golden_minimum,
    lstsq,
    pattern_minimum,
    scale,
    scan_refine,
    span,
)

# References are cited only where the work treats the model. Author, year and title
# are given; a chapter or page is added only where it is certain.
_LEVENSPIEL = "Levenspiel O. (1962) Chemical Reaction Kinetics, Wiley"
_FOGLER = "Fogler H.S. (1999) Elements of Chemical Reaction Engineering, 3rd ed., Prentice Hall"
_LANGMUIR = (
    "Langmuir I. (1918) The adsorption of gases on plane surfaces of glass, mica and "
    "platinum, J. Am. Chem. Soc. 40:1361-1403"
)
_HINSHELWOOD = "Hinshelwood C.N. (1926) The Kinetics of Chemical Change, Oxford University Press"
_BATEMAN = (
    "Bateman H. (1910) The solution of a system of differential equations occurring in the "
    "theory of radioactive transformations, Proc. Cambridge Philos. Soc. 15:423-427"
)
_DEAL_GROVE = (
    "Deal B.E., Grove A.S. (1965) General relationship for the thermal oxidation of silicon, "
    "J. Appl. Phys. 36:3770-3778"
)
_CABRERA_MOTT = (
    "Cabrera N., Mott N.F. (1949) Theory of oxidation of metals, Rep. Prog. Phys. 12:163-184"
)
_BUTLER = "Butler J.A.V. (1932) Trans. Faraday Soc. 28:379"
_BARD = "Bard A.J., Faulkner L.R. (2001) Electrochemical Methods, 2nd ed., Wiley"
_CRANK = "Crank J. (1975) The Mathematics of Diffusion, 2nd ed., Oxford University Press"
_BIRD = "Bird R.B., Stewart W.E., Lightfoot E.N. (1960) Transport Phenomena, Wiley"
_RUTHVEN = "Ruthven D.M. (1984) Principles of Adsorption and Adsorption Processes, Wiley"
_MICHAELIS_MENTEN = "Michaelis L., Menten M.L. (1913) Biochem. Z. 49:333-369"
_HALDANE = "Haldane J.B.S. (1930) Enzymes, Longmans, Green & Co., London"

_TINY = 1e-12
"""Smallest allowed gap between the two rates of the Bateman pair (a zero gap is 0/0)."""


# -- small helpers ----------------------------------------------------------------------


def _fallback(values: dict[str, float], defaults: dict[str, float]) -> dict[str, float]:
    """``values`` with every missing, non-finite or non-positive entry replaced.

    A sign or an order can come out of a linearisation with the wrong sign (a
    negative rate, a negative constant); such a value is not usable, so the
    family's own default takes its place.
    """
    out: dict[str, float] = {}
    for name, default in defaults.items():
        value = values.get(name, default)
        out[name] = float(value) if math.isfinite(value) else default
    return out


def _positive_root_sum(value: float) -> float:
    """``value`` when it is positive and finite, otherwise 1."""
    return value if math.isfinite(value) and value > 0.0 else 1.0


def _linear_guess(x: FloatArray, y: FloatArray, columns: list[FloatArray]) -> FloatArray | None:
    """Least-squares coefficients of ``y`` on ``columns``; ``None`` when not determined."""
    return lstsq(columns, y, np.ones(y.size))


def _sse(curve: FloatArray, y: FloatArray) -> float:
    """Sum of squared residuals; not finite when the curve left the domain."""
    with np.errstate(all="ignore"):
        residual = np.asarray(curve, dtype=np.float64) - y
        total = float(np.sum(residual**2))
    return total if math.isfinite(total) else math.inf


# -- kinetics: conversions ---------------------------------------------------------------


def _first_order_conversion(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = xinf*(1 - exp(-k*x))``.

    For a given ``xinf`` the model is linear in ``ln(1 - y/xinf) = -k*x``, so the rate
    follows from a straight line through the origin; ``xinf`` itself is found by a
    one-dimensional search on the residual sum of squares.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    if xs.size < 3:
        return {"xinf": _positive_root_sum(scale(ys)), "k": 1.0 / extent}
    top = float(np.max(ys))
    if top <= 0.0:
        return {"xinf": 1.0, "k": 1.0 / extent}

    def rate_for(xinf: float) -> float:
        remainder = 1.0 - ys / xinf
        keep = remainder > 0.0
        if int(np.count_nonzero(keep)) < 2:
            return math.nan
        with np.errstate(all="ignore"):
            slope = float(np.sum(xs[keep] * np.log(remainder[keep])) / np.sum(xs[keep] ** 2))
        return -slope

    def error(xinf: float) -> float:
        rate = rate_for(xinf)
        if not math.isfinite(rate) or rate <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            return _sse(xinf * (1.0 - np.exp(-rate * xs)), ys)

    best_xinf, _ = scan_refine(error, top * (1.0 + 1e-6), max(top * 20.0, 2.0))
    rate = rate_for(best_xinf)
    if not math.isfinite(rate) or rate <= 0.0:
        rate = 1.0 / extent
    return {"xinf": best_xinf, "k": rate}


def _second_order_conversion(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = x/(x + thalf)``: the half-time is the only parameter."""
    xs, ys = clean(x, y)
    extent = span(xs)
    if xs.size < 2:
        return {"thalf": extent}

    def error(thalf: float) -> float:
        with np.errstate(all="ignore"):
            return _sse(xs / (xs + thalf), ys)

    best, _ = scan_refine(error, extent * 1e-6, extent * 1e3)
    return {"thalf": best if best > 0.0 else extent}


def _nth_order_conversion(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = 1 - (1 + c*x)**(1/(1-n))`` with ``c = (2**(n-1) - 1)/thalf``.

    For a fixed order ``n`` the model is linear in nothing but ``c``, which is found
    by a one-dimensional search; the order is then searched over a grid and refined.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    if xs.size < 3:
        return {"thalf": extent, "n": 2.0}

    def curve_for(order: float, coefficient: float) -> FloatArray:
        with np.errstate(all="ignore"):
            return 1.0 - np.power(1.0 + coefficient * xs, 1.0 / (1.0 - order))

    def best_coefficient(order: float) -> tuple[float, float]:
        def error(coefficient: float) -> float:
            if coefficient <= 0.0:
                return math.inf
            return _sse(curve_for(order, coefficient), ys)

        coefficient, value = scan_refine(error, extent * 1e-6, extent * 1e3)
        if coefficient <= 0.0:
            return extent, math.inf
        return coefficient, value

    orders = np.linspace(1.1, 5.0, 25)
    scored = [(best_coefficient(float(order))[1], float(order)) for order in orders]
    value, order = min(scored)
    if not math.isfinite(value):
        return {"thalf": extent, "n": 2.0}
    low = max(1.02, order - 0.2)
    high = min(20.0, order + 0.2)

    def order_error(candidate: float) -> float:
        return best_coefficient(candidate)[1]

    refined, _ = golden_minimum(order_error, low, high)
    coefficient, _ = best_coefficient(refined)
    half_time = (2.0 ** (refined - 1.0) - 1.0) / coefficient if coefficient > 0.0 else extent
    return {"thalf": half_time if half_time > 0.0 else extent, "n": refined}


# -- kinetics: rate laws -----------------------------------------------------------------


def _haldane(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = vmax*x/(km + x + x**2/ki)``.

    Reciprocating gives ``1/y = (km/vmax)/x + 1/vmax + x/(vmax*ki)``, a straight line
    in ``1/x``, ``1`` and ``x``: the three constants follow from one least-squares fit.
    """
    xs, ys = clean(x, y)
    keep = (xs > 0.0) & (ys > 0.0)
    if int(np.count_nonzero(keep)) < 4:
        top = scale(ys)
        return {"vmax": 1.1 * top, "km": span(xs), "ki": 10.0 * span(xs)}
    coef = _linear_guess(
        xs[keep], 1.0 / ys[keep], [1.0 / xs[keep], np.ones(int(keep.sum())), xs[keep]]
    )
    if coef is None or coef[1] <= 0.0:
        top = scale(ys)
        return {"vmax": 1.1 * top, "km": span(xs), "ki": 10.0 * span(xs)}
    vmax = 1.0 / coef[1]
    km = coef[0] / coef[1]
    ki = 1.0 / (vmax * coef[2]) if coef[2] != 0.0 else 10.0 * span(xs)
    return _fallback(
        {"vmax": vmax, "km": km, "ki": ki},
        {"vmax": 1.1 * scale(ys), "km": span(xs), "ki": 10.0 * span(xs)},
    )


def _two_rate_surface(x: FloatArray, y: FloatArray, power: int) -> dict[str, float]:
    """Guess for ``a*x**power/((1+b*x)*(1+(b+d)*x))``.

    The two Langmuir constants are searched for; when that fails (too few points,
    a curve the search cannot bracket) the reciprocal linearisation is used instead:
    ``1/y = (1/a)[x**-power + (b+c)/x + b*c]`` gives ``a``, the sum and the product
    of the two constants, and the constants are the roots of a quadratic.
    """
    xs, ys = clean(x, y)
    defaults = {"a": scale(ys), "b": 1.0 / span(xs), "d": 0.0}
    scale_value, low, high = _surface_constants(xs, ys, power)
    if math.isfinite(scale_value) and scale_value > 0.0 and low > 0.0:
        return {"a": scale_value, "b": low, "d": max(0.0, high - low)}
    keep = (xs > 0.0) & (ys > 0.0)
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    if power == 2:
        columns = [1.0 / xs[keep] ** 2, 1.0 / xs[keep], np.ones(int(keep.sum()))]
    else:
        columns = [1.0 / xs[keep], np.ones(int(keep.sum())), xs[keep]]
    coef = _linear_guess(xs[keep], 1.0 / ys[keep], columns)
    if coef is None or coef[0] <= 0.0:
        return defaults
    a = 1.0 / coef[0]
    total = coef[1] / coef[0]
    product = coef[2] / coef[0]
    discriminant = total * total - 4.0 * product
    if discriminant < 0.0:
        return defaults
    root = math.sqrt(discriminant)
    low = 0.5 * (total - root)
    high = 0.5 * (total + root)
    if low <= 0.0:
        return defaults
    return _fallback({"a": a, "b": low, "d": high - low}, defaults)


def _surface_constants(x: FloatArray, y: FloatArray, power: int) -> tuple[float, float, float]:
    """``(a, b, c)`` of ``a*x**power/((1+b*x)*(1+c*x))`` with ``b <= c``.

    The curve is linear in ``a``, so the two Langmuir constants are searched over a
    grid; the scale follows from the line. Searching on the curve itself (rather
    than on its reciprocal) keeps the small-concentration points from taking over.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    keep = xs > 0.0
    if int(np.count_nonzero(keep)) < 4:
        return math.nan, math.nan, math.nan
    xs, ys = xs[keep], ys[keep]

    def error(point: FloatArray) -> float:
        low, high = float(point[0]), float(point[1])
        if low <= 0.0 or high <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = xs**power / ((1.0 + low * xs) * (1.0 + high * xs))
        coef = _linear_guess(xs, ys, [basis])
        if coef is None:
            return math.inf
        return _sse(coef[0] * basis, ys)

    grid = np.geomspace(0.005 / extent, 20.0 / extent, 18)
    scored: list[tuple[float, FloatArray]] = []
    for low in grid:
        for high in grid:
            value = error(np.array([low, high]))
            if math.isfinite(value):
                scored.append((value, np.array([low, high])))
    if not scored:
        return math.nan, math.nan, math.nan
    scored.sort(key=lambda item: item[0])
    best_value, best_point = math.inf, scored[0][1]
    for _, start in scored[:3]:
        point, value = pattern_minimum(error, start, 0.5 / extent)
        point, value = pattern_minimum(error, point, 0.05 / extent)
        if value < best_value:
            best_value, best_point = value, point
    point = best_point
    smallest, largest = sorted((float(point[0]), float(point[1])))
    with np.errstate(all="ignore"):
        basis = xs**power / ((1.0 + smallest * xs) * (1.0 + largest * xs))
    coef = _linear_guess(xs, ys, [basis])
    scale_value = float(coef[0]) if coef is not None else math.nan
    return scale_value, smallest, largest


def _dissociative_adsorption(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*sqrt(x)/(1 + b*sqrt(x))``.

    Reciprocating gives ``1/y = (1/a)(1/sqrt(x) + b)``, a straight line in
    ``1/sqrt(x)`` and a constant.
    """
    xs, ys = clean(x, y)
    defaults = {"a": scale(ys), "b": 1.0}
    keep = (xs > 0.0) & (ys > 0.0)
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    with np.errstate(all="ignore"):
        column = 1.0 / np.sqrt(xs[keep])
    coef = _linear_guess(xs[keep], 1.0 / ys[keep], [column, np.ones(int(np.count_nonzero(keep)))])
    if coef is None or coef[0] <= 0.0:
        return defaults
    return _fallback({"a": 1.0 / coef[0], "b": coef[1] / coef[0]}, defaults)


def _bateman(x: FloatArray, y: FloatArray, product: bool) -> dict[str, float]:
    """Guess for the Bateman pair ``A -> B -> C``.

    The curve is linear in the initial concentration, so only the two rates are
    searched: a coarse grid locates the basin and a pattern search refines it.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a0": scale(ys), "k1": 1.0 / extent, "g": 1.0 / extent}
    if xs.size < 4:
        return defaults

    def curve(rate1: float, gap: float) -> FloatArray:
        rate2 = rate1 + gap
        with np.errstate(all="ignore"):
            if product:
                return 1.0 - (rate2 * np.exp(-rate1 * xs) - rate1 * np.exp(-rate2 * xs)) / gap
            return rate1 * (np.exp(-rate1 * xs) - np.exp(-rate2 * xs)) / gap

    def error(point: FloatArray) -> float:
        rate1, gap = float(point[0]), float(point[1])
        if rate1 <= 0.0 or gap <= 0.0:
            return math.inf
        basis = curve(rate1, gap)
        coef = _linear_guess(xs, ys, [basis])
        if coef is None:
            return math.inf
        return _sse(coef[0] * basis, ys)

    rates = np.linspace(0.05, 5.0, 30) / extent
    scored: list[tuple[float, FloatArray]] = []
    for rate1 in rates:
        for gap in rates:
            if gap <= 0.0:
                continue
            value = error(np.array([rate1, gap]))
            if math.isfinite(value):
                scored.append((value, np.array([rate1, gap])))
    if not scored:
        return defaults
    scored.sort(key=lambda item: item[0])
    best_value, best_point = math.inf, scored[0][1]
    for _, start in scored[:3]:
        point, value = pattern_minimum(error, start, 0.25 / extent)
        point, value = pattern_minimum(error, point, 0.05 / extent)
        if value < best_value:
            best_value, best_point = value, point
    point = best_point
    basis = curve(float(point[0]), float(point[1]))
    coef = _linear_guess(xs, ys, [basis])
    a0 = float(coef[0]) if coef is not None else defaults["a0"]
    if not math.isfinite(a0) or a0 <= 0.0:
        a0 = defaults["a0"]
    return _fallback({"a0": a0, "k1": float(point[0]), "g": float(point[1])}, defaults)


def _power_exponential(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*x**b*exp(-c*x)``: straight line in ``ln y`` against ``1``, ``ln x`` and ``x``."""
    xs, ys = clean(x, y)
    keep = (xs > 0.0) & (ys > 0.0)
    defaults = {"a": scale(ys), "b": 1.0, "c": 1.0 / span(xs)}
    if int(np.count_nonzero(keep)) < 4:
        return defaults
    coef = _linear_guess(
        xs[keep], np.log(ys[keep]), [np.ones(int(keep.sum())), np.log(xs[keep]), xs[keep]]
    )
    if coef is None:
        return defaults
    return _fallback({"a": math.exp(min(coef[0], 700.0)), "b": coef[1], "c": -coef[2]}, defaults)


def _butler_volmer(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = j0*(exp(a*x) - exp(-b*x))``.

    On each side of the origin one exponential dominates, so the two branches give
    the rates by a straight line in ``ln|y|``; the two rates are then refined
    together with the exchange current as the only linear parameter.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"j0": scale(ys), "a": 1.0 / extent, "b": 1.0 / extent}
    if xs.size < 5:
        return defaults
    positive = (xs > 0.0) & (ys > 0.0)
    negative = (xs < 0.0) & (ys < 0.0)
    rates: list[float] = []
    currents: list[float] = []
    for mask, sign in ((positive, 1.0), (negative, -1.0)):
        if int(np.count_nonzero(mask)) < 2:
            continue
        coef = _linear_guess(
            xs[mask], np.log(sign * ys[mask]), [np.ones(int(mask.sum())), xs[mask]]
        )
        if coef is None:
            continue
        rates.append(abs(coef[1]))
        currents.append(math.exp(min(coef[0], 700.0)))
    if not rates:
        return defaults
    a = rates[0]
    b = rates[1] if len(rates) > 1 else rates[0]
    j0 = float(np.sqrt(np.prod(currents))) if currents else defaults["j0"]

    def error(point: FloatArray) -> float:
        rate_a, rate_b = float(point[0]), float(point[1])
        if rate_a <= 0.0 or rate_b <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = np.exp(rate_a * xs) - np.exp(-rate_b * xs)
        coef = _linear_guess(xs, ys, [basis])
        if coef is None:
            return math.inf
        return _sse(coef[0] * basis, ys)

    point, _ = pattern_minimum(error, np.array([a, b]), 0.25 / extent)
    point, _ = pattern_minimum(error, point, 0.05 / extent)
    with np.errstate(all="ignore"):
        basis = np.exp(float(point[0]) * xs) - np.exp(-float(point[1]) * xs)
    coef = _linear_guess(xs, ys, [basis])
    current = float(coef[0]) if coef is not None else j0
    return _fallback(
        {"j0": current, "a": float(point[0]), "b": float(point[1])},
        {"j0": j0, "a": a, "b": b},
    )


def _two_exponential_decay(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*exp(-b*x) + c*exp(-(b+g)*x)``.

    The curve is linear in the two amplitudes, so the two rates are searched over a
    grid and refined; the grid also fixes the order of the two rates.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": 0.5 * scale(ys), "b": 1.0 / extent, "c": 0.5 * scale(ys), "g": 1.0 / extent}
    if xs.size < 5:
        return defaults

    def error(point: FloatArray) -> float:
        rate1, gap = float(point[0]), float(point[1])
        if rate1 <= 0.0 or gap <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = [np.exp(-rate1 * xs), np.exp(-(rate1 + gap) * xs)]
        coef = _linear_guess(xs, ys, basis)
        if coef is None:
            return math.inf
        return _sse(coef[0] * basis[0] + coef[1] * basis[1], ys)

    rates = np.linspace(0.05, 4.0, 26) / extent
    scored: list[tuple[float, FloatArray]] = []
    for rate1 in rates:
        for gap in rates:
            if gap <= 0.0:
                continue
            value = error(np.array([rate1, gap]))
            if math.isfinite(value):
                scored.append((value, np.array([rate1, gap])))
    if not scored:
        return defaults
    scored.sort(key=lambda item: item[0])
    best_value, best_point = math.inf, scored[0][1]
    for _, start in scored[:3]:
        point, value = pattern_minimum(error, start, 0.25 / extent)
        point, value = pattern_minimum(error, point, 0.05 / extent)
        if value < best_value:
            best_value, best_point = value, point
    point = best_point
    with np.errstate(all="ignore"):
        basis = [np.exp(-float(point[0]) * xs), np.exp(-float(point[1]) * xs)]
    coef = _linear_guess(xs, ys, basis)
    if coef is None:
        return defaults
    return _fallback(
        {
            "a": float(coef[0]),
            "b": float(point[0]),
            "c": float(coef[1]),
            "g": float(point[1]) - float(point[0]),
        },
        defaults,
    )


# -- kinetics: films and diffusion -------------------------------------------------------


def _parabolic_oxidation(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = sqrt(2*b*(x+tau))``: the squared film thickness is a straight line in time."""
    xs, ys = clean(x, y)
    defaults = {"b": 1.0 / max(span(xs), 1e-12) ** 2, "tau": 0.0}
    if xs.size < 3:
        return defaults
    coef = _linear_guess(xs, ys**2, [xs, np.ones(xs.size)])
    if coef is None or coef[0] <= 0.0:
        return defaults
    b = coef[0] / 2.0
    tau = coef[1] / (2.0 * b)
    return _fallback({"b": b, "tau": tau}, defaults)


def _deal_grove(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = (-A + sqrt(A**2 + 4*B*(x+tau)))/2``.

    For a fixed ``A`` the squared relation is linear in the two constants, so ``A``
    is searched over a grid and the pair is read off the line.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"A": 0.0, "B": scale(ys) ** 2 / extent, "tau": 0.0}
    if xs.size < 4:
        return defaults

    def constants_for(constant: float) -> tuple[float, float, float]:
        squared = (2.0 * ys + constant) ** 2
        coef = _linear_guess(xs, squared, [xs, np.ones(xs.size)])
        if coef is None or coef[0] <= 0.0:
            return math.nan, math.nan, math.inf
        rate = coef[0] / 4.0
        tau = (coef[1] - constant**2) / (4.0 * rate)
        with np.errstate(all="ignore"):
            curve = (-constant + np.sqrt(constant**2 + 4.0 * rate * (xs + tau))) / 2.0
        return rate, tau, _sse(curve, ys)

    grid = np.linspace(0.0, 4.0 * scale(ys) + 1e-9, 21)
    scored = [(constants_for(float(value))[2], float(value)) for value in grid]
    value, constant = min(scored)
    if not math.isfinite(value):
        return defaults
    refined, _ = golden_minimum(
        lambda candidate: constants_for(candidate)[2],
        max(0.0, constant - 0.2 * (scale(ys) + 1e-9)),
        constant + 0.2 * (scale(ys) + 1e-9),
    )
    rate, tau, _ = constants_for(refined)
    if not math.isfinite(rate):
        return defaults
    return _fallback({"A": refined, "B": rate, "tau": tau}, defaults)


def _logarithmic_oxidation(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*log(1 + b*x)``: linear in ``a`` for a fixed ``b``; ``b`` is searched."""
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": 1.0 / max(extent, 1e-12)}
    if xs.size < 3:
        return defaults

    def line(rate: float) -> tuple[float, float]:
        with np.errstate(all="ignore"):
            column = np.log1p(rate * xs)
        coef = _linear_guess(xs, ys, [column])
        if coef is None or coef[0] <= 0.0:
            return math.nan, math.inf
        with np.errstate(all="ignore"):
            return float(coef[0]), _sse(coef[0] * column, ys)

    def error(rate: float) -> float:
        if rate <= 0.0:
            return math.inf
        return line(rate)[1]

    best, _ = scan_refine(error, 1e-6 / max(extent, 1e-12), 1e3 / max(extent, 1e-12), count=80)
    amplitude, _ = line(best)
    if not math.isfinite(amplitude):
        return defaults
    return _fallback({"a": amplitude, "b": best}, defaults)


def _error_function_diffusion(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a - b*erf(c*x)``: a diffusion front.

    For a fixed ``c`` the model is straight in ``a`` and ``b``, so ``c`` is searched.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": scale(ys), "c": 1.0 / extent}
    if xs.size < 4:
        return defaults

    def error(rate: float) -> float:
        if rate <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = [-np.vectorize(_erf)(rate * xs), np.ones(xs.size)]
        coef = _linear_guess(xs, ys, basis)
        if coef is None:
            return math.inf
        return _sse(coef[0] * basis[0] + coef[1], ys)

    best, _ = scan_refine(error, 1e-3 / extent, 1e3 / extent)
    with np.errstate(all="ignore"):
        basis = [-np.vectorize(_erf)(best * xs), np.ones(xs.size)]
    coef = _linear_guess(xs, ys, basis)
    if coef is None:
        return defaults
    return _fallback({"a": coef[1], "b": coef[0], "c": best}, defaults)


def _erf(value: float) -> float:
    """The error function of a real number."""
    return math.erf(value)


# -- the models --------------------------------------------------------------------------


_P_FIRST_ORDER = (
    Parameter(
        "xinf",
        "Conversion reached as time goes to infinity (the plateau of the curve)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("k", "First-order rate constant (per unit time)", lower=0.0, upper=math.inf),
)
_P_SECOND_ORDER = (
    Parameter(
        "thalf",
        "Time at which half of the conversion is reached",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_NTH_ORDER = (
    Parameter(
        "thalf",
        "Time at which half of the conversion is reached",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "n",
        "Reaction order (greater than 1; the first order is the limit n = 1)",
        lower=1.02,
        upper=20.0,
    ),
)
_P_HALDANE = (
    Parameter(
        "vmax",
        "Scale of the rate (the curve peaks at vmax*sqrt(km*ki)/(2*km+sqrt(km*ki)) on "
        "x = sqrt(km*ki))",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "km",
        "Michaelis constant: the substrate concentration of the half-maximum rate",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "ki",
        "Substrate inhibition constant: the rate turns over at x = sqrt(km*ki) and falls "
        "towards zero as the substrate accumulates",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_LH = (
    Parameter(
        "a",
        "Scale of the rate (the rate per unit product of the two coverages)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Smaller of the two Langmuir constants (adsorption constant of one reactant)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "d",
        "Gap to the larger Langmuir constant (the constants are ordered b <= b + d)",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_DISSOCIATIVE = (
    Parameter(
        "a",
        "Scale of the rate (the rate per unit pressure of the adsorbed molecule)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("b", "Adsorption constant of the dissociative reaction", lower=0.0, upper=math.inf),
)
_P_BATEMAN = (
    Parameter(
        "a0",
        "Initial concentration of the reactant that decays first",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("k1", "Rate constant of the first step (the slower one)", lower=0.0, upper=math.inf),
    Parameter(
        "g",
        "Gap to the rate constant of the second step (the rates are ordered k1 <= k1 + g)",
        lower=_TINY,
        upper=math.inf,
    ),
)
_P_PARABOLIC = (
    Parameter("b", "Parabolic rate constant of the film growth", lower=0.0, upper=math.inf),
    Parameter("tau", "Incubation time before the film starts to grow", lower=0.0, upper=math.inf),
)
_P_DEAL_GROVE = (
    Parameter(
        "A",
        "Linear rate constant of the Deal-Grove relation (the interface reaction term)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter("B", "Parabolic rate constant of the film growth", lower=0.0, upper=math.inf),
    Parameter("tau", "Incubation time before the film starts to grow", lower=0.0, upper=math.inf),
)
_P_LOG_OXIDATION = (
    Parameter(
        "a",
        "Thickness scale of the logarithmic law",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Rate constant of the logarithmic law: the reciprocal of its time constant",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_POWER_EXPONENTIAL = (
    Parameter(
        "a",
        "Scale of the rate (the rate at x = 1 without the exponential factor)",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Power of the substrate concentration in the rate",
        lower=-math.inf,
        upper=math.inf,
    ),
    Parameter("c", "Decay rate of the exponential factor", lower=0.0, upper=math.inf),
)
_P_BUTLER_VOLMER = (
    Parameter(
        "j0",
        "Exchange current density at the reference overpotential",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "a",
        "Anodic transfer coefficient divided by the thermal voltage",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "b",
        "Cathodic transfer coefficient divided by the thermal voltage",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_DIFFUSION_FRONT = (
    Parameter("a", "Value of y at the surface (x = 0)", lower=-math.inf, upper=math.inf),
    Parameter(
        "b",
        "Depth of the concentration drop across the front",
        lower=0.0,
        upper=math.inf,
    ),
    Parameter(
        "c",
        "Inverse width of the front (the diffusion length is 1/c)",
        lower=0.0,
        upper=math.inf,
    ),
)

_P_TWO_EXPONENTIAL = (
    Parameter("a", "Amplitude of the slower exponential component", lower=0.0, upper=math.inf),
    Parameter("b", "Rate of the slower exponential component", lower=0.0, upper=math.inf),
    Parameter("c", "Amplitude of the faster exponential component", lower=0.0, upper=math.inf),
    Parameter(
        "g",
        "Gap between the two rates (the rates are ordered b <= b + g)",
        lower=0.0,
        upper=math.inf,
    ),
)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="kinetics_first_order_conversion",
        name="First-order conversion",
        category="kinetics",
        description=(
            "Fraction of a reactant consumed by a first-order reaction: it rises to a plateau."
        ),
        example=ModelExample({"xinf": 1.0, "k": 0.35}, x_range=(0.0, 12.0)),
        expression="xinf*(1-exp(-k*x))",
        parameters=_P_FIRST_ORDER,
        initial_guess=bespoke_guess(_P_FIRST_ORDER, _first_order_conversion),
        domain_check=domain_check(x_non_negative=True),
        references=(_LEVENSPIEL, _FOGLER),
        assumptions=(
            "Time runs from zero and the reaction is first order in the reactant; the "
            "conversion approaches xinf asymptotically and never exceeds it."
        ),
    ),
    ModelSpec(
        id="kinetics_second_order_conversion",
        name="Second-order conversion",
        category="kinetics",
        description=(
            "Fraction of a reactant consumed by a second-order reaction, given by its half-time."
        ),
        example=ModelExample({"thalf": 3.0}, x_range=(0.0, 20.0)),
        expression="x/(x+thalf)",
        parameters=_P_SECOND_ORDER,
        initial_guess=bespoke_guess(_P_SECOND_ORDER, _second_order_conversion),
        domain_check=domain_check(x_non_negative=True),
        references=(_LEVENSPIEL,),
        assumptions=(
            "Time runs from zero and the reaction is second order with equal initial "
            "concentrations; the conversion is complete only in the limit of infinite time."
        ),
    ),
    ModelSpec(
        id="kinetics_nth_order_conversion",
        name="N-th order conversion",
        category="kinetics",
        description=(
            "Fraction of a reactant consumed by an n-th order reaction, given by its half-time."
        ),
        example=ModelExample({"thalf": 4.0, "n": 2.5}, x_range=(0.0, 20.0)),
        expression="1-(1+(2**(n-1)-1)*x/thalf)**(1/(1-n))",
        parameters=_P_NTH_ORDER,
        initial_guess=bespoke_guess(_P_NTH_ORDER, _nth_order_conversion),
        domain_check=domain_check(x_non_negative=True),
        references=(_LEVENSPIEL, _FOGLER),
        assumptions=(
            "The reaction order is greater than one; the first-order case is the limit "
            "n = 1 and is shipped separately."
        ),
    ),
    ModelSpec(
        id="kinetics_haldane_substrate_inhibition",
        name="Haldane rate law (substrate inhibition)",
        category="kinetics",
        description=(
            "Rate of a reaction that is inhibited by its own substrate: it rises, turns over and "
            "falls."
        ),
        example=ModelExample({"vmax": 10.0, "km": 2.0, "ki": 50.0}, x_range=(0.5, 30.0)),
        expression="vmax*x/(km+x+x**2/ki)",
        parameters=_P_HALDANE,
        initial_guess=bespoke_guess(_P_HALDANE, _haldane),
        domain_check=domain_check(x_positive_for="the rate law"),
        references=(_HALDANE, _MICHAELIS_MENTEN, _LEVENSPIEL),
        assumptions=(
            "The substrate concentration is positive; the rate reaches its maximum at "
            "x = sqrt(km*ki) and falls again when the substrate inhibits the reaction."
        ),
    ),
    ModelSpec(
        id="kinetics_langmuir_hinshelwood_bimolecular",
        name="Langmuir-Hinshelwood bimolecular surface reaction",
        category="kinetics",
        description=(
            "Rate of a bimolecular surface reaction: two adsorbed reactants meet on the surface."
        ),
        example=ModelExample({"a": 20.0, "b": 0.2, "d": 0.4}, x_range=(0.1, 20.0)),
        expression="a*x**2/((1+b*x)*(1+(b+d)*x))",
        parameters=_P_LH,
        initial_guess=bespoke_guess(_P_LH, lambda x, y: _two_rate_surface(x, y, 2)),
        domain_check=domain_check(x_positive_for="the surface coverage"),
        references=(_LANGMUIR, _HINSHELWOOD),
        assumptions=(
            "Both reactants compete for the same sites and the surface reaction needs one "
            "molecule of each; the rate rises to a maximum and falls again."
        ),
    ),
    ModelSpec(
        id="kinetics_dissociative_adsorption",
        name="Dissociative adsorption rate",
        category="kinetics",
        description="Rate of a dissociative adsorption: the adsorbed molecule covers two sites.",
        example=ModelExample({"a": 4.0, "b": 0.5}, x_range=(0.25, 16.0)),
        expression="a*sqrt(x)/(1+b*sqrt(x))",
        parameters=_P_DISSOCIATIVE,
        initial_guess=bespoke_guess(_P_DISSOCIATIVE, _dissociative_adsorption),
        domain_check=domain_check(x_positive_for="the pressure"),
        references=(_RUTHVEN, _LANGMUIR),
        assumptions=(
            "The pressure is positive and the adsorption is dissociative, so the coverage "
            "follows the square root of the pressure; the rate saturates with the coverage."
        ),
    ),
    ModelSpec(
        id="kinetics_bateman_intermediate",
        name="Bateman intermediate (A to B to C)",
        category="kinetics",
        description=(
            "Concentration of the intermediate of a consecutive reaction: it rises and then decays."
        ),
        example=ModelExample({"a0": 1.0, "k1": 0.2, "g": 0.6}, x_range=(0.0, 20.0)),
        expression="a0*k1/g*(exp(-k1*x)-exp(-(k1+g)*x))",
        parameters=_P_BATEMAN,
        initial_guess=bespoke_guess(_P_BATEMAN, lambda x, y: _bateman(x, y, False)),
        domain_check=domain_check(x_non_negative=True),
        references=(_BATEMAN, _LEVENSPIEL),
        assumptions=(
            "A decays into B with the rate k1 and B decays into C with the rate k1 + g; "
            "the intermediate is never produced directly."
        ),
    ),
    ModelSpec(
        id="kinetics_bateman_product",
        name="Bateman product (A to B to C)",
        category="kinetics",
        description=(
            "Concentration of the final product of a consecutive first-order reaction: it rises "
            "towards a plateau."
        ),
        example=ModelExample({"a0": 1.0, "k1": 0.3, "g": 0.9}, x_range=(0.0, 30.0)),
        expression="a0*((k1+g)*(1-exp(-k1*x))-k1*(1-exp(-(k1+g)*x)))/g",
        parameters=_P_BATEMAN,
        initial_guess=bespoke_guess(_P_BATEMAN, lambda x, y: _bateman(x, y, True)),
        domain_check=domain_check(x_non_negative=True),
        references=(_BATEMAN, _LEVENSPIEL),
        assumptions=(
            "A decays into B with the rate k1 and B decays into C with the rate k1 + g; "
            "the product is formed only through the intermediate."
        ),
    ),
    ModelSpec(
        id="kinetics_parabolic_oxidation",
        name="Parabolic film growth",
        category="kinetics",
        description=(
            "Thickness of an oxide film that grows parabolically: the square of the thickness is "
            "linear in time."
        ),
        example=ModelExample({"b": 0.02, "tau": 1.0}, x_range=(0.0, 50.0)),
        expression="sqrt(2*b*(x+tau))",
        parameters=_P_PARABOLIC,
        initial_guess=bespoke_guess(_P_PARABOLIC, _parabolic_oxidation),
        domain_check=domain_check(x_non_negative=True),
        references=(_DEAL_GROVE,),
        assumptions=(
            "The film grows parabolically because the diffusion through it is the slow "
            "step; an incubation time tau may precede the growth."
        ),
    ),
    ModelSpec(
        id="kinetics_deal_grove_oxidation",
        name="Deal-Grove oxidation",
        category="kinetics",
        description=(
            "Thickness of an oxide film that follows the Deal-Grove relation x**2 + A*x = "
            "B*(t + tau): linear growth at short times that turns parabolic."
        ),
        example=ModelExample({"A": 2.0, "B": 0.05, "tau": 1.0}, x_range=(0.0, 40.0)),
        expression="(-A+sqrt(A**2+4*B*(x+tau)))/2",
        parameters=_P_DEAL_GROVE,
        initial_guess=bespoke_guess(_P_DEAL_GROVE, _deal_grove),
        domain_check=domain_check(x_non_negative=True),
        references=(_DEAL_GROVE,),
        assumptions=(
            "The film follows the Deal-Grove relation x**2 + A*x = B*(t + tau), so the growth "
            "is linear while the interface term dominates and parabolic at long times; an "
            "incubation time tau may precede the growth. (The relation was shipped under the "
            "id kinetics_cabrera_mott_oxidation before it was correctly attributed.)"
        ),
    ),
    ModelSpec(
        id="kinetics_cabrera_mott_oxidation",
        name="Cabrera-Mott oxidation (logarithmic)",
        category="kinetics",
        description=(
            "Thickness of a thin oxide film that grows logarithmically with time, as the "
            "Cabrera-Mott high-field mechanism predicts."
        ),
        example=ModelExample({"a": 5.0, "b": 0.5}, x_range=(0.0, 40.0)),
        expression="a*log(1+b*x)",
        parameters=_P_LOG_OXIDATION,
        initial_guess=bespoke_guess(_P_LOG_OXIDATION, _logarithmic_oxidation),
        domain_check=domain_check(x_non_negative=True),
        references=(_CABRERA_MOTT,),
        assumptions=(
            "The electric field across the thin film drives the ions and decays as the film "
            "thickens, so the growth follows the logarithmic law a*log(1 + b*t); the film is "
            "thin enough for the high-field mechanism to hold."
        ),
    ),
    ModelSpec(
        id="kinetics_power_exponential",
        name="Power-law rate with exponential decay",
        category="kinetics",
        description=(
            "Rate that grows with a power of the substrate and is damped by an exponential factor."
        ),
        example=ModelExample({"a": 2.0, "b": 0.5, "c": 0.3}, x_range=(0.5, 20.0)),
        expression="a*x**b*exp(-c*x)",
        parameters=_P_POWER_EXPONENTIAL,
        initial_guess=bespoke_guess(_P_POWER_EXPONENTIAL, _power_exponential),
        domain_check=domain_check(x_positive_for="the power of the substrate"),
        references=(_LEVENSPIEL, _FOGLER),
        assumptions=(
            "The substrate concentration is positive and the rate is proportional to a "
            "power of it, damped by an exponential factor that falls with concentration."
        ),
    ),
    ModelSpec(
        id="kinetics_butler_volmer",
        name="Butler-Volmer electrode kinetics",
        category="kinetics",
        description=(
            "Current of an electrode reaction: an anodic and a cathodic branch meet at the origin."
        ),
        example=ModelExample({"j0": 0.5, "a": 1.2, "b": 0.8}, x_range=(-3.0, 3.0)),
        expression="j0*(exp(a*x)-exp(-b*x))",
        parameters=_P_BUTLER_VOLMER,
        initial_guess=bespoke_guess(_P_BUTLER_VOLMER, _butler_volmer),
        domain_check=domain_check(),
        references=(_BUTLER, _BARD),
        assumptions=(
            "The overpotential is measured from the equilibrium potential and the current "
            "is the difference of an anodic and a cathodic partial current."
        ),
    ),
    ModelSpec(
        id="kinetics_two_exponential_decay",
        name="Two-exponential decay",
        category="kinetics",
        description=(
            "Quantity that relaxes with two exponential rates, as a sum of two first-order decays."
        ),
        example=ModelExample({"a": 3.0, "b": 0.4, "c": 1.5, "g": 0.8}, x_range=(0.0, 15.0)),
        expression="a*exp(-b*x)+c*exp(-(b+g)*x)",
        parameters=_P_TWO_EXPONENTIAL,
        initial_guess=bespoke_guess(_P_TWO_EXPONENTIAL, _two_exponential_decay),
        domain_check=domain_check(x_non_negative=True),
        references=(_FOGLER, _LEVENSPIEL),
        assumptions=(
            "Two first-order processes act at once; the rates are ordered so that the "
            "two components cannot be exchanged."
        ),
    ),
    ModelSpec(
        id="kinetics_error_function_diffusion",
        name="Diffusion front (error-function profile)",
        category="kinetics",
        description=(
            "Concentration profile of a diffusion front: it falls from the surface value along an "
            "error function."
        ),
        example=ModelExample({"a": 1.0, "b": 0.8, "c": 0.12}, x_range=(0.0, 12.0)),
        expression="a-b*erf(c*x)",
        parameters=_P_DIFFUSION_FRONT,
        initial_guess=bespoke_guess(_P_DIFFUSION_FRONT, _error_function_diffusion),
        domain_check=domain_check(x_non_negative=True),
        references=(_CRANK, _BIRD),
        assumptions=(
            "The diffusion is semi-infinite and the front is driven by a constant "
            "concentration at the surface."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # 1/y = km/(vmax*x) + 1/vmax + x/(vmax*ki): the Haldane rate is the generated
    # 1/(a + b*x + c/x) with the three constants read off the enzyme parameters.
    ("kinetics_haldane_substrate_inhibition", "tc3_inv_x_inv"),
    # a0*(1 - (k2*exp(-k1*x) - k1*exp(-k2*x))/(k2 - k1)) is the Chapman-Richards
    # a*(1 - b*exp(-c*x))**d curve: the product of the Bateman pair is its exact
    # reparametrisation with d tied to the rate ratio.
    ("kinetics_bateman_product", "chapman_richards"),
    # a*x**b*exp(-c*x) is a*exp(b*x)*x**c with the exponent sign folded into the rate.
    ("kinetics_power_exponential", "exponential_power_growth"),
    # sqrt(2*b*(x + tau)) is sqrt(a + b*x) with a = 2*b*tau.
    ("kinetics_parabolic_oxidation", "tc_sqrtof_x"),
    # a*exp(-b*x) + c*exp(-(b+g)*x) is the generated sum of two decaying exponentials
    # with the rates ordered by construction.
    ("kinetics_two_exponential_decay", "exp_sum_2"),
)
"""No two kinetics models of this family describe the same curve.

The pairs declared above name models of other families that are the same curve on
purpose (the generated algebraic defaults and the growth family's
Chapman-Richards curve, which the Bateman product's plateau reproduces exactly);
each is declared so the duplicate hunt accepts it and Auto-fit can tie the twins.

The Michaelis-Menten curve is deliberately absent: it is in the core starter set and
in the rational family, and the inhibition forms that reproduce it are the same curve
with another meaning, which the catalogue does not ship twice.

The first-order conversion is the two-parameter nested case of the core
``exp_offset``/``exp_decay_offset`` pair (its plateau plays the role of the offset);
the equivalence harness compares models with the same number of parameters only, so
the nesting cannot be declared there and is recorded here instead.
"""
