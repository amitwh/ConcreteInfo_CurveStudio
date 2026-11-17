"""Harmonic models: oscillations, spectra and the shapes built on a sine.

The models here are the ones that a signal, a spectrum or a periodic measurement
shows: a sine on a baseline, the first harmonic with a free frequency, an
oscillation that decays, two oscillations whose frequencies are told apart by
their order, and the sinc kernel.

The initial guesses come from the structure of the model rather than from a blind
search: for a fixed frequency a harmonic model is a straight line (in the
coefficients of the sine and the cosine), so the frequency is the only thing to
search for, and the amplitudes and the phase follow from the line.
"""

import math
from collections.abc import Callable

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

# References are cited only where the work treats the model.
_HARMONIC_ANALYSIS = "Brillouin L. (1946) Wave Propagation in Periodic Structures, McGraw-Hill"
_FOURIER = "Fourier J.B.J. (1822) Théorie analytique de la chaleur"
_SINC = (
    "Whittaker E.T., Watson G.N. (1913) A Course of Modern Analysis, 2nd ed., "
    "Cambridge University Press"
)
_DAMPED = "Rayleigh J.W.S. (1877) The Theory of Sound, Macmillan"
_RIEMANN = (
    "Riemann B. (1854) Ueber die Darstellbarkeit einer Function durch eine trigonometrische Reihe"
)


# -- small helpers ----------------------------------------------------------------------


def _fit(columns: list[FloatArray], target: FloatArray) -> FloatArray | None:
    """Least-squares coefficients of ``target`` on ``columns``; ``None`` when not determined."""
    return lstsq(columns, target, np.ones(target.size, dtype=np.float64))


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


def _frequency_grid(extent: float, low: float, high: float, count: int) -> FloatArray:
    """Candidate frequencies (per unit x) for a search."""
    return np.geomspace(low / extent, high / extent, count)


def _zero_crossings(x: FloatArray, y: FloatArray) -> FloatArray:
    """The x values where ``y`` changes sign, by linear interpolation."""
    if x.size < 2:
        return np.array([])
    sign_change = np.nonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0.0)[0]
    out: list[float] = []
    for index in sign_change:
        y0, y1 = float(y[index]), float(y[index + 1])
        if y0 == y1:
            continue
        t = -y0 / (y1 - y0)
        out.append(float(x[index] + t * (x[index + 1] - x[index])))
    return np.asarray(out, dtype=np.float64)


def _tangent_zero(x: FloatArray, y: FloatArray) -> float:
    """The first zero of a curve that touches zero without changing sign.

    The kernel is squared, so its zeros are local minima; the minimum is located
    and then refined by the parabola through the three points around it.
    """
    if x.size < 3:
        return math.nan
    interior = np.arange(1, x.size - 1)
    if interior.size == 0:
        return math.nan
    minima = [
        index
        for index in interior
        if y[index] <= y[index - 1] and y[index] <= y[index + 1] and y[index] >= 0.0
    ]
    if not minima:
        return math.nan
    index = int(minima[0])
    left, middle, right = y[index - 1], y[index], y[index + 1]
    denominator = left - 2.0 * middle + right
    if denominator == 0.0:
        return float(x[index])
    shift = 0.5 * (left - right) / denominator
    step = float(x[index + 1] - x[index])
    return float(x[index] + shift * step)


def _refine_frequency(
    error: Callable[[FloatArray], float], point: FloatArray, index: int
) -> FloatArray:
    """Refine one frequency of a two-parameter search around its current value."""
    value = float(point[index])
    if value <= 0.0:
        return point

    def objective(candidate: float) -> float:
        trial = point.copy()
        trial[index] = candidate
        return error(trial)

    best, _ = scan_refine(objective, value / 1.4, value * 1.4, count=40)
    trial = point.copy()
    trial[index] = best
    return trial


def _two_shape_search(
    error: Callable[[FloatArray], float],
    first: FloatArray,
    second: FloatArray,
    *,
    step: float,
) -> FloatArray:
    """Best point of a two-parameter search: a grid over both, then a pattern search."""
    point, _ = best_of_grid(error, first, second)
    point, _ = pattern_minimum(error, point, step)
    return pattern_minimum(error, point, step * 0.2)[0]


# -- guesses ---------------------------------------------------------------------------


def _sine_offset(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b*sin(c*x)``: for a fixed frequency the model is a straight line."""
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": float(np.mean(ys)) if ys.size else 0.0, "b": scale(ys), "c": 1.0 / extent}

    def error(frequency: float) -> float:
        if frequency <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = np.sin(frequency * xs)
        coef = _fit([np.ones(xs.size), basis], ys)
        if coef is None:
            return math.inf
        return _sse(coef[0] + coef[1] * basis, ys)

    best, _ = scan_refine(error, 1e-2 / extent, 50.0 / extent, count=200)
    with np.errstate(all="ignore"):
        basis = np.sin(best * xs)
    coef = _fit([np.ones(xs.size), basis], ys)
    if coef is None:
        return defaults
    return _fallback({"a": coef[0], "b": coef[1], "c": best}, defaults)


def _fourier_first(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b*cos(c*x + d)``: for a fixed frequency the model is a straight line."""
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {
        "a": float(np.mean(ys)) if ys.size else 0.0,
        "b": scale(ys),
        "c": 1.0 / extent,
        "d": 0.0,
    }

    def error(frequency: float) -> float:
        if frequency <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = [np.cos(frequency * xs), np.sin(frequency * xs)]
        coef = _fit([np.ones(xs.size), *basis], ys)
        if coef is None:
            return math.inf
        return _sse(coef[0] + coef[1] * basis[0] + coef[2] * basis[1], ys)

    best, _ = scan_refine(error, 1e-2 / extent, 50.0 / extent, count=200)
    with np.errstate(all="ignore"):
        basis = [np.cos(best * xs), np.sin(best * xs)]
    coef = _fit([np.ones(xs.size), *basis], ys)
    if coef is None:
        return defaults
    amplitude = math.hypot(coef[1], coef[2])
    phase = math.atan2(-coef[2], coef[1])
    return _fallback({"a": coef[0], "b": amplitude, "c": best, "d": phase}, defaults)


def _harmonic_with_drift(x: FloatArray, y: FloatArray, quadratic: bool) -> dict[str, float]:
    """``y = a + b*x (+ c*x**2) + d*sin(f*x)``.

    For a fixed frequency the model is a straight line in the trend and the sine,
    so the frequency is the only thing to search for.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {
        "a": float(np.mean(ys)) if ys.size else 0.0,
        "b": 0.0,
        "c": scale(ys),
        "d": 1.0 / extent,
    }
    if quadratic:
        defaults["c"] = 0.0
        defaults["d"] = scale(ys)
        defaults["e"] = 1.0 / extent
    if xs.size < 4:
        return defaults

    def error(frequency: float) -> float:
        if frequency <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            columns = [np.ones(xs.size), xs]
            if quadratic:
                columns.append(xs**2)
            columns.append(np.sin(frequency * xs))
            coef = _fit(columns, ys)
        if coef is None:
            return math.inf
        with np.errstate(all="ignore"):
            curve = sum(c * col for c, col in zip(coef, columns, strict=True))
        return _sse(curve, ys)

    best, _ = scan_refine(error, 1e-2 / extent, 50.0 / extent, count=200)
    with np.errstate(all="ignore"):
        columns = [np.ones(xs.size), xs]
        if quadratic:
            columns.append(xs**2)
        columns.append(np.sin(best * xs))
    coef = _fit(columns, ys)
    if coef is None:
        return defaults
    if quadratic:
        values = {
            "a": float(coef[0]),
            "b": float(coef[1]),
            "c": float(coef[2]),
            "d": float(coef[3]),
            "e": best,
        }
    else:
        values = {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "d": best}
    return _fallback(values, defaults)


def _damped_sine(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b*exp(-d*x)*sin(c*x + e)``.

    The frequency and the decay rate are searched together; for both of them the
    model is a straight line in the baseline, the sine and the cosine.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {
        "a": float(np.mean(ys)) if ys.size else 0.0,
        "b": scale(ys),
        "c": 1.0 / extent,
        "d": 0.0,
        "e": 0.0,
    }
    if xs.size < 5:
        return defaults

    def error(point: FloatArray) -> float:
        frequency, decay = float(point[0]), float(point[1])
        if frequency <= 0.0 or decay < 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            envelope = np.exp(-decay * xs)
            columns = [
                np.ones(xs.size),
                envelope * np.sin(frequency * xs),
                envelope * np.cos(frequency * xs),
            ]
        coef = _fit(columns, ys)
        if coef is None:
            return math.inf
        with np.errstate(all="ignore"):
            curve = sum(c * col for c, col in zip(coef, columns, strict=True))
        return _sse(curve, ys)

    frequencies = _frequency_grid(extent, 1e-2, 50.0, 60)
    decays = np.linspace(0.0, 20.0 / extent, 16)
    point = _two_shape_search(error, frequencies, decays, step=0.5 / extent)
    with np.errstate(all="ignore"):
        envelope = np.exp(-float(point[1]) * xs)
        columns = [
            np.ones(xs.size),
            envelope * np.sin(float(point[0]) * xs),
            envelope * np.cos(float(point[0]) * xs),
        ]
    coef = _fit(columns, ys)
    if coef is None:
        return defaults
    amplitude = math.hypot(coef[1], coef[2])
    phase = math.atan2(coef[2], coef[1])
    return _fallback(
        {
            "a": float(coef[0]),
            "b": amplitude,
            "c": float(point[0]),
            "d": float(point[1]),
            "e": phase,
        },
        defaults,
    )


def _two_sine(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b*sin(c*x) + d*sin((c+g)*x)``.

    The two frequencies are searched together (the amplitudes and the baseline
    follow from a straight line); the gap ``g`` keeps them in order.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {
        "a": float(np.mean(ys)) if ys.size else 0.0,
        "b": scale(ys),
        "c": 1.0 / extent,
        "d": scale(ys),
        "g": 1.0 / extent,
    }
    if xs.size < 6:
        return defaults

    def error(point: FloatArray) -> float:
        first, second = float(point[0]), float(point[1])
        if first <= 0.0 or second <= first:
            return math.inf
        with np.errstate(all="ignore"):
            columns = [
                np.ones(xs.size),
                np.sin(first * xs),
                np.sin(second * xs),
            ]
        coef = _fit(columns, ys)
        if coef is None:
            return math.inf
        with np.errstate(all="ignore"):
            curve = sum(c * col for c, col in zip(coef, columns, strict=True))
        return _sse(curve, ys)

    frequencies = _frequency_grid(extent, 1e-2, 50.0, 50)
    point = _two_shape_search(error, frequencies, frequencies, step=0.5 / extent)
    point = _refine_frequency(error, point, 0)
    point = _refine_frequency(error, point, 1)
    first, second = sorted((float(point[0]), float(point[1])))
    with np.errstate(all="ignore"):
        columns = [np.ones(xs.size), np.sin(first * xs), np.sin(second * xs)]
    coef = _fit(columns, ys)
    if coef is None:
        return defaults
    return _fallback(
        {
            "a": float(coef[0]),
            "b": float(coef[1]),
            "c": first,
            "d": float(coef[2]),
            "g": second - first,
        },
        defaults,
    )


def _sinc(x: FloatArray, y: FloatArray, squared: bool) -> dict[str, float]:
    """``y = a*sin(b*x)/(b*x)`` (or its square).

    The first zero of the kernel gives the scale ``b``; the amplitude then
    follows from a straight line through the origin.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {"a": scale(ys), "b": math.pi / extent}
    if squared:
        first_zero = _tangent_zero(xs, ys)
    else:
        crossings = _zero_crossings(xs, ys)
        first_zero = float(crossings[0]) if crossings.size else math.nan
    if not math.isfinite(first_zero) or first_zero <= 0.0:
        return defaults
    frequency = math.pi / first_zero
    with np.errstate(all="ignore"):
        argument = frequency * xs
        kernel = np.sin(argument) / argument
    keep = np.abs(kernel) > 1e-6
    if int(np.count_nonzero(keep)) < 3:
        return defaults
    with np.errstate(all="ignore"):
        shape = kernel[keep] ** 2 if squared else kernel[keep]
        amplitude = float(np.sum(shape * ys[keep]) / np.sum(shape**2))
    return _fallback({"a": amplitude, "b": frequency}, defaults)


def _gaussian_modulated_sine(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a*exp(-b*(x-c)**2)*cos(d*(x-c))``.

    The envelope of the wave packet is recovered from the signal and its
    logarithm (a quadratic in x) gives the width and the centre; the carrier
    frequency follows from the zero crossings.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {
        "a": scale(ys),
        "b": 1.0 / extent**2,
        "c": float(np.median(xs)) if xs.size else 0.0,
        "d": 1.0 / extent,
    }
    if xs.size < 6:
        return defaults

    crossings = _zero_crossings(xs, ys)
    if crossings.size < 2:
        return defaults
    spacing = float(np.mean(np.diff(crossings)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        return defaults
    frequency = math.pi / spacing if spacing > 0.0 else 1.0 / extent
    with np.errstate(all="ignore"):
        slope = np.gradient(ys, xs)
        envelope = np.sqrt(ys**2 + (slope / frequency) ** 2)
    good = envelope > 0.0
    if int(np.count_nonzero(good)) < 4:
        return defaults
    with np.errstate(all="ignore"):
        coef = _fit(
            [np.ones(int(np.count_nonzero(good))), xs[good], xs[good] ** 2], np.log(envelope[good])
        )
    if coef is None or coef[2] >= 0.0:
        return defaults
    # ln(envelope) = ln(a) - b*(x - c)**2, so the quadratic gives b, c and a.
    width = -coef[2]
    centre = -coef[1] / (2.0 * coef[2])
    amplitude = math.exp(min(coef[0] - coef[1] ** 2 / (4.0 * coef[2]), 700.0))

    def error(candidate: float) -> float:
        if candidate <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            basis = np.exp(-width * (xs - centre) ** 2) * np.cos(candidate * (xs - centre))
        coef = _fit([basis], ys)
        if coef is None:
            return math.inf
        return _sse(coef[0] * basis, ys)

    best, _ = scan_refine(error, 0.9 * frequency, 1.1 * frequency, count=40)
    return _fallback({"a": amplitude, "b": width, "c": centre, "d": best}, defaults)


def _exponential_trend_sine(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b*exp(-c*x) + d*sin(e*x)``.

    For a fixed decay rate and a fixed frequency the model is a straight line in
    the baseline, the trend and the sine, so the two shape parameters are searched
    together.
    """
    xs, ys = clean(x, y)
    extent = span(xs)
    defaults = {
        "a": float(np.mean(ys)) if ys.size else 0.0,
        "b": scale(ys),
        "c": 1.0 / extent,
        "d": scale(ys),
        "e": 1.0 / extent,
    }
    if xs.size < 6:
        return defaults

    def error(point: FloatArray) -> float:
        decay, frequency = float(point[0]), float(point[1])
        if decay < 0.0 or frequency <= 0.0:
            return math.inf
        with np.errstate(all="ignore"):
            columns = [
                np.ones(xs.size),
                np.exp(-decay * xs),
                np.sin(frequency * xs),
            ]
        coef = _fit(columns, ys)
        if coef is None:
            return math.inf
        with np.errstate(all="ignore"):
            curve = sum(c * col for c, col in zip(coef, columns, strict=True))
        return _sse(curve, ys)

    decays = np.linspace(0.0, 20.0 / extent, 16)
    frequencies = _frequency_grid(extent, 1e-2, 20.0, 40)
    point = _two_shape_search(error, decays, frequencies, step=0.5 / extent)
    with np.errstate(all="ignore"):
        columns = [
            np.ones(xs.size),
            np.exp(-float(point[0]) * xs),
            np.sin(float(point[1]) * xs),
        ]
    coef = _fit(columns, ys)
    if coef is None:
        return defaults
    return _fallback(
        {
            "a": float(coef[0]),
            "b": float(coef[1]),
            "c": float(point[0]),
            "d": float(coef[2]),
            "e": float(point[1]),
        },
        defaults,
    )


# -- the models --------------------------------------------------------------------------


_P_SINE_OFFSET = (
    Parameter("a", "Baseline that y oscillates about", lower=-math.inf, upper=math.inf),
    Parameter("b", "Amplitude of the oscillation", lower=-math.inf, upper=math.inf),
    Parameter("c", "Angular frequency of the oscillation", lower=0.0, upper=math.inf),
)
_P_FOURIER_FIRST = (
    Parameter("a", "Baseline that y oscillates about", lower=-math.inf, upper=math.inf),
    Parameter("b", "Amplitude of the first harmonic", lower=0.0, upper=math.inf),
    Parameter("c", "Angular frequency of the first harmonic", lower=0.0, upper=math.inf),
    Parameter("d", "Phase of the first harmonic (radians)", lower=-math.inf, upper=math.inf),
)
_P_DAMPED_SINE = (
    Parameter("a", "Value that y settles towards", lower=-math.inf, upper=math.inf),
    Parameter("b", "Amplitude of the oscillation at x = 0", lower=0.0, upper=math.inf),
    Parameter("c", "Angular frequency of the oscillation", lower=0.0, upper=math.inf),
    Parameter("d", "Decay rate of the oscillation", lower=0.0, upper=math.inf),
    Parameter("e", "Phase of the oscillation (radians)", lower=-math.inf, upper=math.inf),
)
_P_SINE_DRIFT = (
    Parameter("a", "Value of y at x = 0", lower=-math.inf, upper=math.inf),
    Parameter("b", "Slope of the linear trend", lower=-math.inf, upper=math.inf),
    Parameter("c", "Amplitude of the oscillation", lower=-math.inf, upper=math.inf),
    Parameter("d", "Angular frequency of the oscillation", lower=0.0, upper=math.inf),
)
_P_SINE_QUADRATIC_DRIFT = (
    Parameter("a", "Value of y at x = 0", lower=-math.inf, upper=math.inf),
    Parameter("b", "Slope of the linear trend", lower=-math.inf, upper=math.inf),
    Parameter("c", "Curvature of the trend", lower=-math.inf, upper=math.inf),
    Parameter("d", "Amplitude of the oscillation", lower=-math.inf, upper=math.inf),
    Parameter("e", "Angular frequency of the oscillation", lower=0.0, upper=math.inf),
)
_P_TWO_SINE = (
    Parameter("a", "Baseline that y oscillates about", lower=-math.inf, upper=math.inf),
    Parameter("b", "Amplitude of the slower oscillation", lower=-math.inf, upper=math.inf),
    Parameter("c", "Angular frequency of the slower oscillation", lower=0.0, upper=math.inf),
    Parameter("d", "Amplitude of the faster oscillation", lower=-math.inf, upper=math.inf),
    Parameter(
        "g",
        "Gap to the faster frequency (the frequencies are ordered c <= c + g)",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_SINC = (
    Parameter("a", "Height of the central lobe", lower=-math.inf, upper=math.inf),
    Parameter(
        "b",
        "Angular frequency of the kernel (the first zero is at pi/b)",
        lower=0.0,
        upper=math.inf,
    ),
)
_P_WAVE_PACKET = (
    Parameter("a", "Amplitude of the wave packet", lower=0.0, upper=math.inf),
    Parameter("b", "Inverse of the square of the width of the envelope", lower=0.0, upper=math.inf),
    Parameter("c", "Centre of the wave packet", lower=-math.inf, upper=math.inf),
    Parameter("d", "Angular frequency of the carrier", lower=0.0, upper=math.inf),
)
_P_TREND_SINE = (
    Parameter("a", "Value that y tends towards", lower=-math.inf, upper=math.inf),
    Parameter("b", "Height of the decaying trend above that value", lower=0.0, upper=math.inf),
    Parameter("c", "Decay rate of the trend", lower=0.0, upper=math.inf),
    Parameter("d", "Amplitude of the oscillation", lower=-math.inf, upper=math.inf),
    Parameter("e", "Angular frequency of the oscillation", lower=0.0, upper=math.inf),
)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="harmonic_sine_offset",
        name="Sine with an offset",
        category="harmonic",
        description="A sine wave on a constant baseline.",
        example=ModelExample({"a": 1.0, "b": 2.0, "c": 1.5}, x_range=(0.0, 8.0)),
        expression="a+b*sin(c*x)",
        parameters=_P_SINE_OFFSET,
        initial_guess=bespoke_guess(_P_SINE_OFFSET, _sine_offset),
        domain_check=domain_check(),
        references=(_HARMONIC_ANALYSIS,),
        assumptions=(
            "The oscillation starts at its zero crossing (no phase parameter); the "
            "frequency is found from the data, so the range must show whole periods."
        ),
    ),
    ModelSpec(
        id="harmonic_fourier_first_free_frequency",
        name="First harmonic with a free frequency",
        category="harmonic",
        description=(
            "The first harmonic of a Fourier series: a baseline, a cosine and a sine of one "
            "frequency."
        ),
        example=ModelExample({"a": 1.0, "b": 2.0, "c": 1.5, "d": 0.6}, x_range=(0.0, 8.0)),
        expression="a+b*cos(c*x+d)",
        parameters=_P_FOURIER_FIRST,
        initial_guess=bespoke_guess(_P_FOURIER_FIRST, _fourier_first),
        domain_check=domain_check(),
        references=(_FOURIER, _HARMONIC_ANALYSIS),
        assumptions=(
            "One frequency only; the phase is free, so the curve may start anywhere in "
            "its cycle. The frequency is found from the data."
        ),
    ),
    ModelSpec(
        id="harmonic_damped_sine",
        name="Damped sine",
        category="harmonic",
        description="An oscillation whose amplitude decays exponentially towards a constant.",
        example=ModelExample(
            {"a": 0.0, "b": 3.0, "c": 2.0, "d": 0.35, "e": 0.4}, x_range=(0.0, 10.0)
        ),
        expression="a+b*exp(-d*x)*sin(c*x+e)",
        parameters=_P_DAMPED_SINE,
        initial_guess=bespoke_guess(_P_DAMPED_SINE, _damped_sine),
        domain_check=domain_check(),
        references=(_DAMPED, _HARMONIC_ANALYSIS),
        assumptions=(
            "The envelope decays with a single rate and the oscillation keeps its "
            "frequency; the data must cover several periods of the decaying part."
        ),
    ),
    ModelSpec(
        id="harmonic_sine_with_drift",
        name="Sine with a linear drift",
        category="harmonic",
        description="An oscillation on a straight line that rises (or falls) through it.",
        example=ModelExample({"a": 0.5, "b": 0.3, "c": 2.0, "d": 1.5}, x_range=(0.0, 8.0)),
        expression="a+b*x+c*sin(d*x)",
        parameters=_P_SINE_DRIFT,
        initial_guess=bespoke_guess(_P_SINE_DRIFT, lambda x, y: _harmonic_with_drift(x, y, False)),
        domain_check=domain_check(),
        references=(_RIEMANN, _HARMONIC_ANALYSIS),
        assumptions=(
            "The trend is a straight line and the oscillation starts at its zero "
            "crossing; the range must show whole periods of the oscillation."
        ),
    ),
    ModelSpec(
        id="harmonic_sine_with_quadratic_drift",
        name="Sine with a quadratic drift",
        category="harmonic",
        description="An oscillation on a parabolic trend.",
        example=ModelExample(
            {"a": 0.5, "b": 0.2, "c": 0.05, "d": 2.0, "e": 1.5}, x_range=(0.0, 8.0)
        ),
        expression="a+b*x+c*x**2+d*sin(e*x)",
        parameters=_P_SINE_QUADRATIC_DRIFT,
        initial_guess=bespoke_guess(
            _P_SINE_QUADRATIC_DRIFT, lambda x, y: _harmonic_with_drift(x, y, True)
        ),
        domain_check=domain_check(),
        references=(_RIEMANN, _HARMONIC_ANALYSIS),
        assumptions=(
            "The trend is a parabola and the oscillation starts at its zero crossing; "
            "the range must show whole periods of the oscillation."
        ),
    ),
    ModelSpec(
        id="harmonic_two_sine_ordered",
        name="Two sine waves (ordered)",
        category="harmonic",
        description="Two oscillations of different frequency on a common baseline.",
        example=ModelExample(
            {"a": 0.0, "b": 1.5, "c": 1.0, "d": 0.8, "g": 1.5}, x_range=(0.0, 12.0)
        ),
        expression="a+b*sin(c*x)+d*sin((c+g)*x)",
        parameters=_P_TWO_SINE,
        initial_guess=bespoke_guess(_P_TWO_SINE, _two_sine),
        domain_check=domain_check(),
        references=(_RIEMANN, _FOURIER),
        assumptions=(
            "The two frequencies are ordered (the gap g is not negative), so the two "
            "oscillations cannot be exchanged; they must be far enough apart to be told apart."
        ),
    ),
    ModelSpec(
        id="harmonic_sinc",
        name="Sinc kernel",
        category="harmonic",
        description="The sinc kernel: a central lobe with side lobes of decreasing height.",
        example=ModelExample({"a": 1.0, "b": 1.0}, x_range=(0.5, 12.0)),
        expression="a*sin(b*x)/(b*x)",
        parameters=_P_SINC,
        initial_guess=bespoke_guess(_P_SINC, lambda x, y: _sinc(x, y, False)),
        domain_check=domain_check(x_nonzero_for="the sinc kernel"),
        references=(_SINC,),
        assumptions=(
            "The kernel is not defined at x = 0 (its limit there is a), so the range "
            "must not contain zero; the data must show the first zero of the kernel."
        ),
    ),
    ModelSpec(
        id="harmonic_sinc_squared",
        name="Squared sinc kernel",
        category="harmonic",
        description=(
            "The square of the sinc kernel: a central lobe with side lobes of rapidly decreasing "
            "height."
        ),
        example=ModelExample({"a": 1.0, "b": 1.0}, x_range=(0.5, 12.0)),
        expression="a*(sin(b*x)/(b*x))**2",
        parameters=_P_SINC,
        initial_guess=bespoke_guess(_P_SINC, lambda x, y: _sinc(x, y, True)),
        domain_check=domain_check(x_nonzero_for="the sinc kernel"),
        references=(_SINC,),
        assumptions=(
            "The kernel is not defined at x = 0 (its limit there is a), so the range "
            "must not contain zero; the data must show the first zero of the kernel."
        ),
    ),
    ModelSpec(
        id="harmonic_gaussian_modulated_sine",
        name="Wave packet (Gaussian envelope)",
        category="harmonic",
        description="A carrier oscillation whose amplitude follows a Gaussian envelope.",
        example=ModelExample({"a": 2.0, "b": 0.08, "c": 5.0, "d": 3.0}, x_range=(0.0, 10.0)),
        expression="a*exp(-b*(x-c)**2)*cos(d*(x-c))",
        parameters=_P_WAVE_PACKET,
        initial_guess=bespoke_guess(_P_WAVE_PACKET, _gaussian_modulated_sine),
        domain_check=domain_check(),
        references=(_HARMONIC_ANALYSIS,),
        assumptions=(
            "The envelope is Gaussian and the carrier keeps its frequency; the data "
            "must cover the whole packet so that the envelope is visible."
        ),
    ),
    ModelSpec(
        id="harmonic_exponential_trend_sine",
        name="Sine on an exponential trend",
        category="harmonic",
        description="An oscillation on a baseline that approaches a level exponentially.",
        example=ModelExample(
            {"a": 0.0, "b": 3.0, "c": 0.4, "d": 2.0, "e": 1.5}, x_range=(0.0, 10.0)
        ),
        expression="a+b*exp(-c*x)+d*sin(e*x)",
        parameters=_P_TREND_SINE,
        initial_guess=bespoke_guess(_P_TREND_SINE, _exponential_trend_sine),
        domain_check=domain_check(),
        references=(_RIEMANN, _HARMONIC_ANALYSIS),
        assumptions=(
            "The trend decays with a single rate and the oscillation starts at its zero "
            "crossing; the range must show whole periods of the oscillation."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # The first harmonic with a free phase contains the zero-phase sine as the case
    # d = pi/2, so the two describe the same curve family.
    ("harmonic_sine_offset", "harmonic_fourier_first_free_frequency"),
    # a + b*cos(c*x + d) is the generated a0 + a1*cos(w*x) + b1*sin(w*x) with the
    # cosine and the sine read off one phase.
    ("harmonic_fourier_first_free_frequency", "fourier_1"),
)
