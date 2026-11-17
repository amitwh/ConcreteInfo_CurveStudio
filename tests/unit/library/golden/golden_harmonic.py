"""Independent NumPy reference implementations of the harmonic family.

Each callable is written from the mathematical definition (not from the expression
string of the model) and takes ``x`` and the parameters by keyword.
``test_catalogue_golden`` compares it with the model at the example parameters, so a
sign slip, a wrong power or a swapped parameter in an expression is caught.

The definitions are the usual ones: a sine on a baseline, the first harmonic
``a + b*cos(c*x + d)``, an exponentially damped oscillation, two oscillations whose
frequencies are ordered, and the sinc kernel with its square.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sine_offset(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a + b * np.sin(c * x))


def _fourier_first(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    # the first harmonic of a Fourier series
    return np.asarray(a + b * np.cos(c * x + d))


def _damped_sine(x: FloatArray, *, a: float, b: float, c: float, d: float, e: float) -> FloatArray:
    # an oscillation whose envelope decays with the rate d
    return np.asarray(a + b * np.exp(-d * x) * np.sin(c * x + e))


def _sine_with_drift(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(a + b * x + c * np.sin(d * x))


def _sine_with_quadratic_drift(
    x: FloatArray, *, a: float, b: float, c: float, d: float, e: float
) -> FloatArray:
    return np.asarray(a + b * x + c * x**2 + d * np.sin(e * x))


def _two_sine(x: FloatArray, *, a: float, b: float, c: float, d: float, g: float) -> FloatArray:
    return np.asarray(a + b * np.sin(c * x) + d * np.sin((c + g) * x))


def _sinc(x: FloatArray, *, a: float, b: float) -> FloatArray:
    argument = b * x
    return np.asarray(a * np.sin(argument) / argument)


def _sinc_squared(x: FloatArray, *, a: float, b: float) -> FloatArray:
    argument = b * x
    return np.asarray(a * (np.sin(argument) / argument) ** 2)


def _wave_packet(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    # a carrier oscillation under a Gaussian envelope
    offset = x - c
    return np.asarray(a * np.exp(-b * offset**2) * np.cos(d * offset))


def _exponential_trend_sine(
    x: FloatArray, *, a: float, b: float, c: float, d: float, e: float
) -> FloatArray:
    return np.asarray(a + b * np.exp(-c * x) + d * np.sin(e * x))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "harmonic_sine_offset": _sine_offset,
    "harmonic_fourier_first_free_frequency": _fourier_first,
    "harmonic_damped_sine": _damped_sine,
    "harmonic_sine_with_drift": _sine_with_drift,
    "harmonic_sine_with_quadratic_drift": _sine_with_quadratic_drift,
    "harmonic_two_sine_ordered": _two_sine,
    "harmonic_sinc": _sinc,
    "harmonic_sinc_squared": _sinc_squared,
    "harmonic_gaussian_modulated_sine": _wave_packet,
    "harmonic_exponential_trend_sine": _exponential_trend_sine,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {}
