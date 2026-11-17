"""Independent NumPy reference implementations of the sigmoidal family.

Each callable is written from the mathematical definition (not from the expression string of
the model) and takes ``x`` and the parameters by keyword. ``erf`` comes from SciPy's special
functions (the same routine ``sympy.lambdify`` resolves the model's ``erf`` to), so the
comparison tests the algebra, not a different implementation of the error function.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

FloatArray = NDArray[np.float64]


def _logistic4(x: FloatArray, *, a: float, k: float, b: float, x0: float) -> FloatArray:
    return np.asarray(a + (k - a) / (1.0 + np.exp(-b * (x - x0))))


def _logistic5(x: FloatArray, *, a: float, k: float, b: float, x0: float, g: float) -> FloatArray:
    return np.asarray(a + (k - a) / np.power(1.0 + np.exp(-b * (x - x0)), g))


def _boltzmann(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(a + b / (1.0 + np.exp(-(x - c) / d)))


def _tanh_sigmoid(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(a + b * np.tanh((x - c) / d))


def _erf_sigmoid(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(a + b * 0.5 * (1.0 + erf((x - c) / d)))


def _arctan_sigmoid(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(a + b * (np.arctan((x - c) / d) / np.pi + 0.5))


def _weibull_cdf3(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * (1.0 - np.exp(-np.power(x / b, c))))


def _logistic_quadratic5(
    x: FloatArray, *, a: float, b: float, c: float, d: float, e: float
) -> FloatArray:
    u = x - d
    return np.asarray(a + b / (1.0 + np.exp(-(c * u + e * u**2))))


def _arctan_drift5(
    x: FloatArray, *, a: float, b: float, c: float, d: float, e: float
) -> FloatArray:
    return np.asarray(a + b * (np.arctan((x - c) / d) / np.pi + 0.5) + e * x)


def _hill_sigmoid(x: FloatArray, *, vmax: float, km: float, n: float) -> FloatArray:
    xn = np.power(x, n)
    return np.asarray(vmax * xn / (np.power(km, n) + xn))


def _algebraic_sigmoid(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a + b * x / np.sqrt(c**2 + x**2))


def _logistic_drift5(
    x: FloatArray, *, a: float, k: float, b: float, x0: float, e: float
) -> FloatArray:
    return np.asarray(a + (k - a) / (1.0 + np.exp(-b * (x - x0))) + e * x)


def _frechet_cdf(x: FloatArray, *, c: float, d: float, e: float, b: float) -> FloatArray:
    return np.asarray(c + (d - c) * np.exp(-np.power(x / e, -b)))


def _erf_drift5(x: FloatArray, *, a: float, b: float, c: float, d: float, e: float) -> FloatArray:
    return np.asarray(a + b * 0.5 * (1.0 + erf((x - c) / d)) + e * x)


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "logistic4": _logistic4,
    "logistic5": _logistic5,
    "boltzmann": _boltzmann,
    "tanh_sigmoid": _tanh_sigmoid,
    "erf_sigmoid": _erf_sigmoid,
    "arctan_sigmoid": _arctan_sigmoid,
    "weibull_cdf3": _weibull_cdf3,
    "logistic_quadratic5": _logistic_quadratic5,
    "arctan_drift5": _arctan_drift5,
    "hill_sigmoid": _hill_sigmoid,
    "algebraic_sigmoid": _algebraic_sigmoid,
    "logistic_drift5": _logistic_drift5,
    "frechet_cdf": _frechet_cdf,
    "erf_drift5": _erf_drift5,
}
