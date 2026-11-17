"""Independent NumPy reference implementations of the saturation family.

Every callable is written from the mathematical definition of the named model rather than
from the expression string of ``curvestudio_models.families.saturation``, so a swapped
parameter, a wrong power or a sign slip in the expression is caught by
``test_catalogue_golden``. ``DENOMINATORS`` holds the denominator of every model written as
a ratio, with the parameters of its example.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _hill(x: FloatArray, *, vmax: float, k: float, n: float) -> FloatArray:
    power_n = np.power(x, n)
    return np.asarray(vmax * power_n / (k**n + power_n))


def _hill_offset(x: FloatArray, *, c: float, vmax: float, k: float, n: float) -> FloatArray:
    power_n = np.power(x, n)
    return np.asarray(c + vmax * power_n / (k**n + power_n))


def _exponential_saturation(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a * (1.0 - np.exp(-b * x)))


def _exponential_saturation_lag(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * (1.0 - np.exp(-b * np.maximum(x - c, 0.0))))


def _freundlich(x: FloatArray, *, k: float, n: float) -> FloatArray:
    return np.asarray(k * np.power(x, 1.0 / n))


def _toth(x: FloatArray, *, a: float, b: float, n: float) -> FloatArray:
    return np.asarray(a * x / np.power(b**n + np.power(x, n), 1.0 / n))


def _redlich_peterson(x: FloatArray, *, k: float, a: float, g: float) -> FloatArray:
    return np.asarray(k * x / (1.0 + a * np.power(x, g)))


def _holling_ii(x: FloatArray, *, attack: float, handling: float) -> FloatArray:
    return np.asarray(attack * x / (1.0 + attack * handling * x))


def _holling_iii(x: FloatArray, *, vmax: float, k: float) -> FloatArray:
    return np.asarray(vmax * np.square(x) / (k * k + np.square(x)))


def _haldane(x: FloatArray, *, vmax: float, km: float, ki: float) -> FloatArray:
    return np.asarray(vmax * x / (km + x + x * x / ki))


def _ricker(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a * x * np.exp(-b * x))


def _ricker_offset(x: FloatArray, *, c: float, a: float, b: float) -> FloatArray:
    return np.asarray(c + a * x * np.exp(-b * x))


def _rectangular_hyperbola_offset(x: FloatArray, *, c: float, a: float, b: float) -> FloatArray:
    return np.asarray(c + a * x / (b + x))


def _exponential_saturation_double(
    x: FloatArray, *, a: float, b: float, r: float, d: float
) -> FloatArray:
    return np.asarray(a * (1.0 - np.exp(-r * x)) + b * (1.0 - np.exp(-(r + d) * x)))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "hill": _hill,
    "hill_offset": _hill_offset,
    "exponential_saturation": _exponential_saturation,
    "exponential_saturation_lag": _exponential_saturation_lag,
    "freundlich": _freundlich,
    "toth": _toth,
    "redlich_peterson": _redlich_peterson,
    "holling_ii": _holling_ii,
    "holling_iii": _holling_iii,
    "haldane": _haldane,
    "ricker": _ricker,
    "ricker_offset": _ricker_offset,
    "rectangular_hyperbola_offset": _rectangular_hyperbola_offset,
    "exponential_saturation_double": _exponential_saturation_double,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "hill": lambda x, *, vmax, k, n: np.asarray(k**n + np.power(x, n)),
    "hill_offset": lambda x, *, c, vmax, k, n: np.asarray(k**n + np.power(x, n)),
    "toth": lambda x, *, a, b, n: np.asarray(np.power(b**n + np.power(x, n), 1.0 / n)),
    "redlich_peterson": lambda x, *, k, a, g: np.asarray(1.0 + a * np.power(x, g)),
    "holling_ii": lambda x, *, attack, handling: np.asarray(1.0 + attack * handling * x),
    "holling_iii": lambda x, *, vmax, k: np.asarray(k * k + np.square(x)),
    "haldane": lambda x, *, vmax, km, ki: np.asarray(km + x + x * x / ki),
    "rectangular_hyperbola_offset": lambda x, *, c, a, b: np.asarray(b + x),
}
