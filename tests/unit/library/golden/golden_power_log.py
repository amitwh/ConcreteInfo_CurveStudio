"""Independent NumPy reference implementations of the power and logarithmic family.

Written from the mathematical definition of each model (generalised and shifted power laws,
the modified power with an exponential cut-off, sums of two power laws, and the logarithmic
variants) rather than from the expression strings.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _generalised_power(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a + b * np.power(x, c))


def _shifted_power(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.power(x + c, b))


def _modified_power(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.power(x, b) * np.exp(-c * x))


def _modified_power_offset(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(d + a * np.power(x, b) * np.exp(-c * x))


def _power_sum_double(x: FloatArray, *, a: float, b: float, c: float, e: float) -> FloatArray:
    return np.asarray(a * np.power(x, b) + c * np.power(x, b + e))


def _log_one_plus(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a * np.log1p(b * x))


def _log_shifted(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a + b * np.log(x + c))


def _log_quadratic(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    u = np.log(x)
    return np.asarray(a + b * u + c * np.square(u))


def _log_cubic(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    u = np.log(x)
    return np.asarray(a + b * u + c * np.square(u) + d * u * np.square(u))


def _linear_log(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a + b * x + c * np.log(x))


def _square_root_offset(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a * np.sqrt(x) + b)


def _log_quadratic_shifted(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    u = np.log(x + c)
    return np.asarray(a + b * u + d * np.square(u))


def _log_of_linear(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(np.log(a + b * x))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "generalised_power": _generalised_power,
    "shifted_power": _shifted_power,
    "modified_power": _modified_power,
    "modified_power_offset": _modified_power_offset,
    "power_sum_double": _power_sum_double,
    "log_one_plus": _log_one_plus,
    "log_shifted": _log_shifted,
    "log_quadratic": _log_quadratic,
    "log_cubic": _log_cubic,
    "linear_log": _linear_log,
    "square_root_offset": _square_root_offset,
    "log_quadratic_shifted": _log_quadratic_shifted,
    "log_of_linear": _log_of_linear,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {}
