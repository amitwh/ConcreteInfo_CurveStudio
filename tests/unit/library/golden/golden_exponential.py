"""Independent NumPy reference implementations of the exponential/decay family.

Written from the mathematical definition of each model (ordered multi-exponential sums,
stretched exponential, monomolecular approach, lagged decay and growth, difference of
exponentials) rather than from the expression strings, so ``test_catalogue_golden``
catches a swapped rate, a wrong sign or a missing offset.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _decay(x: FloatArray, *, a: float, r: float) -> FloatArray:
    return np.asarray(a * np.exp(-r * x))


def _decay_double(x: FloatArray, *, a: float, b: float, r: float, d: float) -> FloatArray:
    return np.asarray(a * np.exp(-r * x) + b * np.exp(-(r + d) * x))


def _decay_triple(
    x: FloatArray, *, a: float, b: float, c: float, r: float, d1: float, d2: float
) -> FloatArray:
    return np.asarray(
        a * np.exp(-r * x) + b * np.exp(-(r + d1) * x) + c * np.exp(-(r + d1 + d2) * x)
    )


def _decay_double_offset(
    x: FloatArray, *, a: float, b: float, r: float, d: float, c: float
) -> FloatArray:
    return np.asarray(c + a * np.exp(-r * x) + b * np.exp(-(r + d) * x))


def _decay_lag(x: FloatArray, *, a: float, r: float, c: float) -> FloatArray:
    return np.asarray(a * np.exp(-r * np.maximum(x - c, 0.0)))


def _stretched(x: FloatArray, *, a: float, tau: float, beta: float) -> FloatArray:
    return np.asarray(a * np.exp(-np.power(x / tau, beta)))


def _stretched_offset(x: FloatArray, *, a: float, tau: float, beta: float, c: float) -> FloatArray:
    return np.asarray(c + a * np.exp(-np.power(x / tau, beta)))


def _monomolecular(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * (1.0 - b * np.exp(-c * x)))


def _biexponential_difference(x: FloatArray, *, a: float, b: float, d: float) -> FloatArray:
    return np.asarray(a * (np.exp(-b * x) - np.exp(-(b + d) * x)))


def _biexponential_difference_offset(
    x: FloatArray, *, a: float, b: float, d: float, c: float
) -> FloatArray:
    return np.asarray(c + a * (np.exp(-b * x) - np.exp(-(b + d) * x)))


def _growth_double(x: FloatArray, *, a: float, b: float, r: float, d: float) -> FloatArray:
    return np.asarray(a * np.exp(r * x) + b * np.exp((r + d) * x))


def _growth_double_offset(
    x: FloatArray, *, a: float, b: float, r: float, d: float, c: float
) -> FloatArray:
    return np.asarray(c + a * np.exp(r * x) + b * np.exp((r + d) * x))


def _growth_lag(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.exp(b * np.maximum(x - c, 0.0)))


def _growth_offset(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(c + a * np.exp(b * x))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "exponential_decay": _decay,
    "exponential_decay_double": _decay_double,
    "exponential_decay_triple": _decay_triple,
    "exponential_decay_double_offset": _decay_double_offset,
    "exponential_decay_lag": _decay_lag,
    "stretched_exponential": _stretched,
    "stretched_exponential_offset": _stretched_offset,
    "monomolecular": _monomolecular,
    "biexponential_difference": _biexponential_difference,
    "biexponential_difference_offset": _biexponential_difference_offset,
    "exponential_growth_double": _growth_double,
    "exponential_growth_double_offset": _growth_double_offset,
    "exponential_growth_lag": _growth_lag,
    "exponential_growth_offset": _growth_offset,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {}
