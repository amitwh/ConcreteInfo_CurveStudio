"""Independent NumPy reference implementations of the growth family.

Each callable is written from the mathematical definition (not from the expression string of
the model) and takes ``x`` and the parameters by keyword. ``test_catalogue_golden`` compares it
with the model at the example parameters, so a sign slip, a wrong power or a swapped parameter
in an expression is caught.

``max`` in :func:`_baranyi_lag` is ``numpy.maximum``: the model's lag is a flat zero-growth
phase before the exponential approach to the plateau.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _gompertz(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.exp(-np.exp(-b * (x - c))))


def _gompertz4(x: FloatArray, *, a: float, c: float, b: float, d: float) -> FloatArray:
    return np.asarray(c + (a - c) * np.exp(-np.exp(-b * (x - d))))


def _richards(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(a * np.power(1.0 + d * np.exp(-b * (x - c)), -1.0 / d))


def _von_bertalanffy(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.power(1.0 - np.exp(-b * (x - c)), 3.0))


def _brody(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * (1.0 - b * np.exp(-c * x)))


def _chapman_richards(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.power(1.0 - np.exp(-b * x), c))


def _weibull_growth(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(d + (a - d) * (1.0 - np.exp(-np.power(x / b, c))))


def _morgan_mercer_flodin(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    xd = np.power(x, d)
    return np.asarray((a * b + c * xd) / (b + xd))


def _baranyi_lag(x: FloatArray, *, y0: float, ymax: float, mu: float, lag: float) -> FloatArray:
    return np.asarray(y0 + (ymax - y0) * (1.0 - np.exp(-mu * np.maximum(x - lag, 0.0))))


def _generalized_von_bertalanffy(
    x: FloatArray, *, a: float, b: float, c: float, d: float
) -> FloatArray:
    return np.asarray(a * np.power(1.0 - b * np.exp(-c * x), d))


def _schumacher(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.exp(-b / (x + c)))


def _allometric_offset(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.power(x, b) + c)


def _exponential_linear_growth(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.exp(b * x) + c * x)


def _exponential_power_growth(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * np.exp(b * x) * np.power(x, c))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "gompertz": _gompertz,
    "gompertz4": _gompertz4,
    "richards": _richards,
    "von_bertalanffy": _von_bertalanffy,
    "brody": _brody,
    "chapman_richards": _chapman_richards,
    "weibull_growth": _weibull_growth,
    "morgan_mercer_flodin": _morgan_mercer_flodin,
    "baranyi_lag": _baranyi_lag,
    "generalized_von_bertalanffy": _generalized_von_bertalanffy,
    "schumacher": _schumacher,
    "allometric_offset": _allometric_offset,
    "exponential_linear_growth": _exponential_linear_growth,
    "exponential_power_growth": _exponential_power_growth,
}
