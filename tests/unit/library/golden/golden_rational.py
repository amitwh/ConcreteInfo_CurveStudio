"""Independent NumPy reference implementations of the rational family.

Each callable is written from the mathematical definition (not from the expression string of
the model) and takes ``x`` and the parameters by keyword. ``test_catalogue_golden`` compares it
with the model at the example parameters, so a sign slip, a wrong power or a swapped parameter
in an expression is caught.

Pattern for a new family: add ``golden/golden_<family>.py`` (any ``golden_*.py`` is discovered)
with ``GOLDEN: dict[str, Callable[..., np.ndarray]]`` keyed by model id, and, where the model
is a ratio, ``DENOMINATORS`` with the denominator as a function of ``x`` and the parameters
(the pole check demands it stays away from zero over the example range).
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _reciprocal_offset(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a + b * np.reciprocal(x))


def _reciprocal_linear(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(np.reciprocal(a + b * x))


def _x_over_linear(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(x / (a + b * x))


def _reciprocal_quadratic_offset(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    u = np.reciprocal(x)
    return np.asarray(np.polyval([c, b, a], u))


def _pade_1_1(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(np.polyval([b, a], x) / np.polyval([c, 1.0], x))


def _pade_2_1(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(np.polyval([c, b, a], x) / np.polyval([d, 1.0], x))


def _agnesi(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a / (1.0 + b * np.square(x)))


def _pade_1_2(x: FloatArray, *, a: float, b: float, c: float, d: float) -> FloatArray:
    return np.asarray(np.polyval([b, a], x) / np.polyval([d, c, 1.0], x))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "rational_reciprocal_offset": _reciprocal_offset,
    "rational_reciprocal_linear": _reciprocal_linear,
    "rational_x_over_linear": _x_over_linear,
    "rational_reciprocal_quadratic_offset": _reciprocal_quadratic_offset,
    "rational_1_1": _pade_1_1,
    "rational_2_1": _pade_2_1,
    "rational_agnesi": _agnesi,
    "rational_1_2": _pade_1_2,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "rational_reciprocal_offset": lambda x, *, a, b: np.asarray(x),
    "rational_reciprocal_linear": lambda x, *, a, b: np.asarray(a + b * x),
    "rational_x_over_linear": lambda x, *, a, b: np.asarray(a + b * x),
    "rational_reciprocal_quadratic_offset": lambda x, *, a, b, c: np.asarray(x**2),
    "rational_1_1": lambda x, *, a, b, c: np.asarray(1.0 + c * x),
    "rational_2_1": lambda x, *, a, b, c, d: np.asarray(1.0 + d * x),
    "rational_agnesi": lambda x, *, a, b: np.asarray(1.0 + b * x**2),
    "rational_1_2": lambda x, *, a, b, c, d: np.asarray(1.0 + c * x + d * x**2),
}
