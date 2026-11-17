"""Independent NumPy references for the ``engineering.strength`` models.

Each callable is written from the mathematical definition (not from the expression string),
takes ``x`` positionally and the parameters by keyword, and is compared with the model at the
model's example parameters by ``tests/unit/engineering/test_contract.py``. ``DENOMINATORS``
holds the parameter-dependent denominator of every model that can have a pole.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _exponential_age(x: FloatArray, *, f_cu: float, k: float) -> FloatArray:
    return np.asarray(f_cu * (1.0 - np.exp(-k * x)))


def _hyperbolic_age(x: FloatArray, *, f_cu: float, k: float) -> FloatArray:
    return np.asarray(f_cu * x / (k + x))


def _logistic_age(x: FloatArray, *, f_cu: float, r: float, t_0: float) -> FloatArray:
    return np.asarray(f_cu / (1.0 + np.exp(-r * (x - t_0))))


def _aci209(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(x / (a + b * x))


def _abrams(x: FloatArray, *, A: float, B: float) -> FloatArray:
    return np.asarray(A * np.power(B, -x))


def _bolomey(x: FloatArray, *, K: float, a: float) -> FloatArray:
    return np.asarray(K * (x - a))


def _feret(x: FloatArray, *, K: float) -> FloatArray:
    return np.asarray(K * np.square(x))


def _cementitious_content(x: FloatArray, *, f_0: float, f_u: float, k: float) -> FloatArray:
    return np.asarray(f_0 + f_u * (1.0 - np.exp(-k * x)))


def _scm_replacement(x: FloatArray, *, f_0: float, b: float) -> FloatArray:
    return np.asarray(f_0 * np.power(1.0 - x, b))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "engineering_strength_exponential_age": _exponential_age,
    "engineering_strength_hyperbolic_age": _hyperbolic_age,
    "engineering_strength_logistic_age": _logistic_age,
    "engineering_strength_aci209": _aci209,
    "engineering_strength_abrams": _abrams,
    "engineering_strength_bolomey": _bolomey,
    "engineering_strength_feret": _feret,
    "engineering_strength_cementitious_content": _cementitious_content,
    "engineering_strength_scm_replacement": _scm_replacement,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "engineering_strength_hyperbolic_age": lambda x, *, f_cu, k: np.asarray(k + x),
    "engineering_strength_aci209": lambda x, *, a, b: np.asarray(a + b * x),
}
