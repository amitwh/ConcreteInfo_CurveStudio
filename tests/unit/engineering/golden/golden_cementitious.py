"""Independent NumPy references for the ``engineering.cementitious`` models.

See ``golden_strength.py`` for the convention.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _cement_strength(x: FloatArray, *, f_28: float, s: float) -> FloatArray:
    return np.asarray(f_28 * np.exp(s * (1.0 - np.sqrt(28.0 / x))))


def _fly_ash(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(1.0 - a * np.power(x, b))


def _ggbs(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(1.0 - a * x / (b + x))


def _pozzolanic(x: FloatArray, *, R_u: float, k: float, n: float) -> FloatArray:
    powered = np.power(x, n)
    return np.asarray(R_u * powered / (np.power(k, n) + powered))


def _hydration_heat(x: FloatArray, *, Q_u: float, tau: float, beta: float) -> FloatArray:
    return np.asarray(Q_u * np.exp(-np.power(tau / x, beta)))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "engineering_cementitious_cement_strength": _cement_strength,
    "engineering_cementitious_fly_ash_replacement": _fly_ash,
    "engineering_cementitious_ggbs_replacement": _ggbs,
    "engineering_cementitious_pozzolanic_reactivity": _pozzolanic,
    "engineering_cementitious_hydration_heat": _hydration_heat,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "engineering_cementitious_ggbs_replacement": lambda x, *, a, b: np.asarray(b + x),
    "engineering_cementitious_pozzolanic_reactivity": lambda x, *, R_u, k, n: np.asarray(
        np.power(k, n) + np.power(x, n)
    ),
}
