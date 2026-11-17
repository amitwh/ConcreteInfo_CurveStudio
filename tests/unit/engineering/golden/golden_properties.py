"""Independent NumPy references for the ``engineering.properties`` models.

See ``golden_strength.py`` for the convention: the reference comes from the mathematics, not
from the model's expression string, and every ratio declares its denominator.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_KELVIN_REFERENCE = 293.15
"""20 degC in kelvin, the reference temperature of the Arrhenius model."""


def _slump_dosage(x: FloatArray, *, S_u: float, k: float, b: float) -> FloatArray:
    return np.asarray(S_u * (1.0 - np.exp(-np.power(x / k, b))))


def _upv_exponential(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a * (np.exp(b * x) - 1.0))


def _upv_power(x: FloatArray, *, a: float, b: float, V_0: float) -> FloatArray:
    return np.asarray(a * np.power(x - V_0, b))


def _elastic_modulus(x: FloatArray, *, a: float, b: float) -> FloatArray:
    return np.asarray(a * np.power(x, b))


def _shrinkage(x: FloatArray, *, e_u: float, k: float, m: float) -> FloatArray:
    return np.asarray(e_u * np.power(x / (k + x), m))


def _creep(x: FloatArray, *, phi_u: float, psi: float, d: float) -> FloatArray:
    powered = np.power(x, psi)
    return np.asarray(phi_u * powered / (d + powered))


def _maturity_strength(x: FloatArray, *, f_u: float, k: float, M_0: float) -> FloatArray:
    return np.asarray(f_u * (1.0 - np.exp(-(x - M_0) / k)))


def _arrhenius(x: FloatArray, *, f_ref: float, b: float) -> FloatArray:
    return np.asarray(f_ref * np.exp(-b * (1.0 / x - 1.0 / _KELVIN_REFERENCE)))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "engineering_properties_slump_dosage": _slump_dosage,
    "engineering_properties_upv_strength_exponential": _upv_exponential,
    "engineering_properties_upv_strength_power": _upv_power,
    "engineering_properties_elastic_modulus": _elastic_modulus,
    "engineering_properties_shrinkage_hyperbolic": _shrinkage,
    "engineering_properties_creep": _creep,
    "engineering_properties_maturity_strength": _maturity_strength,
    "engineering_properties_arrhenius_equivalent_age": _arrhenius,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "engineering_properties_shrinkage_hyperbolic": lambda x, *, e_u, k, m: np.asarray(k + x),
    "engineering_properties_creep": lambda x, *, phi_u, psi, d: np.asarray(d + np.power(x, psi)),
}
