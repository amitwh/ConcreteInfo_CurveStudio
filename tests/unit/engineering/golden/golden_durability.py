"""Independent NumPy references for the ``engineering.durability`` models.

The chloride profile uses ``scipy.special.erfc``, the same complementary error function the
model's compiled expression reaches through the ``scipy`` namespace of ``sympy.lambdify``.
See ``golden_strength.py`` for the convention.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc

FloatArray = NDArray[np.float64]


def _chloride_profile(x: FloatArray, *, C_i: float, C_s: float, D_t: float) -> FloatArray:
    """Fick's second law with a constant surface concentration, semi-infinite medium."""
    return np.asarray(C_i + (C_s - C_i) * erfc(x / (2.0 * np.sqrt(D_t))))


def _carbonation(x: FloatArray, *, k: float, n: float) -> FloatArray:
    return np.asarray(k * np.power(x, n))


def _sorptivity(x: FloatArray, *, S: float, A: float) -> FloatArray:
    return np.asarray(A + S * np.sqrt(x))


def _water_absorption(x: FloatArray, *, A: float, S: float, c: float) -> FloatArray:
    return np.asarray(A + S * np.sqrt(x) + c * x)


def _permeability(x: FloatArray, *, k_l: float, b: float) -> FloatArray:
    return np.asarray(k_l * (1.0 + b / x))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "engineering_durability_chloride_profile": _chloride_profile,
    "engineering_durability_carbonation": _carbonation,
    "engineering_durability_sorptivity": _sorptivity,
    "engineering_durability_water_absorption": _water_absorption,
    "engineering_durability_permeability": _permeability,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "engineering_durability_permeability": lambda x, *, k_l, b: np.asarray(x),
}
