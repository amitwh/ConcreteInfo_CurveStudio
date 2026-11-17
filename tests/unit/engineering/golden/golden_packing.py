"""Independent NumPy references for the ``engineering.packing`` models.

All four return cumulative percent passing (0 to 100) for a particle size ``x`` in the same
unit as the size parameters. See ``golden_strength.py`` for the convention.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _fuller(x: FloatArray, *, D: float) -> FloatArray:
    return np.asarray(100.0 * np.sqrt(x / D))


def _modified_fuller(x: FloatArray, *, D: float, q: float) -> FloatArray:
    return np.asarray(100.0 * np.power(x / D, q))


def _andreasen(x: FloatArray, *, D: float) -> FloatArray:
    return np.asarray(100.0 * np.power(x / D, 1.0 / 3.0))


def _modified_andreasen(x: FloatArray, *, d_min: float, d_span: float, n: float) -> FloatArray:
    top = d_min + d_span
    return np.asarray(
        100.0 * (np.power(x, n) - np.power(d_min, n)) / (np.power(top, n) - np.power(d_min, n))
    )


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "engineering_packing_fuller": _fuller,
    "engineering_packing_modified_fuller": _modified_fuller,
    "engineering_packing_andreasen": _andreasen,
    "engineering_packing_modified_andreasen": _modified_andreasen,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "engineering_packing_modified_andreasen": lambda x, *, d_min, d_span, n: np.asarray(
        np.power(d_min + d_span, n) - np.power(d_min, n)
    ),
}
