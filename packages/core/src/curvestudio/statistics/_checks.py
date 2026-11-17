"""Shared argument checks for the statistics package.

One place validates the arguments the uncertainty routines repeat: the confidence
``level`` and the array of ``x`` values a band or propagation is requested at. The
checks raise :class:`~curvestudio.statistics.StatisticsError` with the codes the calling
routines document (``INVALID_LEVEL``, ``INVALID_X``); sharing them changes nothing the
caller sees.
"""

import numpy as np
from numpy.typing import NDArray

from curvestudio.statistics.errors import StatisticsError

FloatArray = NDArray[np.float64]


def check_level(level: float) -> None:
    """Raise ``INVALID_LEVEL`` unless ``level`` is finite and strictly inside (0, 1)."""
    if not (np.isfinite(level) and 0.0 < level < 1.0):
        raise StatisticsError(
            "INVALID_LEVEL", f"level must lie strictly between 0 and 1, got {level}"
        )


def check_x(x: FloatArray) -> FloatArray:
    """Return ``x`` as a non-empty finite 1-D ``float64`` array, or raise ``INVALID_X``."""
    points = np.asarray(x, dtype=np.float64)
    if points.ndim != 1 or points.size == 0:
        raise StatisticsError("INVALID_X", "x must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(points)):
        raise StatisticsError("INVALID_X", "x must contain only finite values")
    return points
