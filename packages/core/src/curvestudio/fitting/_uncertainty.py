"""Small covariance helpers shared by the linear and nonlinear fitting back ends."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def finite_or_none(matrix: FloatArray) -> list[list[float]] | None:
    """``matrix`` as nested lists, or ``None`` when any entry is not finite."""
    return matrix.tolist() if bool(np.all(np.isfinite(matrix))) else None


def correlation(cov: FloatArray, se: FloatArray) -> list[list[float]] | None:
    """``cov_ij / (se_i * se_j)`` with an exact unit diagonal; ``None`` if undefined."""
    if not bool(np.all(np.isfinite(se)) and np.all(se > 0.0)):
        return None
    corr = np.clip(cov / np.outer(se, se), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.tolist() if bool(np.all(np.isfinite(corr))) else None
