"""Residual diagnostics and model-adequacy checks built on :mod:`curvestudio.statistics`."""

from curvestudio.diagnostics.residuals import ResidualSet, press_rmse, residual_set

__all__ = [
    "ResidualSet",
    "press_rmse",
    "residual_set",
]
