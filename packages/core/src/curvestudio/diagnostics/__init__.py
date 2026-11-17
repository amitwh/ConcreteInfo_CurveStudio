"""Residual diagnostics and model-adequacy checks built on :mod:`curvestudio.statistics`."""

from curvestudio.diagnostics.hypothesis_tests import (
    DiagnosticsReport,
    DiagnosticTest,
    anderson_darling,
    breusch_pagan,
    durbin_watson,
    ljung_box,
    run_diagnostics,
    shapiro_wilk,
)
from curvestudio.diagnostics.residuals import ResidualSet, press_rmse, residual_set

__all__ = [
    "DiagnosticTest",
    "DiagnosticsReport",
    "ResidualSet",
    "anderson_darling",
    "breusch_pagan",
    "durbin_watson",
    "ljung_box",
    "press_rmse",
    "residual_set",
    "run_diagnostics",
    "shapiro_wilk",
]
