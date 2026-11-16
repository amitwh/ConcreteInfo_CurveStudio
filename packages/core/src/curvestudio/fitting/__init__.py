"""Fitting engine: statistics, results, linear and nonlinear solvers."""

from curvestudio.fitting.api import fit
from curvestudio.fitting.config import FitConfig
from curvestudio.fitting.linear import fit_polynomial
from curvestudio.fitting.methods import check_method_compatibility
from curvestudio.fitting.metrics import FitStatistics, fit_statistics
from curvestudio.fitting.result import (
    FitResult,
    FitStatus,
    ParameterEstimate,
    Provenance,
    collect_provenance,
)

__all__ = [
    "FitConfig",
    "FitResult",
    "FitStatistics",
    "FitStatus",
    "ParameterEstimate",
    "Provenance",
    "check_method_compatibility",
    "collect_provenance",
    "fit",
    "fit_polynomial",
    "fit_statistics",
]
