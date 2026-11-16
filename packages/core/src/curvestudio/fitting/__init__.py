"""Fitting engine: statistics, results and (later) solvers."""

from curvestudio.fitting.linear import fit_polynomial
from curvestudio.fitting.metrics import FitStatistics, fit_statistics
from curvestudio.fitting.result import (
    FitResult,
    FitStatus,
    ParameterEstimate,
    Provenance,
    collect_provenance,
)

__all__ = [
    "FitResult",
    "FitStatistics",
    "FitStatus",
    "ParameterEstimate",
    "Provenance",
    "collect_provenance",
    "fit_polynomial",
    "fit_statistics",
]
