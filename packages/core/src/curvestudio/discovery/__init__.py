"""Model discovery: candidates, screening and ranking for Auto-fit."""

from curvestudio.discovery.candidates import enumerate_candidates
from curvestudio.discovery.ranking import (
    best_overall_order,
    find_same_curves,
    overfitting_note,
    rank,
)
from curvestudio.discovery.screening import Screening, screen
from curvestudio.discovery.types import (
    AutoFitConfig,
    AutoFitError,
    CandidateMetrics,
    CandidateRef,
    ModelRef,
    PolynomialRef,
    RankBy,
    RankedFit,
    RegisteredRef,
)

__all__ = [
    "AutoFitConfig",
    "AutoFitError",
    "CandidateMetrics",
    "CandidateRef",
    "ModelRef",
    "PolynomialRef",
    "RankBy",
    "RankedFit",
    "RegisteredRef",
    "Screening",
    "best_overall_order",
    "enumerate_candidates",
    "find_same_curves",
    "overfitting_note",
    "rank",
    "screen",
]
