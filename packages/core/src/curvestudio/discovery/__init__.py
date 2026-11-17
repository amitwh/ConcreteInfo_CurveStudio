"""Model discovery: candidates, screening and ranking for Auto-fit."""

from curvestudio.discovery.candidates import candidate_ref, enumerate_candidates, resolve_spec
from curvestudio.discovery.curves import RowCurve
from curvestudio.discovery.engine import CancelToken, autofit, iter_autofit
from curvestudio.discovery.ranking import (
    best_overall_order,
    find_same_curves,
    overfitting_note,
    rank,
)
from curvestudio.discovery.result import (
    AutoFitResult,
    CandidateOutcome,
    annotate_same_curves,
    partial_ranking,
    rank_outcomes,
)
from curvestudio.discovery.screening import Screening, screen
from curvestudio.discovery.types import (
    AutoFitConfig,
    AutoFitError,
    CandidateMetrics,
    CandidateRef,
    CustomRef,
    ModelRef,
    PolynomialRef,
    RankBy,
    RankedFit,
    RegisteredRef,
)
from curvestudio.discovery.worker import CandidateJob, candidate_seed, fit_candidate, run_candidate
from curvestudio.models import CustomModelDefinition

__all__ = [
    "AutoFitConfig",
    "AutoFitError",
    "AutoFitResult",
    "CancelToken",
    "CandidateJob",
    "CandidateMetrics",
    "CandidateOutcome",
    "CandidateRef",
    "CustomModelDefinition",
    "CustomRef",
    "ModelRef",
    "PolynomialRef",
    "RankBy",
    "RankedFit",
    "RegisteredRef",
    "RowCurve",
    "Screening",
    "annotate_same_curves",
    "autofit",
    "best_overall_order",
    "candidate_ref",
    "candidate_seed",
    "enumerate_candidates",
    "find_same_curves",
    "fit_candidate",
    "iter_autofit",
    "overfitting_note",
    "partial_ranking",
    "rank",
    "rank_outcomes",
    "resolve_spec",
    "run_candidate",
    "screen",
]
