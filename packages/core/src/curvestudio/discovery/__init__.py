"""Model discovery: candidates, screening and ranking for Auto-fit."""

from curvestudio.discovery.candidates import (
    candidate_ref,
    enumerate_candidates,
    enumerate_with_capped,
    resolve_spec,
)
from curvestudio.discovery.curves import RowCurve
from curvestudio.discovery.engine import Plan, autofit, iter_autofit, plan_run
from curvestudio.discovery.pool import (
    AutoFitPool,
    CancelToken,
    get_shared_pool,
    shutdown_shared_pools,
)
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
    resolve_exactness,
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
    "AutoFitPool",
    "AutoFitResult",
    "CancelToken",
    "CandidateJob",
    "CandidateMetrics",
    "CandidateOutcome",
    "CandidateRef",
    "CustomModelDefinition",
    "CustomRef",
    "ModelRef",
    "Plan",
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
    "enumerate_with_capped",
    "find_same_curves",
    "fit_candidate",
    "get_shared_pool",
    "iter_autofit",
    "overfitting_note",
    "partial_ranking",
    "plan_run",
    "rank",
    "rank_outcomes",
    "resolve_exactness",
    "resolve_spec",
    "run_candidate",
    "screen",
    "shutdown_shared_pools",
]
