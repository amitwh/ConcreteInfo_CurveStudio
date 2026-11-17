"""One Auto-fit job: fit a candidate, score it, cross-validate it. Runs in a worker process.

Everything here is module-level and picklable. A job carries the data arrays and a
:class:`~curvestudio.discovery.types.CandidateRef`, never a ``ModelSpec`` (those hold lambdas);
the worker rebuilds the model from the registry (which loads the entry-point plugins in the
fresh interpreter) or, for a custom equation, from its definition.

:func:`run_candidate` never raises for a candidate-specific problem: a model that cannot be
built, an optimizer that raises, statistics that overflow all come back as an outcome of kind
``failed`` with the reason, so one bad model cannot end a run of hundreds.
"""

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from curvestudio.data import XY
from curvestudio.discovery.candidates import resolve_spec
from curvestudio.discovery.result import CandidateOutcome
from curvestudio.discovery.types import CandidateMetrics, CandidateRef, PolynomialRef
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit, fit_polynomial
from curvestudio.models import ModelSpec, load_plugins
from curvestudio.statistics import (
    NonlinearFitter,
    PolynomialFitter,
    StatisticsError,
    information_criteria,
    k_fold_cv,
)
from curvestudio.statistics.validation import Fitter

FloatArray = NDArray[np.float64]

EXACT_FIT_RELATIVE_TOLERANCE = 1e-8
"""A fit whose RMSE is below this fraction of the data range counts as exact.

Noiseless data never gives ``SSE == 0`` exactly (rounding leaves ~1e-30), and the Gaussian
likelihood of such a fit is decided by rounding noise, so its information criteria would
rank the candidates by numerical dust. Below this tolerance the fit is exact for every
purpose a measurement can have; above it, real residuals exist and the criteria are valid.
"""

_NOTE_LIMIT = 3
"""How many cross-validation fold failures a note spells out."""


def candidate_seed(seed: int, candidate_id: str) -> int:
    """The seed of one candidate: ``sha256(f"{seed}:{id}")`` folded to 32 bits.

    Derived from the run seed and the id alone, so a candidate's numbers do not depend on
    which worker ran it, in which order, or which other candidates were in the run.
    """
    digest = hashlib.sha256(f"{seed}:{candidate_id}".encode()).digest()
    return int.from_bytes(digest[:8]) % 2**32


@dataclass(frozen=True)
class CandidateJob:
    """Everything one worker needs to fit and score one candidate."""

    x: FloatArray
    y: FloatArray
    weights: FloatArray | None
    x_name: str
    y_name: str
    fingerprint: str
    ref: CandidateRef
    seed: int
    n_starts: int
    time_limit: float
    cv_folds: int | None


def init_worker() -> None:
    """Pool initialiser: load the model plugins once, before the first job needs them."""
    load_plugins()


def describe(ref: CandidateRef, spec: ModelSpec | None) -> tuple[str, str]:
    """``(name, category)`` of a candidate."""
    if isinstance(ref, PolynomialRef):
        suffix = ", through the origin" if ref.through_origin else ""
        return f"Polynomial of degree {ref.degree}{suffix}", "polynomial"
    assert spec is not None
    return spec.name, spec.category


def fit_candidate(
    data: XY,
    ref: CandidateRef,
    spec: ModelSpec | None,
    *,
    seed: int,
    n_starts: int,
    time_limit: float | None,
) -> FitResult:
    """Fit one candidate exactly as a job does (shared with the "open this row" helper)."""
    if isinstance(ref, PolynomialRef):
        return fit_polynomial(data, ref.degree, through_origin=ref.through_origin)
    assert spec is not None
    return fit(spec, data, FitConfig(n_starts=n_starts, seed=seed, time_limit=time_limit))


def run_candidate(job: CandidateJob) -> CandidateOutcome:
    """Fit, score and cross-validate one candidate; see the module docstring.

    Outcome kinds: ``converged`` (metrics filled), ``exact_fit`` (an exact fit: no AICc/BIC),
    ``not_converged`` (the optimizer did not reach a valid optimum), ``timed_out`` (the
    cooperative time limit ended it before it converged), ``failed`` (anything raised, or
    the optimizer failed) and ``skipped`` (the model refused the data). A fit that converged
    before the time limit is kept, with the fit's own warning. Only ``KeyboardInterrupt``
    and ``SystemExit`` pass through.
    """
    started = time.perf_counter()
    name, category = job.ref.id, ""
    try:
        spec = resolve_spec(job.ref)
        name, category = describe(job.ref, spec)
        outcome = _score(job, spec, name, category)
    except Exception as exc:  # reported as a failed candidate, never swallowed
        outcome = CandidateOutcome(
            candidate_id=job.ref.id,
            name=name,
            category=category,
            outcome="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
    return outcome.model_copy(update={"elapsed_seconds": time.perf_counter() - started})


def _is_exact(fit_result: FitResult, y: FloatArray) -> bool:
    stats = fit_result.stats
    assert stats is not None
    if stats.sse == 0.0:
        return True
    spread = float(np.ptp(y))
    floor = 64.0 * float(np.finfo(np.float64).eps) * float(np.max(np.abs(y)))
    return stats.rmse <= max(EXACT_FIT_RELATIVE_TOLERANCE * spread, floor)


def _score(job: CandidateJob, spec: ModelSpec | None, name: str, category: str) -> CandidateOutcome:
    ref = job.ref
    data = XY(job.x, job.y, job.weights, job.x_name, job.y_name, job.fingerprint)
    seed = candidate_seed(job.seed, ref.id)
    result = fit_candidate(
        data, ref, spec, seed=seed, n_starts=job.n_starts, time_limit=job.time_limit
    )

    def outcome(
        kind: str, reason: str, metrics: CandidateMetrics | None = None
    ) -> CandidateOutcome:
        return CandidateOutcome.model_validate(
            {
                "candidate_id": ref.id,
                "name": name,
                "category": category,
                "outcome": kind,
                "reason": reason,
                "fit": result,
                "metrics": metrics,
            }
        )

    if result.status is not FitStatus.CONVERGED:
        reason = result.termination_reason
        if result.status is FitStatus.NOT_CONVERGED:
            kind = "timed_out" if "time limit" in reason else "not_converged"
        elif result.status is FitStatus.INVALID_INPUT:
            kind = "skipped"
        else:
            kind = "failed"
        return outcome(kind, reason)
    stats = result.stats
    if stats is None:
        return outcome("failed", "the converged fit carries no statistics")

    exact = _is_exact(result, job.y)
    aicc: float | None = None
    bic: float | None = None
    if not exact:
        try:
            criteria = information_criteria(result)
        except StatisticsError as exc:
            if exc.code != "PERFECT_FIT":
                return outcome("failed", f"{exc.code}: {exc}")
            exact = True
        else:
            aicc, bic = criteria.aicc, criteria.bic
    cv_rmse, note = _cross_validate(job, data, spec, result, seed, stats.n)
    metrics = CandidateMetrics(
        k=stats.k,
        n=stats.n,
        r2=stats.r2,
        adj_r2=stats.adj_r2,
        rmse=stats.rmse,
        aicc=aicc,
        bic=bic,
        cv_rmse=cv_rmse,
    )
    if exact:
        reason = "exact fit (residuals at rounding level): information criteria are not defined"
        return outcome("exact_fit", "; ".join(filter(None, (reason, note))), metrics)
    return outcome("converged", note, metrics)


def _cross_validate(
    job: CandidateJob,
    data: XY,
    spec: ModelSpec | None,
    result: FitResult,
    seed: int,
    n_active: int,
) -> tuple[float | None, str]:
    """``(cv_rmse, note)``: the RMSE is ``None`` with a note when CV is incomplete or fails."""
    if job.cv_folds is None:
        return None, ""
    fitter: Fitter
    if isinstance(job.ref, PolynomialRef):
        fitter = PolynomialFitter(job.ref.degree, through_origin=job.ref.through_origin)
    else:
        assert spec is not None
        start = {p.name: p.value for p in result.parameters}
        config = FitConfig(seed=seed, time_limit=job.time_limit)
        fitter = NonlinearFitter(spec, config, start=start)
    try:
        cv = k_fold_cv(data, fitter, folds=min(job.cv_folds, n_active), seed=seed)
    except Exception as exc:  # the fit stays valid; the missing CV is explained, not hidden
        return None, f"cross-validation failed: {type(exc).__name__}: {exc}"
    if cv.rmse is not None:
        return cv.rmse, ""
    shown = "; ".join(cv.reasons[:_NOTE_LIMIT])
    more = len(cv.reasons) - _NOTE_LIMIT
    return None, f"cross-validation incomplete: {shown}" + (f" (+{more} more)" if more > 0 else "")
