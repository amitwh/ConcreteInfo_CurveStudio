"""The Auto-fit engine: enumerate, screen, fit every candidate in parallel, rank.

``autofit`` is one call that returns every converged fit ranked best to worst, with the
candidates that did not make it listed beside the ranking and the reason for each.
``iter_autofit`` is the streaming form: it yields each candidate's outcome as soon as that
candidate finishes, so a UI can show (and re-rank, with
:func:`~curvestudio.discovery.partial_ranking`) rows while the run is in progress.

Stages. Every candidate is first fitted from two starts (``stage="screen"``). Only candidates
that could still change the ranking (AICc within 10 of the best, or not converged but with an
SSE within three times the best converged SSE) are fitted again with the full ``n_starts``
(``stage="final"``); that outcome replaces the screening one, so ``iter_autofit`` can yield a
candidate twice and the later outcome wins. The choice depends only on the first-stage
outcomes, never on completion order, so results are identical for any worker count. A dataset
of more than 5000 active points is screened on a deterministic subsample of about 2000, and the
best candidates under each ranking criterion are then refitted on every point.

Parallelism. Candidates run in a spawn process pool (see :mod:`curvestudio.discovery.pool`:
crash isolation, a hard watchdog per candidate, workers that die with their parent), or
in-process with ``workers=1``. A script that calls ``autofit`` with more than one worker needs
the usual ``if __name__ == "__main__":`` guard. A warm :class:`AutoFitPool` can be passed as
``pool=`` to skip the two seconds of worker start-up on repeated runs.

Cancellation. :class:`CancelToken` is checked every 50 ms while waiting, so ``cancel()`` from
another thread (or from ``on_outcome``) is noticed promptly; queued candidates are dropped and
the worker processes are terminated. In-process (``workers=1``) the token is checked between
candidates and a running candidate is bounded only by the cooperative ``time_limit``. Closing
the iterator early cancels the same way.

Reproducibility. A candidate's seed derives from the run seed and its id
(:func:`~curvestudio.discovery.worker.candidate_seed`), never from scheduling.
"""

import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, replace

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery.assemble import assemble
from curvestudio.discovery.candidates import (
    enumerate_with_capped,
    parameter_cap,
    parameter_count,
)
from curvestudio.discovery.pool import (
    AutoFitPool,
    CancelToken,
    default_workers,
    get_shared_pool,
    stream_inline,
    stream_pool,
)
from curvestudio.discovery.result import (
    AutoFitResult,
    CandidateOutcome,
    annotate_same_curves,
    rank_outcomes,
    resolve_exactness,
)
from curvestudio.discovery.screening import screen
from curvestudio.discovery.types import (
    AutoFitConfig,
    CandidateRef,
    PolynomialRef,
    RankBy,
)
from curvestudio.discovery.worker import CandidateJob

__all__ = [
    "AutoFitPool",
    "CancelToken",
    "autofit",
    "get_shared_pool",
    "iter_autofit",
    "stream_inline",
    "stream_pool",
]

SCREEN_STARTS = 2
"""Starts every candidate gets in the first stage."""

PROMOTE_AICC = 10.0
"""A screened candidate within this many AICc units of the best gets the full multistart."""

PROMOTE_SSE_FACTOR = 3.0
"""A screened candidate that did not converge gets the full multistart when its SSE is within
this factor of the best converged SSE."""

LARGE_N = 5000
"""More active points than this are screened on a subsample."""

SUBSAMPLE_N = 2000
"""Size of the deterministic subsample."""

REFIT_TOP = 10
"""Candidates refitted on the full data, per ranking criterion."""

REFIT_STARTS = 1
"""Starts of the full-data refit of a large run: one, warm-started from the subsample fit."""

LARGE_N_WORKERS = 4
"""Worker cap for a large run when the count is automatic: every worker interpreter holds
about 200 MB of libraries, and the full-data refits add their per-point vectors, so four
workers keep a 100000-point run near 1.3 GB in total."""

SMALL_RUN_JOBS = 4
"""With ``workers=None`` a run of at most this many candidates stays in-process: starting
worker interpreters (a couple of seconds of imports) costs more than fitting them."""


@dataclass(frozen=True)
class Plan:
    """What a run will do: the candidates, the data they are fitted on, what was left out."""

    data: XY
    screen_data: XY
    config: AutoFitConfig
    n_active: int
    refs: dict[str, CandidateRef]
    jobs: list[CandidateJob]
    skipped: dict[str, tuple[str, ...]]
    capped: dict[str, int]
    cap: int

    @property
    def subsampled(self) -> bool:
        return self.screen_data is not self.data


def subsample(data: XY, size: int = SUBSAMPLE_N) -> XY:
    """A deterministic evenly strided subset of the active points (first and last included)."""
    active = np.flatnonzero(
        np.ones(data.x.size, bool) if data.weights is None else data.weights > 0
    )
    keep = active[np.unique(np.linspace(0, active.size - 1, size).round().astype(np.intp))]
    return XY(
        x=data.x[keep],
        y=data.y[keep],
        weights=None if data.weights is None else data.weights[keep],
        x_name=data.x_name,
        y_name=data.y_name,
        fingerprint=f"{data.fingerprint}#subsample-{size}",
    )


def plan_run(data: XY, config: AutoFitConfig) -> Plan:
    """Enumerate and screen the candidates; raises ``AutoFitError`` before any fitting.

    Raises:
        AutoFitError: ``TOO_FEW_POINTS``, ``NO_X_VARIATION``, ``CONSTANT_RESPONSE``,
            ``INVALID_CUSTOM_MODEL`` or ``DUPLICATE_MODEL_ID``.
    """
    n_active = screen(data, []).n_active  # the data checks are decided here
    refs, capped = enumerate_with_capped(config, n_active)
    screening = screen(data, refs)
    runnable = sorted(screening.runnable, key=lambda ref: (-parameter_count(ref), ref.id))
    screen_data = subsample(data) if n_active > LARGE_N else data
    jobs = [
        CandidateJob(
            x=screen_data.x,
            y=screen_data.y,
            weights=screen_data.weights,
            x_name=data.x_name,
            y_name=data.y_name,
            fingerprint=screen_data.fingerprint,
            ref=ref,
            seed=config.seed,
            n_starts=config.n_starts,
            time_limit=config.time_limit,
            cv_folds=config.cv_folds,
        )
        for ref in runnable
    ]
    return Plan(
        data=data,
        screen_data=screen_data,
        config=config,
        n_active=n_active,
        refs={ref.id: ref for ref in refs},
        jobs=jobs,
        skipped=screening.skipped,
        capped=capped,
        cap=parameter_cap(config, n_active),
    )


def _worker_count(plan: Plan, pool: AutoFitPool | None) -> int:
    config, n_jobs = plan.config, len(plan.jobs)
    if pool is not None:
        return pool.workers
    if config.workers is not None:
        return max(1, min(config.workers, n_jobs))
    if n_jobs <= SMALL_RUN_JOBS:
        return 1
    cap = LARGE_N_WORKERS if plan.subsampled else default_workers()
    return max(1, min(default_workers(), cap, n_jobs))


def _criterion(o: CandidateOutcome) -> float | None:
    if o.metrics is None:
        return None
    return o.metrics.aicc if o.metrics.aicc is not None else o.metrics.bic


def promoted_ids(first: Sequence[CandidateOutcome]) -> set[str]:
    """Screened candidates that get the full multistart, from the first-stage outcomes alone.

    Converged candidates within ``PROMOTE_AICC`` of the best criterion (AICc, else BIC), exact
    fits and converged candidates without a criterion; candidates that did not converge but
    whose SSE is within ``PROMOTE_SSE_FACTOR`` of the best converged SSE (all of them when
    nothing converged). Failed, skipped and timed-out candidates are not promoted.
    """
    resolved = resolve_exactness(list(first))
    fitted = [o for o in resolved if o.outcome in ("converged", "exact_fit")]
    criteria = [c for o in fitted if (c := _criterion(o)) is not None]
    best = min(criteria) if criteria else None
    sses = [o.fit.stats.sse for o in fitted if o.fit is not None and o.fit.stats is not None]
    best_sse = min(sses) if sses else None
    chosen: set[str] = set()
    for o in resolved:
        if o.stage != "screen":
            continue
        if o.outcome in ("converged", "exact_fit"):
            value = _criterion(o)
            if best is None or value is None or value - best <= PROMOTE_AICC:
                chosen.add(o.candidate_id)
        elif (
            o.outcome == "not_converged"
            and o.fit is not None
            and o.fit.stats is not None
            and (best_sse is None or o.fit.stats.sse <= PROMOTE_SSE_FACTOR * best_sse)
        ):
            chosen.add(o.candidate_id)
    return chosen


def _refit_ids(latest: dict[str, CandidateOutcome]) -> list[str]:
    """The best candidates under every ranking criterion, for the full-data refit."""
    outcomes = annotate_same_curves(resolve_exactness(list(latest.values())))
    chosen: dict[str, None] = {}
    for rank_by in RankBy:
        for row in rank_outcomes(outcomes, rank_by)[:REFIT_TOP]:
            chosen[row.model_id] = None
    return list(chosen)


@dataclass
class _State:
    complete: bool = False


def _pipeline(
    plan: Plan, cancel: CancelToken | None, pool: AutoFitPool | None, state: _State | None = None
) -> Generator[CandidateOutcome]:
    """All stages of a run; yields every outcome as it arrives (later ones replace earlier)."""
    config = plan.config
    workers = _worker_count(plan, pool)
    temporary = AutoFitPool(workers) if pool is None and workers > 1 else None
    active = pool if pool is not None else temporary

    short = [False]

    def run(jobs: list[CandidateJob]) -> Generator[CandidateOutcome]:
        inner = (
            stream_inline(jobs, cancel)
            if workers == 1 and pool is None
            else stream_pool(jobs, workers, cancel, pool=active)
        )
        count = 0
        for outcome in inner:
            count += 1
            yield outcome
        if count < len(jobs):  # the stage ended early: cancelled
            short[0] = True

    def mark(outcome: CandidateOutcome) -> CandidateOutcome:
        return outcome.model_copy(update={"on_subsample": True}) if plan.subsampled else outcome

    try:
        staged = config.staged_starts and config.n_starts > SCREEN_STARTS
        first_jobs = [
            replace(job, n_starts=SCREEN_STARTS, stage="screen")
            if staged and not isinstance(job.ref, PolynomialRef)
            else job
            for job in plan.jobs
        ]
        latest: dict[str, CandidateOutcome] = {}
        for outcome in run(first_jobs):
            latest[outcome.candidate_id] = mark(outcome)
            yield latest[outcome.candidate_id]
        if short[0]:
            return
        if staged:
            chosen = promoted_ids(list(latest.values()))
            # slowest first (by the screening time): the longest fit then starts early and the
            # tail of the stage is short. Order only affects scheduling, never the numbers.
            second = sorted(
                (job for job in plan.jobs if job.ref.id in chosen),
                key=lambda job: -latest[job.ref.id].elapsed_seconds,
            )
            for outcome in run(second):
                latest[outcome.candidate_id] = mark(outcome)
                yield latest[outcome.candidate_id]
            if short[0]:
                return
        if plan.subsampled:
            by_id = {job.ref.id: job for job in plan.jobs}
            refits = []
            for cid in _refit_ids(latest):
                previous = latest[cid]
                start = (
                    None
                    if previous.fit is None or isinstance(plan.refs[cid], PolynomialRef)
                    else {p.name: p.value for p in previous.fit.parameters}
                )
                refits.append(
                    replace(
                        by_id[cid],
                        x=plan.data.x,
                        y=plan.data.y,
                        weights=plan.data.weights,
                        fingerprint=plan.data.fingerprint,
                        n_starts=REFIT_STARTS,
                        stage="final",
                        start=start,
                        cv_folds=None,
                    )
                )
            for outcome in run(refits):
                # cross-validating 100000 points per candidate is not worth it: keep the
                # subsample's CV-RMSE, which ranks the same candidates
                previous = latest[outcome.candidate_id]
                if outcome.outcome not in ("converged", "exact_fit"):
                    # the full-data refit failed or ran out of time: keep the subsample fit
                    # (and say so) instead of losing a candidate that did converge
                    outcome = previous.model_copy(
                        update={
                            "reason": "; ".join(
                                filter(
                                    None,
                                    (
                                        f"the refit on all points ended {outcome.outcome} "
                                        f"({outcome.reason}); the numbers are from the subsample",
                                        previous.reason,
                                    ),
                                )
                            )
                        }
                    )
                    yield outcome
                    continue
                before = previous.metrics
                if outcome.metrics is not None and before is not None:
                    metrics = outcome.metrics.model_copy(update={"cv_rmse": before.cv_rmse})
                    outcome = outcome.model_copy(update={"metrics": metrics})
                yield outcome
        if state is not None and not short[0]:
            state.complete = True
    finally:
        if temporary is not None:
            temporary.shutdown()


# A frozen AutoFitConfig is immutable, so sharing one default instance is safe (B008).
def iter_autofit(
    data: XY,
    config: AutoFitConfig = AutoFitConfig(),  # noqa: B008
    *,
    cancel: CancelToken | None = None,
    pool: AutoFitPool | None = None,
) -> Generator[CandidateOutcome]:
    """Fit every applicable candidate and yield each candidate's outcome as it finishes.

    Only candidates that were actually attempted are yielded, in completion order (so the
    order varies with the worker count); candidates the data cannot support are not fitted
    and appear only in :func:`autofit`'s result. A candidate promoted to the full multistart
    is yielded twice, first its two-start outcome (``stage="screen"``), later the replacing
    one (``stage="final"``); likewise a candidate refitted on the full data of a very large
    run. Keep the last outcome per ``candidate_id``. Planning happens when this is called,
    so ``AutoFitError`` is raised here, not on the first ``next``. Closing the iterator early
    cancels the run. Use :func:`~curvestudio.discovery.partial_ranking` to rank what has
    arrived. ``pool`` is a warm :class:`AutoFitPool` to run in (see :func:`autofit`).

    Raises:
        AutoFitError: ``TOO_FEW_POINTS``, ``NO_X_VARIATION``, ``CONSTANT_RESPONSE``,
            ``INVALID_CUSTOM_MODEL`` or ``DUPLICATE_MODEL_ID``.
    """
    return _pipeline(plan_run(data, config), cancel, pool)


def autofit(
    data: XY,
    config: AutoFitConfig = AutoFitConfig(),  # noqa: B008
    *,
    cancel: CancelToken | None = None,
    on_outcome: Callable[[CandidateOutcome], None] | None = None,
    pool: AutoFitPool | None = None,
) -> AutoFitResult:
    """Fit every applicable model to ``data`` and rank the converged fits best to worst.

    The candidates are the registered and generated models (filtered by ``config.families``
    and the parameter cap), the polynomials and the custom models of the configuration.
    Candidates the data cannot support, or the cap leaves out, are listed in
    ``result.skipped`` with their reasons; a candidate that fails, times out or does not
    converge is listed in ``result.unconverged`` and never ranked; exact fits are listed in
    ``result.exact_fits``. ``on_outcome`` is called in the calling thread with each outcome
    as it arrives (a candidate may arrive twice, see :func:`iter_autofit`). When ``cancel``
    is triggered the result so far is returned with ``cancelled=True``, and the candidates
    never fitted are listed as skipped with that reason. ``pool`` runs the search on warm
    workers instead of starting a fresh pool (``config.workers`` is then ignored); the default
    is a fresh pool, shut down when the run ends.

    Raises:
        AutoFitError: ``TOO_FEW_POINTS`` (fewer than 4 active points), ``NO_X_VARIATION``,
            ``CONSTANT_RESPONSE``, ``INVALID_CUSTOM_MODEL`` or ``DUPLICATE_MODEL_ID``.
    """
    started = time.perf_counter()
    plan = plan_run(data, config)
    latest: dict[str, CandidateOutcome] = {}
    state = _State()
    stream = _pipeline(plan, cancel, pool, state)
    try:
        for outcome in stream:
            latest[outcome.candidate_id] = outcome
            if on_outcome is not None:
                on_outcome(outcome)
    finally:
        stream.close()
    cancelled = not state.complete
    return assemble(plan, latest, cancelled, time.perf_counter() - started)
