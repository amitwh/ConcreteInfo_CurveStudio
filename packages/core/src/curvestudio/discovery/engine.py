"""The Auto-fit engine: enumerate, screen, fit every candidate in parallel, rank.

``autofit`` is one call that returns every converged fit ranked best to worst, with the
candidates that did not make it listed beside the ranking and the reason for each.
``iter_autofit`` is the streaming form: it yields each candidate's outcome as soon as that
candidate finishes, so a UI can show (and re-rank, with
:func:`~curvestudio.discovery.partial_ranking`) rows while the run is in progress.

Parallelism. Candidates run in a ``ProcessPoolExecutor`` that always uses the ``spawn`` start
method, so Linux, macOS and Windows behave the same; ``workers=1`` runs everything in the
calling process instead (no processes, no pickling; useful for tests and tiny catalogues).
Each job carries the data and a candidate reference, and the worker rebuilds the model, so
nothing unpicklable crosses the process boundary. Because workers are spawned, a script that
calls ``autofit`` with more than one worker needs the usual ``if __name__ == "__main__":``
guard (an interactive session or an application entry point already has one). A worker that
dies (out of memory, a crash in native code) costs only the candidate it was fitting: the
in-flight candidates are re-run one at a time to find it, it is reported ``failed``, and the
run continues in a fresh pool.

Cancellation. :class:`CancelToken` is checked every 50 ms while waiting, so ``cancel()`` from
another thread (or from ``on_outcome``) is noticed promptly; queued candidates are dropped and
the worker processes are terminated, so the latency is that polling interval, not the
duration of whatever the workers were doing. In-process (``workers=1``) a candidate that is
running cannot be interrupted: the token is checked between candidates, and one candidate is
bounded by ``time_limit``, which is cooperative (checked inside the residual evaluation, so a
single stuck model call is not interrupted). Closing the iterator early cancels the same way.

Reproducibility. A candidate's seed derives from the run seed and its id
(:func:`~curvestudio.discovery.worker.candidate_seed`), never from scheduling, so the numbers
of a run are identical for any worker count and completion order.
"""

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context

from curvestudio.data import XY
from curvestudio.discovery.candidates import (
    enumerate_candidates,
    parameter_count,
    resolve_spec,
)
from curvestudio.discovery.curves import RowCurve, sample_curve
from curvestudio.discovery.ranking import overfitting_note
from curvestudio.discovery.result import (
    AutoFitResult,
    CandidateOutcome,
    annotate_same_curves,
    rank_outcomes,
)
from curvestudio.discovery.screening import screen
from curvestudio.discovery.types import AutoFitConfig, CandidateRef, RankBy
from curvestudio.discovery.worker import CandidateJob, describe, init_worker, run_candidate

_log = logging.getLogger(__name__)

POLL_SECONDS = 0.05
"""How often a waiting run looks at its cancel token."""

MAX_POOL_REBUILDS = 10
"""Fresh pools started after worker crashes before the rest of the run is given up."""

SMALL_RUN_JOBS = 4
"""With ``workers=None`` a run of at most this many candidates stays in-process: starting
worker interpreters (a few seconds of imports) costs more than fitting them."""

_UNCONVERGED = ("not_converged", "failed", "timed_out")


class CancelToken:
    """A flag to stop a run from another thread, or from ``on_outcome``."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask the run to stop; safe to call more than once and from any thread."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class _Plan:
    data: XY
    config: AutoFitConfig
    n_active: int
    refs: dict[str, CandidateRef]
    jobs: list[CandidateJob]
    skipped: dict[str, tuple[str, ...]]


def _plan(data: XY, config: AutoFitConfig) -> _Plan:
    """Enumerate and screen the candidates; raises ``AutoFitError`` before any fitting."""
    n_active = screen(data, []).n_active  # TOO_FEW_POINTS is decided here
    refs = enumerate_candidates(config, n_active)
    screening = screen(data, refs)
    runnable = sorted(screening.runnable, key=lambda ref: (-parameter_count(ref), ref.id))
    jobs = [
        CandidateJob(
            x=data.x,
            y=data.y,
            weights=data.weights,
            x_name=data.x_name,
            y_name=data.y_name,
            fingerprint=data.fingerprint,
            ref=ref,
            seed=config.seed,
            n_starts=config.n_starts,
            time_limit=config.time_limit,
            cv_folds=config.cv_folds,
        )
        for ref in runnable
    ]
    return _Plan(
        data=data,
        config=config,
        n_active=screening.n_active,
        refs={ref.id: ref for ref in refs},
        jobs=jobs,
        skipped=screening.skipped,
    )


def _worker_count(config: AutoFitConfig, n_jobs: int) -> int:
    if config.workers is not None:
        return max(1, min(config.workers, n_jobs))
    if n_jobs <= SMALL_RUN_JOBS:
        return 1
    return max(1, min((os.process_cpu_count() or 2) - 1, n_jobs))


def _lost_outcome(job: CandidateJob, reason: str) -> CandidateOutcome:
    """A ``failed`` outcome for a job whose worker never returned one."""
    try:
        name, category = describe(job.ref, resolve_spec(job.ref))
    except Exception:  # naming the candidate is a courtesy; the reason is what matters
        name, category = job.ref.id, ""
    return CandidateOutcome(
        candidate_id=job.ref.id, name=name, category=category, outcome="failed", reason=reason
    )


def stream_inline(
    jobs: list[CandidateJob],
    cancel: CancelToken | None,
    run: Callable[[CandidateJob], CandidateOutcome] = run_candidate,
) -> Generator[CandidateOutcome]:
    """Run ``jobs`` one after another in this process, checking ``cancel`` between them."""
    for job in jobs:
        if cancel is not None and cancel.cancelled:
            return
        yield run(job)


def _new_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=init_worker
    )


def stream_pool(
    jobs: list[CandidateJob],
    workers: int,
    cancel: CancelToken | None,
    run: Callable[[CandidateJob], CandidateOutcome] = run_candidate,
) -> Generator[CandidateOutcome]:
    """Run ``jobs`` in a spawn process pool, yielding outcomes in completion order.

    At most ``2 * workers`` jobs are in flight, so a cancel drops the rest of the queue and
    memory stays bounded for large data. ``run`` must be a picklable module-level function.
    When the pool breaks (a worker died) the in-flight jobs are re-run one at a time so the
    culprit alone is reported ``failed``; after ``MAX_POOL_REBUILDS`` crashes the remaining
    jobs are reported ``failed`` too.
    """
    queue = deque(jobs)
    suspects: deque[CandidateJob] = deque()
    suspect_ids: set[str] = set()
    in_flight: dict[Future[CandidateOutcome], CandidateJob] = {}
    executor: ProcessPoolExecutor | None = None
    rebuilds = 0
    finished = False
    try:
        while queue or suspects or in_flight:
            if cancel is not None and cancel.cancelled:
                return
            isolating = bool(suspects) or any(j.ref.id in suspect_ids for j in in_flight.values())
            window = 1 if isolating else 2 * workers
            while len(in_flight) < window and (suspects or (queue and not isolating)):
                job = suspects.popleft() if suspects else queue.popleft()
                if executor is None:
                    executor = _new_pool(workers)
                in_flight[executor.submit(run, job)] = job

            done, _ = wait(in_flight, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
            broken = False
            for future in done:
                job = in_flight.pop(future)
                try:
                    outcome = future.result()
                except BrokenProcessPool:
                    broken = True
                    if job.ref.id in suspect_ids:  # it was running alone: it is the culprit
                        yield _lost_outcome(
                            job,
                            "the worker process died while fitting this candidate "
                            "(crash or out of memory)",
                        )
                    else:
                        suspect_ids.add(job.ref.id)
                        suspects.append(job)
                except Exception as exc:  # e.g. a result that cannot be unpickled
                    yield _lost_outcome(job, f"{type(exc).__name__}: {exc}")
                else:
                    yield outcome
            if broken:
                for job in in_flight.values():  # the pool is dead: all of these will fail
                    suspect_ids.add(job.ref.id)
                    suspects.append(job)
                in_flight.clear()
                if executor is not None:
                    executor.terminate_workers()
                    executor = None
                rebuilds += 1
                if rebuilds > MAX_POOL_REBUILDS:
                    for job in (*suspects, *queue):
                        yield _lost_outcome(
                            job, "not fitted: worker processes kept dying during this run"
                        )
                    return
        finished = True
    finally:
        if executor is not None:
            if finished:
                executor.shutdown(wait=True)
            else:  # cancelled, abandoned by the consumer, or an error: do not wait for workers
                executor.terminate_workers()


def _stream(plan: _Plan, cancel: CancelToken | None) -> Generator[CandidateOutcome]:
    workers = _worker_count(plan.config, len(plan.jobs))
    if workers == 1:
        return stream_inline(plan.jobs, cancel)
    return stream_pool(plan.jobs, workers, cancel)


# A frozen AutoFitConfig is immutable, so sharing one default instance is safe (B008).
def iter_autofit(
    data: XY,
    config: AutoFitConfig = AutoFitConfig(),  # noqa: B008
    *,
    cancel: CancelToken | None = None,
) -> Generator[CandidateOutcome]:
    """Fit every applicable candidate and yield each candidate's outcome as it finishes.

    Only candidates that were actually attempted are yielded (once each, in completion
    order, so the order varies with the worker count); candidates the data cannot support
    are not fitted and appear only in :func:`autofit`'s result. Planning happens when this
    is called, so ``AutoFitError`` (too few points, an unusable custom model) is raised
    here, not on the first ``next``. Closing the iterator early cancels the run; see the
    module docstring for how quickly. Use
    :func:`~curvestudio.discovery.partial_ranking` to rank what has arrived.

    Raises:
        AutoFitError: ``TOO_FEW_POINTS``, ``INVALID_CUSTOM_MODEL`` or ``DUPLICATE_MODEL_ID``.
    """
    return _stream(_plan(data, config), cancel)


def autofit(
    data: XY,
    config: AutoFitConfig = AutoFitConfig(),  # noqa: B008
    *,
    cancel: CancelToken | None = None,
    on_outcome: Callable[[CandidateOutcome], None] | None = None,
) -> AutoFitResult:
    """Fit every applicable model to ``data`` and rank the converged fits best to worst.

    The candidates are the registered and generated models (filtered by
    ``config.families`` and the parameter cap), the polynomials and the custom models of the
    configuration. Candidates the data cannot support are listed in ``result.skipped`` with
    their reasons; a candidate that fails, times out or does not converge is listed in
    ``result.unconverged`` and never ranked; exact fits are listed in ``result.exact_fits``.
    ``on_outcome`` is called in the calling thread with each attempted candidate's outcome as
    it finishes. When ``cancel`` is triggered the result so far is returned with
    ``cancelled=True``, and the candidates never fitted are listed as skipped with that reason.

    Raises:
        AutoFitError: ``TOO_FEW_POINTS`` (fewer than 4 active points), ``INVALID_CUSTOM_MODEL``
            or ``DUPLICATE_MODEL_ID``.
    """
    started = time.perf_counter()
    plan = _plan(data, config)
    outcomes: list[CandidateOutcome] = []
    stream = _stream(plan, cancel)
    try:
        for outcome in stream:
            outcomes.append(outcome)
            if on_outcome is not None:
                on_outcome(outcome)
    finally:
        stream.close()
    cancelled = cancel is not None and cancel.cancelled and len(outcomes) < len(plan.jobs)
    return _assemble(plan, outcomes, cancelled, time.perf_counter() - started)


def _unfitted(ref: CandidateRef, reason: str) -> CandidateOutcome:
    spec = resolve_spec(ref)
    name, category = describe(ref, spec)
    return CandidateOutcome(
        candidate_id=ref.id, name=name, category=category, outcome="skipped", reason=reason
    )


def _assemble(
    plan: _Plan, outcomes: list[CandidateOutcome], cancelled: bool, elapsed: float
) -> AutoFitResult:
    config = plan.config
    by_id = {o.candidate_id: o for o in outcomes}
    for cid, reasons in plan.skipped.items():
        by_id[cid] = _unfitted(plan.refs[cid], "; ".join(reasons))
    never_fitted = [job.ref.id for job in plan.jobs if job.ref.id not in by_id]
    for cid in never_fitted:
        by_id[cid] = _unfitted(plan.refs[cid], "the run was cancelled before this was fitted")
    ordered = sorted(annotate_same_curves(list(by_id.values())), key=lambda o: o.candidate_id)

    skipped = {
        o.candidate_id: plan.skipped.get(o.candidate_id, (o.reason,))
        for o in ordered
        if o.outcome == "skipped"
    }
    unconverged = {o.candidate_id: o.reason for o in ordered if o.outcome in _UNCONVERGED}
    exact_fits = tuple(o.candidate_id for o in ordered if o.outcome == "exact_fit")
    ranked = rank_outcomes(ordered, config.rank_by)
    curves, curve_notes = _sample_curves(plan, ordered)

    return AutoFitResult(
        config=config,
        dataset_fingerprint=plan.data.fingerprint,
        n_active=plan.n_active,
        ranked=ranked,
        outcomes=tuple(ordered),
        skipped=skipped,
        unconverged=unconverged,
        exact_fits=exact_fits,
        warnings=_warnings(plan, ordered, cancelled, never_fitted, curve_notes),
        cancelled=cancelled,
        curves=curves,
        overfitting_note=overfitting_note(ranked, config.rank_by),
        elapsed_seconds=elapsed,
    )


def _sample_curves(
    plan: _Plan, ordered: list[CandidateOutcome]
) -> tuple[dict[str, RowCurve], list[str]]:
    """Curves and bands of the top rows under every ranking criterion (one copy each)."""
    top_n = plan.config.curve_top_n
    if top_n == 0:
        return {}, []
    wanted = sorted({r.model_id for rb in RankBy for r in rank_outcomes(ordered, rb)[:top_n]})
    by_id = {o.candidate_id: o for o in ordered}
    curves: dict[str, RowCurve] = {}
    notes: list[str] = []
    for cid in wanted:
        fit = by_id[cid].fit
        assert fit is not None  # only fitted outcomes become rows
        try:
            curve, reason = sample_curve(plan.data, fit, resolve_spec(plan.refs[cid]))
        except Exception as exc:  # one undrawable curve must not lose the whole result
            curve, reason = None, f"{type(exc).__name__}: {exc}"
        if curve is None:
            notes.append(f"no curve was sampled for {cid}: {reason}")
        else:
            curves[cid] = curve
    return curves, notes


def _warnings(
    plan: _Plan,
    ordered: list[CandidateOutcome],
    cancelled: bool,
    never_fitted: list[str],
    curve_notes: list[str],
) -> tuple[str, ...]:
    config, data = plan.config, plan.data
    warnings: list[str] = []
    if cancelled:
        warnings.append(
            f"the run was cancelled: {len(never_fitted)} of {len(plan.jobs)} candidates were "
            "never fitted, so the ranking is partial"
        )
    if plan.n_active < 10:
        warnings.append(
            f"only {plan.n_active} active points: rankings from so few points are unreliable"
        )
    if data.weights is not None:
        zero = int((data.weights <= 0.0).sum())
        warnings.append(
            "the data carries weights: every candidate was fitted weighted, so information "
            "criteria and RMSE are comparable only with each other"
            + (f"; {zero} zero-weight point(s) were left out" if zero else "")
        )
    timed_out = [o.candidate_id for o in ordered if o.outcome == "timed_out"]
    if timed_out:
        warnings.append(
            f"{len(timed_out)} candidate(s) reached the time limit of {config.time_limit:g} s "
            "before converging and are not ranked"
        )
    late = [
        o.candidate_id
        for o in ordered
        if o.outcome in ("converged", "exact_fit")
        and o.fit is not None
        and any(w.startswith("time limit reached") for w in o.fit.warnings)
    ]
    if late:
        warnings.append(
            f"{len(late)} candidate(s) converged but hit the time limit afterwards, so their "
            "multistart search is incomplete (see each row's warnings)"
        )
    exact = [o.candidate_id for o in ordered if o.outcome == "exact_fit"]
    if exact:
        warnings.append(
            f"{len(exact)} candidate(s) fit the data exactly (residuals at rounding level): "
            "no AICc or BIC exists for them, so they are listed in exact_fits and ranked only "
            "by R2, adjusted R2, RMSE or CV-RMSE"
        )
    others = sum(1 for o in ordered if o.outcome in ("not_converged", "failed"))
    if others:
        warnings.append(f"{others} candidate(s) did not converge or failed and are not ranked")
    if not any(o.outcome in ("converged", "exact_fit") for o in ordered) and not cancelled:
        warnings.append("no candidate converged")
    warnings.extend(curve_notes)
    return tuple(warnings)
