"""Off-thread job bodies for the analysis controller: Auto-fit, fits, validation, cross-validation.

Every function here runs on a worker thread (the :class:`~curvestudio_desktop.jobs.JobRunner`)
or, for Auto-fit, on that thread while the engine itself uses its process pool. They import
QtCore-free code only — workflow, tools, discovery, reporting — so they are importable and
testable without a display. Nothing here touches project or UI state; results are delivered
through the runner's callbacks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitResult,
    CancelToken,
    CandidateOutcome,
    RankBy,
    RankedFit,
    autofit,
    get_shared_pool,
    partial_ranking,
    plan_run,
)
from curvestudio.tools import CrossValidation, cross_validate
from curvestudio.workflow import FitBundle, FitSpec, bundle_for_candidate, fit_dataset

LIVE_RANK_INTERVAL_SECONDS = 0.25
"""Minimum gap between two live rankings; between them the run just counts progress."""

_FITTED = ("converged", "exact_fit")
_UNCONVERGED = ("not_converged", "failed", "timed_out")

DRY_RUN_POINTS = 12
DRY_RUN_TIME_LIMIT_S = 1.0


@dataclass(frozen=True)
class AutoFitLive:
    """One streaming snapshot of a running Auto-fit, delivered to the GUI thread."""

    rows: tuple[RankedFit, ...]
    done: int
    total: int
    converged: int
    not_converged: int
    skipped: int

    def counts_text(self) -> str:
        return (
            f"{self.converged} converged · {self.not_converged} not converged · "
            f"{self.skipped} skipped"
        )


def _counts(latest: dict[str, CandidateOutcome], total: int) -> tuple[int, int, int]:
    converged = sum(1 for o in latest.values() if o.outcome in _FITTED)
    not_converged = sum(1 for o in latest.values() if o.outcome in _UNCONVERGED)
    return converged, not_converged, max(0, total - len(latest))


def _snapshot(
    latest: dict[str, CandidateOutcome], rank_by: RankBy, total: int, limit: int
) -> AutoFitLive:
    converged, not_converged, skipped = _counts(latest, total)
    rows = partial_ranking(list(latest.values()), rank_by)[:limit]
    return AutoFitLive(
        rows=rows,
        done=min(len(latest), total),
        total=total,
        converged=converged,
        not_converged=not_converged,
        skipped=skipped,
    )


LiveSink = Callable[[AutoFitLive], None]
ProgressReport = Callable[[int, int], None]


def run_autofit(
    xy: XY,
    config: AutoFitConfig,
    cancel: CancelToken,
    emit: LiveSink,
    progress: ProgressReport | None = None,
    live_rows: int = 5,
) -> AutoFitResult:
    """Run one Auto-fit and stream live snapshots through ``emit`` as outcomes land.

    The total is the run's candidate count (from ``plan_run``; the enumeration and domain
    screening are cheap, so planning twice costs little next to the fitting). ``emit`` and
    ``progress`` are called on this (worker) thread; their owners marshal to the GUI thread.
    A cancelled run still returns its partial :class:`AutoFitResult`.
    """
    total = len(plan_run(xy, config).jobs)
    latest: dict[str, CandidateOutcome] = {}
    last_emit = 0.0

    def on_outcome(outcome: CandidateOutcome) -> None:
        nonlocal last_emit
        latest[outcome.candidate_id] = outcome
        if progress is not None:
            progress(min(len(latest), total), total)
        now = time.monotonic()
        if now - last_emit >= LIVE_RANK_INTERVAL_SECONDS:
            last_emit = now
            emit(_snapshot(latest, config.rank_by, total, live_rows))

    pool = None if config.workers == 1 else get_shared_pool(config.workers)
    try:
        return autofit(xy, config, cancel=cancel, on_outcome=on_outcome, pool=pool)
    finally:
        emit(_snapshot(latest, config.rank_by, total, live_rows))


def fit_job(xy: XY, spec: FitSpec) -> FitBundle:
    """Fit one specification on the active data (``fit_dataset``: never raises numerically)."""
    return fit_dataset(xy, spec)


def open_candidate_job(xy: XY, result: AutoFitResult, candidate_id: str) -> FitBundle:
    """Refit one Auto-fit row exactly as the run fitted it and assemble its bundle."""
    return bundle_for_candidate(xy, result, candidate_id)


def cross_validate_job(
    bundle: FitBundle, spec: FitSpec, folds: int | Literal["loo"]
) -> CrossValidation:
    """Cross-validate one fit for its audit: a refit per fold, so never on the GUI thread.

    ``cross_validate`` raises ``ToolError`` for what it cannot use (a fit that did not
    converge, a fold count the data cannot carry); it reaches the caller as that exception.
    """
    return cross_validate(bundle, spec, folds=folds)


def check_spec_job(spec: FitSpec, weighted: bool) -> None:
    """Dry-run ``spec`` on a small synthetic dataset to surface configuration mistakes.

    A fit's numerical outcome never raises in the workflow layer, so only a specification
    the engine cannot run (bad equation, unknown model, impossible method/bounds, weights
    asked for but absent) raises here — as the :class:`WorkflowError` it would raise on the
    real data.

    Raises:
        WorkflowError: with the code and hint the real fit would report.
    """
    x = np.linspace(1.0, 10.0, DRY_RUN_POINTS)
    probe_data = XY(
        x=x,
        y=2.0 * x + 1.0 + 0.1 * np.sin(7.0 * x),
        weights=np.ones_like(x) if weighted else None,
        x_name="x",
        y_name="y",
        fingerprint="dry-run",
    )
    probe = spec.model_copy(
        update={"time_limit": min(spec.time_limit or DRY_RUN_TIME_LIMIT_S, DRY_RUN_TIME_LIMIT_S)}
    )
    fit_dataset(probe_data, probe, compact=True)


__all__ = [
    "AutoFitLive",
    "LiveSink",
    "check_spec_job",
    "cross_validate_job",
    "fit_job",
    "open_candidate_job",
    "run_autofit",
]
