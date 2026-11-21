"""Auto-fit code that runs inside a job process (top-level and picklable: ``spawn`` start).

* :func:`run_autofit_job` runs the whole search with the engine's own worker pool, reports a
  progress snapshot at most every ``PROGRESS_INTERVAL_S`` seconds, and stops the search at the
  run's time budget (keeping what was fitted so far, flagged partial).
* :func:`run_open_row_job` refits one candidate of a finished run into a full ``FitBundle``.
"""

import json
import threading
import time
from typing import Any

from curvestudio.discovery import (
    AutoFitConfig,
    AutoFitError,
    AutoFitResult,
    CancelToken,
    CandidateOutcome,
    autofit,
    partial_ranking,
)
from curvestudio.discovery.engine import SMALL_RUN_JOBS
from curvestudio.workflow import WorkflowError, bundle_for_candidate
from curvestudio_api.autofit_plan import plan_counts
from curvestudio_api.storage import decode_payload
from curvestudio_api.worker import ProgressReporter

PROGRESS_INTERVAL_S = 0.5
TOP_ROWS = 5
_FITTED = ("converged", "exact_fit")
_UNCONVERGED = ("not_converged", "failed", "timed_out")


class _Progress:
    """Counts the outcomes as they arrive and reports a snapshot at most every 0.5 s."""

    def __init__(
        self, report: ProgressReporter, total: int, skipped: int, config: AutoFitConfig
    ) -> None:
        self._report = report
        self._total = total
        self._skipped = skipped
        self._rank_by = config.rank_by
        # a candidate promoted to the full multistart arrives twice: the later outcome replaces
        self._outcomes: dict[str, CandidateOutcome] = {}
        self._started = time.monotonic()
        self._last = float("-inf")

    def snapshot(self, *, partial: bool = False) -> dict[str, Any]:
        converged = sum(1 for o in self._outcomes.values() if o.outcome in _FITTED)
        failed = sum(1 for o in self._outcomes.values() if o.outcome in _UNCONVERGED)
        top = partial_ranking(list(self._outcomes.values()), self._rank_by)[:TOP_ROWS]
        return {
            "done": len(self._outcomes),
            "total": self._total,
            "converged": converged,
            "failed": failed,
            "skipped": self._skipped,
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "top": [row.model_dump(mode="json") for row in top],
            "partial": partial,
        }

    def start(self) -> None:
        self._send()

    def add(self, outcome: CandidateOutcome) -> None:
        self._outcomes[outcome.candidate_id] = outcome
        if time.monotonic() - self._last >= PROGRESS_INTERVAL_S:
            self._send()

    def _send(self) -> None:
        self._last = time.monotonic()
        self._report(self.snapshot())


def run_autofit_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
    report: ProgressReporter,
) -> dict[str, Any]:
    """Run one Auto-fit search on a stored dataset.

    ``spec`` carries ``config`` (an :class:`AutoFitConfig` as JSON) and ``budget_s``, the
    seconds after which the search is stopped. Returns ``{"ok": True, "autofit": <result
    JSON without per-point vectors>, "progress": <final snapshot>}`` or, for a run the engine
    refuses, ``{"ok": False, "code": ..., "message": ...}``. A run stopped at its budget is
    still ``ok``: its result says ``cancelled`` and the snapshot ``partial``.
    """
    started = time.monotonic()
    xy = decode_payload(payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint)
    config = AutoFitConfig.model_validate(spec["config"])
    try:
        plan = plan_counts(xy, config)
    except AutoFitError as exc:
        return {"ok": False, "code": exc.code, "message": str(exc)}
    # Starting worker interpreters costs more than fitting a handful of candidates.
    if plan.runnable <= SMALL_RUN_JOBS:
        config = config.model_copy(update={"workers": 1})

    token = CancelToken()
    budget = threading.Timer(
        max(0.0, float(spec["budget_s"]) - (time.monotonic() - started)), token.cancel
    )
    budget.daemon = True
    budget.start()
    progress = _Progress(report, plan.runnable, plan.skipped, config)
    progress.start()
    try:
        result = autofit(xy, config, cancel=token, on_outcome=progress.add)
    finally:
        budget.cancel()
    return {
        "ok": True,
        "autofit": json.loads(result.to_json(include_fits=False)),
        "progress": progress.snapshot(partial=result.cancelled),
    }


def run_open_row_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Refit one candidate of a finished run (``spec``: ``config``, ``candidate_id``,
    ``dataset_fingerprint``, and the row's ``n_starts``/``on_subsample``). Returns
    ``{"ok": True, "bundle": ...}`` or a workflow error."""
    xy = decode_payload(payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint)
    config = AutoFitConfig.model_validate(spec["config"])
    # Only the configuration and the data fingerprint are needed: the row's stage (start count
    # and whether it was fitted on the subsample) travels in the spec.
    run = AutoFitResult(
        config=config,
        dataset_fingerprint=spec["dataset_fingerprint"],
        n_active=0,
        ranked=(),
        outcomes=(),
        skipped={},
        unconverged={},
        exact_fits=(),
        warnings=(),
        cancelled=False,
    )
    try:
        bundle = bundle_for_candidate(
            xy,
            run,
            str(spec["candidate_id"]),
            compact=True,
            n_starts=spec.get("n_starts"),
            on_subsample=spec.get("on_subsample"),
        )
    except WorkflowError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    return {"ok": True, "bundle": bundle.model_dump(mode="json")}
