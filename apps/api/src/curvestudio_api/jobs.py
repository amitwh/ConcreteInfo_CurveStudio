"""Fit jobs: a ``spawn`` process pool, a scheduler thread, timeouts, cancel and crash recovery.

The event loop never waits on a fit. Request handlers write a ``queued`` job row and hand a
:class:`Task` to the :class:`JobManager`; its scheduler thread dispatches tasks to worker
processes (at most ``workers`` at a time), polls their futures, and records the outcome.

* A job that overruns ``timeout_s + grace_s`` wall-clock seconds (counted from when its worker
  process is up) is marked ``failed`` / ``TIMEOUT`` and the pool is recycled (workers killed).
* A worker that dies marks its job ``failed`` / ``WORKER_CRASHED`` and the pool is rebuilt.
  When several jobs were in flight the culprit is unknown, so each is re-run alone; only a job
  that crashes by itself is blamed.
* Jobs left ``queued``/``running`` by a previous process are marked ``failed`` /
  ``SERVER_RESTART`` at start-up.

State lives in one process: run a single API process per database (see the deployment notes).
"""

import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from sqlalchemy import update

from curvestudio_api import audit
from curvestudio_api.db import SessionFactory
from curvestudio_api.logging_setup import get_logger, log_fields
from curvestudio_api.orm import Dataset, Fit, Job
from curvestudio_api.timeutil import utcnow
from curvestudio_api.worker import JobWorker, run_fit_job, run_with_marker

_log = get_logger("jobs")

ACTIVE = ("queued", "running")
STARTUP_LIMIT_S = 120.0
"""How long a worker process may take to come up before its job is given up on."""


@dataclass
class Task:
    """One job waiting for, or occupying, a worker."""

    job_id: uuid.UUID
    user_id: uuid.UUID
    dataset_id: uuid.UUID
    request_id: str | None = None
    isolated: bool = False


@dataclass
class _Flight:
    task: Task
    future: Future[dict[str, Any]]
    generation: int
    dispatched: float
    marker: Path
    running_since: float | None = None


class JobManager:
    """Owns the worker pool and the scheduler thread."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        workers: int,
        timeout_s: float,
        grace_s: float,
        worker: JobWorker = run_fit_job,
        poll_s: float = 0.05,
    ) -> None:
        self._sessions = session_factory
        self._workers = workers
        self._timeout_s = timeout_s
        self._grace_s = grace_s
        self._worker = worker
        self._poll_s = poll_s
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool: ProcessPoolExecutor | None = None
        self._generation = 0
        self._pending: deque[Task] = deque()
        self._flights: dict[uuid.UUID, _Flight] = {}
        self._marker_dir: Path | None = None

    # -- lifecycle -----------------------------------------------------------------------------

    def start(self) -> None:
        """Fail jobs a previous process left behind, build the pool, start the scheduler."""
        self._recover_after_restart()
        self._marker_dir = Path(tempfile.mkdtemp(prefix="curvestudio-jobs-"))
        self._pool = self._new_pool()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="curvestudio-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler and kill the workers. Unfinished jobs are failed at next start."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        with self._lock:
            self._kill_pool()
            self._pool = None
        if self._marker_dir is not None:
            shutil.rmtree(self._marker_dir, ignore_errors=True)
            self._marker_dir = None

    def is_running(self) -> bool:
        """Whether the scheduler thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _new_pool(self) -> ProcessPoolExecutor:
        self._generation += 1
        return ProcessPoolExecutor(max_workers=self._workers, mp_context=get_context("spawn"))

    def _kill_pool(self) -> None:
        if self._pool is not None:
            self._pool.terminate_workers()

    def _recycle(self) -> None:
        """Kill every worker and start a fresh pool. Flights of the old pool are lost."""
        self._kill_pool()
        self._pool = self._new_pool()

    def _recover_after_restart(self) -> None:
        with self._sessions() as db:
            db.execute(
                update(Job)
                .where(Job.status.in_(ACTIVE))
                .values(
                    status="failed",
                    error_code="SERVER_RESTART",
                    error_message="the server restarted before this job finished; submit it again",
                    finished_at=utcnow(),
                )
            )
            db.commit()

    # -- public operations ---------------------------------------------------------------------

    def submit(self, task: Task) -> None:
        """Queue a job whose ``queued`` row is already committed."""
        with self._lock:
            self._pending.append(task)
        self._wake.set()

    def cancel(self, job_id: uuid.UUID) -> str:
        """Cancel a queued or running job and return its status afterwards.

        A running job's worker is killed (the pool is recycled; other jobs are re-queued).
        A job that has already finished keeps its status.
        """
        with self._lock:
            self._reap()
            for task in list(self._pending):
                if task.job_id == job_id:
                    self._pending.remove(task)
            flight = self._flights.get(job_id)
            changed = self._finish(job_id, "cancelled", from_=ACTIVE)
            if changed and flight is not None:
                self._recycle_after_losing(flight)
            return self._status_of(job_id)

    def cancel_for_dataset(self, dataset_id: uuid.UUID) -> None:
        """Stop every job of a dataset that is about to be deleted."""
        with self._lock:
            ids = [t.job_id for t in self._pending if t.dataset_id == dataset_id]
            ids += [
                f.task.job_id for f in self._flights.values() if f.task.dataset_id == dataset_id
            ]
        for job_id in ids:
            self.cancel(job_id)

    # -- scheduler -----------------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._tick()
            except Exception:
                _log.exception("job scheduler tick failed")
            self._wake.wait(self._poll_s)
            self._wake.clear()

    def _tick(self) -> None:
        self._reap()
        self._watchdog()
        self._dispatch()

    def _dispatch(self) -> None:
        while self._pending and self._pool is not None and len(self._flights) < self._workers:
            task = self._pending[0]
            if any(f.task.isolated for f in self._flights.values()):
                return
            if task.isolated and self._flights:
                return
            self._pending.popleft()
            if not self._mark_running(task.job_id):
                continue  # cancelled while it waited
            arguments = self._arguments_for(task)
            if arguments is None:
                continue
            marker = self._marker_path(task.job_id)
            marker.unlink(missing_ok=True)
            try:
                future = self._pool.submit(run_with_marker, self._worker, str(marker), arguments)
            except BrokenProcessPool:
                self._recycle()
                self._pending.appendleft(task)
                self._reset_to_queued(task.job_id)
                return
            self._flights[task.job_id] = _Flight(
                task, future, self._generation, time.monotonic(), marker
            )

    def _reap(self) -> None:
        crashed: dict[int, list[_Flight]] = {}
        for flight in list(self._flights.values()):
            if not flight.future.done():
                continue
            error = None if flight.future.cancelled() else flight.future.exception()
            if isinstance(error, BrokenProcessPool):
                crashed.setdefault(flight.generation, []).append(flight)
                continue
            del self._flights[flight.task.job_id]
            self._settle(flight, error)
        for generation in crashed:
            self._handle_crash(generation)

    def _settle(self, flight: _Flight, error: BaseException | None) -> None:
        task = flight.task
        try:
            if error is not None:
                _log.error("job worker raised", exc_info=error)
                self._finish(
                    task.job_id,
                    "failed",
                    from_=("running",),
                    error_code="INTERNAL_ERROR",
                    error_message="the fit job failed unexpectedly",
                )
                return
            result = flight.future.result()
            if result.get("ok"):
                self._store_fit(task, result["bundle"])
            else:
                self._finish(
                    task.job_id,
                    "failed",
                    from_=("running",),
                    error_code=str(result["code"]),
                    error_message=str(result["message"]),
                )
        except Exception:
            _log.exception("could not record the outcome of a job")
            self._finish(
                task.job_id,
                "failed",
                from_=("running",),
                error_code="INTERNAL_ERROR",
                error_message="the fit finished but its result could not be stored",
            )
        finally:
            flight.marker.unlink(missing_ok=True)

    def _handle_crash(self, generation: int) -> None:
        """A worker of ``generation`` died, which breaks every job in that pool."""
        suspects = [f for f in self._flights.values() if f.generation == generation]
        for flight in suspects:
            del self._flights[flight.task.job_id]
            flight.marker.unlink(missing_ok=True)
        if generation == self._generation:
            self._recycle()
        for flight in reversed(suspects):
            task = flight.task
            if len(suspects) == 1 or task.isolated:
                _log.error(
                    "worker process died while running a job",
                    extra=log_fields(job=str(task.job_id)),
                )
                self._finish(
                    task.job_id,
                    "failed",
                    from_=ACTIVE,
                    error_code="WORKER_CRASHED",
                    error_message="the worker process died while fitting; the job did not complete",
                )
            else:
                task.isolated = True
                self._pending.appendleft(task)
                self._reset_to_queued(task.job_id)

    def _watchdog(self) -> None:
        now = time.monotonic()
        for flight in list(self._flights.values()):
            if flight.running_since is None and flight.marker.exists():
                flight.running_since = now
            if flight.running_since is not None:
                overdue = now - flight.running_since > self._timeout_s + self._grace_s
            else:
                overdue = now - flight.dispatched > STARTUP_LIMIT_S + self._timeout_s
            if overdue:
                self._finish(
                    flight.task.job_id,
                    "failed",
                    from_=ACTIVE,
                    error_code="TIMEOUT",
                    error_message=f"the fit did not finish within {self._timeout_s:g} s",
                )
                self._recycle_after_losing(flight)
                return  # the flight table changed; the next tick continues

    def _recycle_after_losing(self, lost: _Flight) -> None:
        """Kill the pool because ``lost`` has to stop; innocent jobs of that pool run again."""
        innocents = [
            f for f in self._flights.values() if f.generation == lost.generation and f is not lost
        ]
        for flight in [lost, *innocents]:
            self._flights.pop(flight.task.job_id, None)
            flight.marker.unlink(missing_ok=True)
        self._recycle()
        for flight in reversed(innocents):
            self._pending.appendleft(flight.task)
            self._reset_to_queued(flight.task.job_id)

    # -- database ------------------------------------------------------------------------------

    def _marker_path(self, job_id: uuid.UUID) -> Path:
        assert self._marker_dir is not None
        return self._marker_dir / f"{job_id}.started"

    def _arguments_for(self, task: Task) -> dict[str, Any] | None:
        """The worker's keyword arguments, loaded now so queued jobs hold no dataset bytes.

        A job whose dataset has vanished is failed and ``None`` is returned.
        """
        with self._sessions() as db:
            job = db.get(Job, task.job_id)
            dataset = db.get(Dataset, task.dataset_id)
            if job is not None and dataset is not None:
                return {
                    "payload": dataset.payload,
                    "x_name": dataset.x_column,
                    "y_name": dataset.y_column,
                    "fingerprint": dataset.fingerprint,
                    "spec": job.spec,
                }
        self._finish(
            task.job_id,
            "failed",
            from_=ACTIVE,
            error_code="DATASET_DELETED",
            error_message="the dataset was deleted before the job started",
        )
        return None

    def _mark_running(self, job_id: uuid.UUID) -> bool:
        with self._sessions() as db:
            result = db.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "queued")
                .values(status="running", started_at=utcnow())
            )
            db.commit()
            return bool(result.rowcount)  # type: ignore[attr-defined]

    def _reset_to_queued(self, job_id: uuid.UUID) -> None:
        with self._sessions() as db:
            db.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "running")
                .values(status="queued", started_at=None)
            )
            db.commit()

    def _finish(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        from_: tuple[str, ...],
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        with self._sessions() as db:
            result = db.execute(
                update(Job)
                .where(Job.id == job_id, Job.status.in_(from_))
                .values(
                    status=status,
                    error_code=error_code,
                    error_message=error_message,
                    finished_at=utcnow(),
                )
            )
            db.commit()
            return bool(result.rowcount)  # type: ignore[attr-defined]

    def _status_of(self, job_id: uuid.UUID) -> str:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            return job.status if job is not None else "cancelled"

    def _store_fit(self, task: Task, bundle: dict[str, Any]) -> None:
        with self._sessions() as db:
            job = db.get(Job, task.job_id)
            if job is None or job.status != "running":
                return  # cancelled or deleted while it finished
            fit = Fit(
                id=uuid.uuid4(),
                user_id=job.user_id,
                dataset_id=job.dataset_id,
                spec=job.spec,
                bundle=bundle,
                status=str(bundle["result"]["status"]),
            )
            db.add(fit)
            db.flush()
            job.status = "succeeded"
            job.result_ref = fit.id
            job.finished_at = utcnow()
            audit.record(
                db,
                "fit.create",
                "fit",
                user_id=job.user_id,
                entity_id=fit.id,
                detail={"job_id": str(job.id), "status": fit.status},
                request_id=task.request_id,
            )
            db.commit()
