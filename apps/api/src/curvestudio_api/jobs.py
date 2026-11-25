"""Fit jobs: a warm worker pool (one process per job all the same), a fair queue, a
scheduler thread.

The event loop never waits on a fit. Request handlers write a ``queued`` job row and hand a
:class:`Task` to the :class:`JobManager`; its scheduler thread runs at most ``workers`` jobs
at a time, picking the next job round-robin across users, and records outcomes.

Each job still has its own process: a warm worker (:mod:`curvestudio_api.worker_pool`)
imports the stack once and then forks a pristine child per job, so a job no longer pays an
interpreter start, while stopping one (cancel, timeout) or losing one (crash) still never
touches another job or another tenant — everything a job does dies with its child, and the
worker itself never runs job code:

* a job that overruns ``timeout_s + grace_s`` wall-clock seconds, counted from when its process
  is up, is killed, its worker recycled, and the job marked ``failed`` / ``TIMEOUT``
  (``timeout_s`` is longer for ``autofit``);
* a process that dies without a result marks its job ``failed`` / ``WORKER_CRASHED``;
* a result larger than ``max_result_bytes`` marks it ``failed`` / ``RESULT_TOO_LARGE``;
* a worker is recycled without ceremony after ``worker_max_jobs`` jobs or above
  ``worker_memory_mb`` of its own resident size (its slot is refilled on demand);
* jobs left ``queued``/``running`` by a previous API process are marked ``failed`` /
  ``SERVER_RESTART`` at start-up.

Job kinds: ``fit`` (one fit), ``autofit`` (a search over many models, with progress and its
own worker pool), ``open_row`` (refit one Auto-fit row into a stored fit), ``audit``
(cross-validate a stored fit; the result is stored on that fit), and the three fit-centric
tools — ``discern`` (the model discernibility report), ``simulate`` (the experiment
feasibility simulator) and ``advise`` (the next-measurement advisor), each storing its result
as a ``stat_results`` row and pointing the job at it. A job child leads its own process
group; stopping a job kills the group, so the engine's pool dies with it.

State lives in one process: run a single API process per database (see the README).
"""

import contextlib
import json
import os
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from curvestudio_api import audit
from curvestudio_api.audit_worker import run_cross_validation_job
from curvestudio_api.autofit_worker import run_autofit_job, run_open_row_job
from curvestudio_api.db import SessionFactory
from curvestudio_api.fit_tools_worker import (
    run_advice_job,
    run_discernibility_job,
    run_simulation_job,
)
from curvestudio_api.logging_setup import get_logger, log_fields
from curvestudio_api.modules.errors.capture import bind_error_capture_sink
from curvestudio_api.orm import Autofit, Dataset, Fit, Job, StatResult
from curvestudio_api.timeutil import utcnow
from curvestudio_api.worker import PROGRESS, JobWorker, run_fit_job
from curvestudio_api.worker_pool import (
    EXITED,
    RETIRE,
    WORKER_FUNCTIONS,
    child_pid_of_hello,
    parse_exited,
    send_ticket,
    warm_worker_main,
)

_log = get_logger("jobs")

ACTIVE = ("queued", "running")
STARTUP_LIMIT_S = 120.0
"""How long a job process may take to come up before its job is given up on."""
REAP_TIMEOUT_S = 5.0
NAMED = {
    "autofit": "the Auto-fit run",
    "audit": "the cross-validation",
    "discern": "the discernibility report",
    "simulate": "the feasibility simulation",
    "advise": "the next-measurement advice",
}
"""How a job's timeout message names it, by kind. A job of any other kind is "the fit"."""
FIT_TOOL_KINDS = {"discern": "discernibility", "simulate": "feasibility", "advise": "advisor"}
"""The fitting tools' job kinds and the ``stat_results.tool`` each stores its result under."""
FIT_TOOL_NAMES = {
    "discern": "discernibility report",
    "simulate": "feasibility simulation",
    "advise": "next-measurement advice",
}
"""How each fit-tool kind names itself inside a message."""
FIT_DELETED = "FIT_DELETED"
"""A fit-tool job whose fit was deleted: its result has nowhere to go."""
NO_RANKING = "NO_RANKING"
"""A discernibility report whose fit was not opened from an Auto-fit result."""


@dataclass
class Task:
    """One job waiting for, or occupying, a worker process."""

    job_id: uuid.UUID
    user_id: uuid.UUID
    dataset_id: uuid.UUID
    request_id: str | None = None
    kind: str = "fit"


class FairQueue:
    """FIFO per user, round-robin across users: one user's backlog cannot starve the others."""

    def __init__(self) -> None:
        self._by_user: dict[uuid.UUID, deque[Task]] = {}
        self._ring: deque[uuid.UUID] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, task: Task) -> None:
        if task.user_id not in self._by_user:
            self._by_user[task.user_id] = deque()
            self._ring.append(task.user_id)
        self._by_user[task.user_id].append(task)
        self._size += 1

    def pop(self) -> Task:
        """The next task: the oldest one of the user whose turn it is.

        Raises:
            IndexError: the queue is empty.
        """
        if not self._ring:
            raise IndexError("pop from an empty FairQueue")
        user = self._ring[0]
        tasks = self._by_user[user]
        task = tasks.popleft()
        self._size -= 1
        if tasks:
            self._ring.rotate(-1)
        else:
            self._ring.popleft()
            del self._by_user[user]
        return task

    def remove(self, job_id: uuid.UUID) -> bool:
        """Drop a waiting task; false when it is not queued."""
        for user, tasks in self._by_user.items():
            for task in tasks:
                if task.job_id == job_id:
                    tasks.remove(task)
                    self._size -= 1
                    if not tasks:
                        del self._by_user[user]
                        self._ring.remove(user)
                    return True
        return False

    def tasks(self) -> list[Task]:
        return [t for tasks in self._by_user.values() for t in tasks]


@dataclass
class _Flight:
    task: Task
    dispatched: float
    worker: _WorkerProc | None = None
    child_pid: int | None = None
    running_since: float | None = None
    killed: bool = False


@dataclass
class _WorkerProc:
    """One slot of the warm pool: a process that forks a child per job.

    ``job_id`` is the job whose ticket it holds (``None`` between jobs). ``recycle`` is set
    when an ``EXITED`` report crossed a recycle limit while the worker was busy: it is
    retired at the next result instead, so a ticket is never lost to a retirement.
    """

    process: BaseProcess | None = None
    connection: Connection | None = None
    job_id: uuid.UUID | None = None
    jobs_done: int = 0
    rss_kb: int = 0
    retired: bool = False
    retired_at: float | None = None
    recycle: bool = False


class JobManager:
    """Owns the queue, the warm worker pool and the scheduler thread."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        workers: int,
        timeout_s: float,
        grace_s: float,
        max_result_bytes: int,
        queue_max: int,
        worker: JobWorker = run_fit_job,
        autofit_worker: JobWorker = run_autofit_job,
        open_row_worker: JobWorker = run_open_row_job,
        audit_worker: JobWorker = run_cross_validation_job,
        discern_worker: JobWorker = run_discernibility_job,
        simulate_worker: JobWorker = run_simulation_job,
        advise_worker: JobWorker = run_advice_job,
        autofit_timeout_s: float | None = None,
        poll_s: float = 0.05,
        worker_max_jobs: int = 50,
        worker_memory_mb: int = 384,
    ) -> None:
        self._sessions = session_factory
        self._workers = workers
        self._timeout_s = timeout_s
        self._grace_s = grace_s
        self._max_result_bytes = max_result_bytes
        self._queue_max = queue_max
        self._workers_by_kind = {
            **WORKER_FUNCTIONS,
            "fit": worker,
            "autofit": autofit_worker,
            "open_row": open_row_worker,
            "audit": audit_worker,
            "discern": discern_worker,
            "simulate": simulate_worker,
            "advise": advise_worker,
        }
        self._autofit_timeout_s = autofit_timeout_s if autofit_timeout_s is not None else timeout_s
        self._poll_s = poll_s
        self._worker_max_jobs = worker_max_jobs
        self._worker_memory_kb = worker_memory_mb * 1024
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue = FairQueue()
        self._flights: dict[uuid.UUID, _Flight] = {}
        self._pool: list[_WorkerProc] = []
        self._context = get_context("spawn")
        self.spawned_workers = 0
        """Warm workers started so far (for tests and diagnostics)."""

    # -- lifecycle -----------------------------------------------------------------------------

    def start(self) -> None:
        """Fail jobs a previous process left behind and start the scheduler."""
        self._recover_after_restart()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="curvestudio-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler and kill every worker and job process. Unfinished jobs fail at
        the next start."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        with self._lock:
            flights = list(self._flights.values())
            self._flights.clear()
            pool, self._pool = self._pool[:], []
        for flight in flights:
            self._kill(flight)  # each job child, and the pool it started
        for worker in pool:
            self._drop(worker)

    def is_running(self) -> bool:
        """Whether the scheduler thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def queue_size(self) -> int:
        """Jobs waiting for a worker (running ones not counted)."""
        with self._lock:
            return len(self._queue)

    def worker_count(self) -> int:
        """How many jobs may run at once (one warm worker each)."""
        return self._workers

    def child_pid_of(self, job_id: uuid.UUID) -> int | None:
        """The process id a running job's child announced, for tests and diagnostics."""
        with self._lock:
            flight = self._flights.get(job_id)
            return flight.child_pid if flight is not None else None

    def queue_max(self) -> int:
        """How many jobs may wait in the queue."""
        return self._queue_max

    def queue_has_room(self) -> bool:
        return self.queue_size() < self._queue_max

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
            self._queue.push(task)
        self._wake.set()

    def cancel(self, job_id: uuid.UUID) -> str:
        """Cancel a queued or running job and return its status afterwards.

        Only that job's own process is killed; the warm worker it ran on never executed job
        code, so it is freed at once for the next one. A finished job keeps its status.
        """
        with self._lock:
            self._queue.remove(job_id)
            changed = self._finish(job_id, "cancelled", from_=ACTIVE)
            flight = self._flights.pop(job_id, None) if changed else None
            if flight is not None:
                worker = flight.worker
                if worker is not None and worker.job_id == job_id:
                    worker.job_id = None
                self._kill(flight)
        return self._status_of(job_id)

    def cancel_for_dataset(self, dataset_id: uuid.UUID) -> None:
        """Stop every job of a dataset that is about to be deleted."""
        with self._lock:
            ids = [t.job_id for t in self._queue.tasks() if t.dataset_id == dataset_id]
            ids += [
                f.task.job_id for f in self._flights.values() if f.task.dataset_id == dataset_id
            ]
        for job_id in ids:
            self.cancel(job_id)

    def cancel_for_user(self, user_id: uuid.UUID) -> None:
        """Stop every queued or running job of a user (account deletion)."""
        with self._lock:
            ids = [t.job_id for t in self._queue.tasks() if t.user_id == user_id]
            ids += [f.task.job_id for f in self._flights.values() if f.task.user_id == user_id]
        for job_id in ids:
            self.cancel(job_id)

    # -- scheduler -----------------------------------------------------------------------------

    def _run(self) -> None:
        # A failure logged on this thread is recorded into this manager's own database.
        bind_error_capture_sink(self._sessions)
        while not self._stop.is_set():
            try:
                self._watchdog()
                self._collect()
                self._dispatch()
            except Exception:
                _log.exception("job scheduler tick failed")
            self._wake.wait(self._poll_s)
            self._wake.clear()

    def _dispatch(self) -> None:
        """Start queued jobs while there are free slots: an idle warm worker, or room to
        grow the pool by one."""
        while True:
            with self._lock:
                if len(self._queue) == 0:
                    return
                usable = any(
                    w.job_id is None and not w.retired and w.connection is not None
                    for w in self._pool
                )
                if not usable and len(self._pool) >= self._workers:
                    return
                task = self._queue.pop()
                if not self._mark_running(task.job_id):
                    continue  # cancelled while it waited
                flight = _Flight(task, time.monotonic())
                self._flights[task.job_id] = flight
            self._start(flight)

    def _start(self, flight: _Flight) -> None:
        """Load the data and hand the job to a warm worker. Runs without the manager lock:
        pickling a large payload, or starting a worker, can take a while."""
        task = flight.task
        arguments = self._arguments_for(task)
        if arguments is None:
            with self._lock:
                self._flights.pop(task.job_id, None)
            return
        worker = self._acquire_worker()
        if worker is None:
            _log.error("no job worker could be started", extra=log_fields(job=str(task.job_id)))
            with self._lock:
                self._flights.pop(task.job_id, None)
            self._finish(
                task.job_id,
                "failed",
                from_=("running",),
                error_code="WORKER_CRASHED",
                error_message="the worker process died while fitting; the job did not complete",
            )
            return
        with self._lock:
            flight.worker = worker
            worker.job_id = task.job_id
        connection = worker.connection
        sent = connection is not None and send_ticket(
            connection,
            self._workers_by_kind[task.kind],
            arguments,
            task.kind == "autofit",
            task.job_id,
        )
        with self._lock:
            if sent and self._flights.get(task.job_id) is flight and not flight.killed:
                return
        if not sent:  # the worker died between acquisition and the ticket
            with self._lock:
                worker.job_id = None
            self._drop(worker)
            with self._lock:
                self._flights.pop(task.job_id, None)
            self._finish(
                task.job_id,
                "failed",
                from_=("running",),
                error_code="WORKER_CRASHED",
                error_message="the worker process died while fitting; the job did not complete",
            )
            return
        self._kill(flight)  # cancelled while it was starting

    def _acquire_worker(self) -> _WorkerProc | None:
        """An idle warm worker, or a freshly started one when the pool has room."""
        with self._lock:
            for worker in self._pool:
                if worker.job_id is None and not worker.retired and worker.connection is not None:
                    return worker
            if len(self._pool) >= self._workers:
                return None
        ours, theirs = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=warm_worker_main,
            args=(theirs,),
            name="curvestudio-job-worker",
            # A job child may start an Auto-fit pool of its own, which a daemon process may
            # not; the worker and its children end with this process anyway.
            daemon=False,
        )
        try:
            process.start()
        except OSError:
            _log.exception("could not start a job worker")
            ours.close()
            return None
        finally:
            theirs.close()  # the worker holds the only far end, so its death reads as EOF
        worker = _WorkerProc(process=process, connection=ours)
        with self._lock:
            self._pool.append(worker)
            self.spawned_workers += 1
        return worker

    def _timeout_for(self, kind: str) -> float:
        return self._autofit_timeout_s if kind == "autofit" else self._timeout_s

    def _watchdog(self) -> None:
        now = time.monotonic()
        with self._lock:
            for flight in list(self._flights.values()):
                timeout_s = self._timeout_for(flight.task.kind)
                if flight.running_since is not None:
                    overdue = now - flight.running_since > timeout_s + self._grace_s
                else:
                    overdue = now - flight.dispatched > STARTUP_LIMIT_S + timeout_s
                what = NAMED.get(flight.task.kind, "the fit")
                if overdue and self._finish(
                    flight.task.job_id,
                    "failed",
                    from_=ACTIVE,
                    error_code="TIMEOUT",
                    error_message=f"{what} did not finish within {timeout_s:g} s",
                ):
                    self._flights.pop(flight.task.job_id, None)
                    worker = flight.worker
                    if worker is not None and worker.job_id == flight.task.job_id:
                        worker.job_id = None
                    self._kill(flight)
                    # A run that had to be stopped is exactly the misbehaving kind: its
                    # worker is replaced, so the next job starts from a fresh process.
                    if worker is not None and flight.child_pid is not None:
                        self._retire(worker)

    def _collect(self) -> None:
        """Pick up results and deaths. Reading and parsing happen without the manager lock."""
        with self._lock:
            workers = list(self._pool)
        for worker in workers:
            self._drain(worker)
        self._reap_pool()

    def _drain(self, worker: _WorkerProc) -> None:
        """Read everything a worker's pipe holds, attributing each message to the job the
        worker is running."""
        connection = worker.connection
        if connection is None:
            return
        try:
            while connection.poll(0):
                data = connection.recv_bytes(maxlength=self._max_result_bytes)
                self._handle(worker, data)
        except EOFError:
            self._worker_died(worker)
        except OSError, ValueError:
            # A message over the result cap leaves its payload half-read — the worker's
            # pipe can carry no second job — or the pipe broke. Either way the worker goes.
            flight = self._flight_on(worker)
            if flight is not None and not flight.killed:
                self._finish_flight(worker, flight, ("too_large", None))
            self._drop(worker)

    def _handle(self, worker: _WorkerProc, data: bytes) -> None:
        """One message off a worker's pipe: a hello, a progress snapshot, a result, or the
        worker's report that its child is gone."""
        if data.startswith(EXITED):
            self._worker_reported(worker, data)
            return
        flight = self._flight_on(worker)
        if flight is None:
            return  # a message from a job that was cancelled or timed out meanwhile
        if (pid := child_pid_of_hello(data)) is not None:
            flight.child_pid = pid
            flight.running_since = time.monotonic()
        elif data.startswith(PROGRESS):
            self._store_progress(flight.task.job_id, data[len(PROGRESS) :])
        else:
            self._finish_flight(worker, flight, ("result", data))

    def _worker_reported(self, worker: _WorkerProc, data: bytes) -> None:
        """A worker reaped its job's child: record the counters, fail the job if the child
        died without a result, and recycle the worker when it is over a limit."""
        report = parse_exited(data)
        if report is None:
            return
        status, jobs_done, rss_kb, job_bytes = report
        worker.jobs_done, worker.rss_kb = jobs_done, rss_kb
        flight = self._flight_on(worker)
        if flight is not None and flight.task.job_id.bytes == job_bytes:
            self._finish_flight(worker, flight, ("crashed", status))
        over = jobs_done >= self._worker_max_jobs or (
            self._worker_memory_kb > 0 and rss_kb > self._worker_memory_kb
        )
        if over and not worker.retired:
            with self._lock:
                if worker.job_id is None:
                    self._retire(worker)
                else:  # a ticket is out: retire the worker when it comes back
                    worker.recycle = True

    def _flight_on(self, worker: _WorkerProc) -> _Flight | None:
        """The flight this worker's current messages belong to, if it still has one."""
        with self._lock:
            if worker.job_id is None:
                return None
            flight = self._flights.get(worker.job_id)
            return flight if flight is not None and flight.worker is worker else None

    def _finish_flight(
        self, worker: _WorkerProc, flight: _Flight, outcome: tuple[str, Any]
    ) -> None:
        """Detach a finished flight from its worker and record the job's outcome."""
        with self._lock:
            if self._flights.get(flight.task.job_id) is not flight:
                return  # cancelled or timed out meanwhile
            del self._flights[flight.task.job_id]
            if worker.job_id == flight.task.job_id:
                worker.job_id = None
        self._finalise(flight, outcome)
        self._wake.set()
        with self._lock:
            if worker.recycle and not worker.retired and worker.job_id is None:
                self._retire(worker)

    def _worker_died(self, worker: _WorkerProc) -> None:
        """A worker's pipe closed: a job it was running is failed (nothing of it can still
        report), and the worker leaves the pool for a fresh one."""
        flight = self._flight_on(worker)
        if flight is not None:
            exitcode = worker.process.exitcode if worker.process is not None else None
            self._finish_flight(worker, flight, ("crashed", exitcode))
        self._drop(worker)

    def _reap_pool(self) -> None:
        """Drop workers that have gone, and retire ones that were asked but refused."""
        with self._lock:
            workers = list(self._pool)
        for worker in workers:
            process = worker.process
            quiet = worker.connection is None or not worker.connection.poll(0)
            if process is not None and not process.is_alive() and quiet:
                self._worker_died(worker)
            elif (
                worker.retired
                and worker.retired_at is not None
                and time.monotonic() - worker.retired_at > REAP_TIMEOUT_S
            ):
                self._drop(worker)

    def _retire(self, worker: _WorkerProc) -> None:
        """Ask an idle worker to stop; its slot is refilled when a job next needs it."""
        worker.retired = True
        worker.retired_at = time.monotonic()
        connection = worker.connection
        if connection is not None:
            with contextlib.suppress(OSError, ValueError):
                connection.send((RETIRE,))

    def _drop(self, worker: _WorkerProc) -> None:
        """Remove a worker from the pool, stopping it and closing its pipe (idempotent)."""
        with self._lock:
            if worker in self._pool:
                self._pool.remove(worker)
        process, connection = worker.process, worker.connection
        worker.process = worker.connection = None
        if process is not None:
            try:
                _kill_group(process)
                process.join(REAP_TIMEOUT_S)
                process.close()
            except ValueError:
                pass  # already closed
        if connection is not None:
            connection.close()

    def _finalise(self, flight: _Flight, outcome: tuple[str, Any]) -> None:
        task = flight.task
        kind, value = outcome
        # The messages that would say "the fit" name the kind they ran, as its timeout message
        # does; every other kind keeps the fit's wording.
        is_audit = task.kind == "audit"
        fit_tool = task.kind in FIT_TOOL_KINDS
        if fit_tool:
            what = f"the {FIT_TOOL_NAMES[task.kind]}"
        elif is_audit:
            what = "the cross-validation"
        else:
            what = "the fit"
        try:
            if kind == "result":
                result = json.loads(value)
                if result.get("ok"):
                    if task.kind == "autofit":
                        self._store_autofit(task, result)
                    elif task.kind == "open_row":
                        self._store_opened_row(task, result["bundle"])
                    elif task.kind == "audit":
                        self._store_cross_validation(task, result["cv"])
                    elif fit_tool:
                        self._store_fit_tool(task, result["result"])
                    else:
                        self._store_fit(task, result["bundle"])
                else:
                    self._finish(
                        task.job_id,
                        "failed",
                        from_=("running",),
                        error_code=str(result["code"]),
                        error_message=str(result["message"]),
                    )
            elif kind == "too_large":
                if fit_tool:
                    remedy = "use smaller options"
                else:
                    remedy = "use fewer folds" if is_audit else "fit fewer rows"
                self._finish(
                    task.job_id,
                    "failed",
                    from_=("running",),
                    error_code="RESULT_TOO_LARGE",
                    error_message=(
                        f"{what} result is larger than {self._max_result_bytes:,} bytes; {remedy}"
                    ),
                )
            else:
                _log.error("job process died", extra=log_fields(job=str(task.job_id), exit=value))
                if fit_tool:
                    doing = f"running the {FIT_TOOL_NAMES[task.kind]}"
                else:
                    doing = "cross-validating" if is_audit else "fitting"
                self._finish(
                    task.job_id,
                    "failed",
                    from_=("running",),
                    error_code="WORKER_CRASHED",
                    error_message=(
                        f"the worker process died while {doing}; the job did not complete"
                    ),
                )
        except Exception:
            _log.exception("could not record the outcome of a job")
            self._finish(
                task.job_id,
                "failed",
                from_=("running",),
                error_code="INTERNAL_ERROR",
                error_message=f"{what} finished but its result could not be stored",
            )
        finally:
            self._kill(flight)

    def _kill(self, flight: _Flight) -> None:
        """Stop one job's process and everything it started (idempotent).

        The job child leads its own process group (see ``worker_pool``), so one ``killpg``
        also ends an Auto-fit run's worker pool; its warm worker, which never ran job code,
        stays. A job whose child has not announced itself yet gets the same treatment a
        starting process used to: the child cannot be named, so the worker is stopped and
        replaced — safe because a child runs no job code before its hello, so nothing it
        started can exist either (any hello already written is picked up here first).
        """
        flight.killed = True
        pid = flight.child_pid
        if pid is None and flight.worker is not None:
            pid = self._pending_hello(flight.worker)
            flight.child_pid = pid
        if pid is not None:
            _kill_pid_group(pid)
            return
        worker = flight.worker
        flight.worker = None
        if worker is not None:
            if worker.job_id == flight.task.job_id:
                worker.job_id = None
            self._drop(worker)

    def _pending_hello(self, worker: _WorkerProc) -> int | None:
        """A job child's hello that was written but not read yet (the scheduler's next tick
        would have picked it up).

        Messages that are not the hello belong to jobs already finished or cancelled —
        their flights are gone — so they are drained and dropped. Reading stops when the
        pipe is empty: a child whose hello is not in it has run no job code, and so has
        nothing living that depends on being killed by name.
        """
        connection = worker.connection
        if connection is None:
            return None
        try:
            while connection.poll(0):
                data = connection.recv_bytes(maxlength=self._max_result_bytes)
                pid = child_pid_of_hello(data)
                if pid is not None:
                    return pid
        except EOFError, OSError, ValueError:
            pass  # the worker is going or gone; its death takes the pre-hello child too
        return None

    # -- database ------------------------------------------------------------------------------

    def _arguments_for(self, task: Task) -> dict[str, Any] | None:
        """The worker's keyword arguments, loaded now so queued jobs hold no dataset bytes.

        A job whose dataset has vanished is failed and ``None`` is returned; so is a
        cross-validation or a fit-tool job whose fit has. A discernibility report also reads
        the Auto-fit result its fit was opened from, and a fit without one fails the job.
        """
        arguments: dict[str, Any] | None = None
        fit_gone = False
        no_ranking = False
        reads_fit = task.kind == "audit" or task.kind in FIT_TOOL_KINDS
        with self._sessions() as db:
            job = db.get(Job, task.job_id)
            dataset = db.get(Dataset, task.dataset_id)
            if job is not None and dataset is not None:
                arguments = {
                    "payload": dataset.payload,
                    "x_name": dataset.x_column,
                    "y_name": dataset.y_column,
                    "fingerprint": dataset.fingerprint,
                    "spec": job.spec,
                }
                if reads_fit:  # it also reads the fit it runs on
                    fit = db.get(Fit, uuid.UUID(job.spec["fit_id"]))
                    fit_gone = fit is None
                    if fit is not None:
                        arguments["bundle"] = fit.bundle
                        arguments["fit_spec"] = fit.spec
                        if task.kind == "discern":
                            no_ranking = True
                            if fit.autofit_id is not None:
                                ranking = db.scalar(
                                    select(Autofit.result).where(Autofit.id == fit.autofit_id)
                                )
                                if ranking is not None:
                                    arguments["ranking"] = ranking
                                    no_ranking = False
        if fit_gone:
            self._finish(
                task.job_id,
                "failed",
                from_=ACTIVE,
                error_code=FIT_DELETED,
                error_message=(
                    "the fit was deleted before its cross-validation started"
                    if task.kind == "audit"
                    else f"the fit was deleted before its {FIT_TOOL_NAMES[task.kind]} started"
                ),
            )
            return None
        if no_ranking:
            self._finish(
                task.job_id,
                "failed",
                from_=ACTIVE,
                error_code=NO_RANKING,
                error_message=(
                    "this fit was not opened from an Auto-fit result; open a row from an "
                    "Auto-fit result first"
                ),
            )
            return None
        if arguments is None:
            self._finish(
                task.job_id,
                "failed",
                from_=ACTIVE,
                error_code="DATASET_DELETED",
                error_message="the dataset was deleted before the job started",
            )
            return None
        return arguments

    def _store_progress(self, job_id: uuid.UUID, raw: bytes) -> None:
        """Record a progress snapshot of a running job (a cancelled job keeps what it had)."""
        try:
            snapshot = json.loads(raw)
            with self._sessions() as db:
                db.execute(
                    update(Job)
                    .where(Job.id == job_id, Job.status == "running")
                    .values(progress=snapshot)
                )
                db.commit()
        except Exception:
            _log.exception("could not record job progress", extra=log_fields(job=str(job_id)))

    def _mark_running(self, job_id: uuid.UUID) -> bool:
        with self._sessions() as db:
            result = db.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "queued")
                .values(status="running", started_at=utcnow())
            )
            db.commit()
            return bool(result.rowcount)  # type: ignore[attr-defined]

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
        fit_id = uuid.uuid4()
        with self._sessions() as db:
            claimed = db.execute(
                # Conditional, so a cancel that lands first wins and nothing is stored.
                update(Job)
                .where(Job.id == task.job_id, Job.status == "running")
                .values(status="succeeded", result_ref=fit_id, finished_at=utcnow())
                .returning(Job.user_id, Job.dataset_id, Job.spec)
            ).one_or_none()
            if claimed is None:
                db.rollback()
                return
            user_id, dataset_id, spec = claimed
            db.add(
                Fit(
                    id=fit_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    spec=spec,
                    bundle=bundle,
                    status=str(bundle["result"]["status"]),
                )
            )
            audit.record(
                db,
                "fit.create",
                "fit",
                user_id=user_id,
                entity_id=fit_id,
                detail={"job_id": str(task.job_id), "status": str(bundle["result"]["status"])},
                request_id=task.request_id,
            )
            db.commit()

    def _store_autofit(self, task: Task, result: dict[str, Any]) -> None:
        autofit_id = uuid.uuid4()
        payload = result["autofit"]
        ranked = payload["ranked"]
        with self._sessions() as db:
            claimed = db.execute(
                update(Job)
                .where(Job.id == task.job_id, Job.status == "running")
                .values(
                    status="succeeded",
                    result_ref=autofit_id,
                    progress=result["progress"],
                    finished_at=utcnow(),
                )
                .returning(Job.user_id, Job.dataset_id)
            ).one_or_none()
            if claimed is None:
                db.rollback()
                return
            user_id, dataset_id = claimed
            db.add(
                Autofit(
                    id=autofit_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    config=payload["config"],
                    result=payload,
                    n_candidates=len(payload["outcomes"]),
                    best_model_id=ranked[0]["model_id"] if ranked else None,
                )
            )
            audit.record(
                db,
                "autofit.create",
                "autofit",
                user_id=user_id,
                entity_id=autofit_id,
                detail={
                    "job_id": str(task.job_id),
                    "candidates": len(payload["outcomes"]),
                    "partial": bool(payload["cancelled"]),
                },
                request_id=task.request_id,
            )
            db.commit()

    def _store_opened_row(self, task: Task, bundle: dict[str, Any]) -> None:
        """Store the refit of an Auto-fit row as an ordinary fit.

        If the row was opened meanwhile (two requests raced), the job points at that fit.
        """
        fit_id = uuid.uuid4()
        with self._sessions() as db:
            claimed = db.execute(
                update(Job)
                .where(Job.id == task.job_id, Job.status == "running")
                .values(status="succeeded", result_ref=fit_id, finished_at=utcnow())
                .returning(Job.user_id, Job.dataset_id, Job.spec)
            ).one_or_none()
            if claimed is None:
                db.rollback()
                return
            user_id, dataset_id, spec = claimed
            autofit_id = uuid.UUID(spec["autofit_id"])
            db.add(
                Fit(
                    id=fit_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    spec=spec["fit_spec"],
                    bundle=bundle,
                    status=str(bundle["result"]["status"]),
                    autofit_id=autofit_id,
                    candidate_id=spec["candidate_id"],
                )
            )
            audit.record(
                db,
                "fit.create",
                "fit",
                user_id=user_id,
                entity_id=fit_id,
                detail={
                    "job_id": str(task.job_id),
                    "status": str(bundle["result"]["status"]),
                    "autofit_id": str(autofit_id),
                    "candidate_id": spec["candidate_id"],
                },
                request_id=task.request_id,
            )
            try:
                db.commit()
                return
            except IntegrityError as exc:
                db.rollback()
                conflict = exc
            existing = db.scalar(
                select(Fit.id).where(
                    Fit.autofit_id == autofit_id, Fit.candidate_id == spec["candidate_id"]
                )
            )
            if existing is None:
                raise conflict  # not the one-fit-per-row constraint: the caller fails the job
            db.execute(
                update(Job)
                .where(Job.id == task.job_id)
                .values(status="succeeded", result_ref=existing, finished_at=utcnow())
            )
            db.commit()

    def _store_cross_validation(self, task: Task, cv: dict[str, Any]) -> None:
        """Store a cross-validation on the fit it was run for, and finish the job.

        The job then points at the dataset it ran on; what it produced is read from the fit.
        The fit may have been deleted while the job ran: nothing is stored and the job fails,
        rather than succeeding with its result thrown away.
        """
        fit_gone = False
        with self._sessions() as db:
            claimed = db.execute(
                # Conditional, so a cancel that lands first wins and nothing is stored.
                update(Job)
                .where(Job.id == task.job_id, Job.status == "running")
                .values(status="succeeded", result_ref=task.dataset_id, finished_at=utcnow())
                .returning(Job.user_id, Job.spec)
            ).one_or_none()
            if claimed is None:
                db.rollback()
                return
            user_id, spec = claimed
            fit_id = uuid.UUID(spec["fit_id"])
            stored = db.execute(update(Fit).where(Fit.id == fit_id).values(cv=cv))
            fit_gone = not stored.rowcount  # type: ignore[attr-defined]
            if not fit_gone:
                audit.record(
                    db,
                    "fit.cross_validate",
                    "fit",
                    user_id=user_id,
                    entity_id=fit_id,
                    detail={"job_id": str(task.job_id), "folds": spec["folds"]},
                    request_id=task.request_id,
                )
                db.commit()
            else:
                db.rollback()
        if fit_gone:
            self._finish(
                task.job_id,
                "failed",
                from_=("running",),
                error_code=FIT_DELETED,
                error_message="the fit was deleted before its cross-validation finished",
            )

    def _store_fit_tool(self, task: Task, payload: dict[str, Any]) -> None:
        """Store a fit tool's result as one ``stat_results`` row and finish the job.

        The row keeps the choices the job was given (its specification, the fit's id
        included) and the result the core returned, on the fit's dataset. The job then points
        at the row: ``GET /jobs/{id}`` gives its ``result_id`` and ``GET /stats/results/{id}``
        the result itself. A cancel that lands first wins, and nothing is stored.
        """
        result_id = uuid.uuid4()
        tool = FIT_TOOL_KINDS[task.kind]
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        with self._sessions() as db:
            claimed = db.execute(
                update(Job)
                .where(Job.id == task.job_id, Job.status == "running")
                .values(status="succeeded", result_ref=result_id, finished_at=utcnow())
                .returning(Job.user_id, Job.dataset_id, Job.spec)
            ).one_or_none()
            if claimed is None:
                db.rollback()
                return
            user_id, dataset_id, spec = claimed
            db.add(
                StatResult(
                    id=result_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    tool=tool,
                    request=spec,
                    result=payload,
                    size_bytes=size,
                )
            )
            audit.record(
                db,
                "stat.create",
                "stat_result",
                user_id=user_id,
                entity_id=result_id,
                detail={"tool": tool, "job_id": str(task.job_id)},
                request_id=task.request_id,
            )
            db.commit()


def _kill_group(process: BaseProcess) -> None:
    """SIGKILL ``process`` and its process group; just ``process`` where there are no groups
    or it never became a group leader."""
    kill_group = getattr(os, "killpg", None)
    if kill_group is not None and process.pid is not None:
        try:
            kill_group(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError, PermissionError:
            pass  # no such group (not started leading it yet, or it is gone)
    if process.is_alive():
        process.kill()


def _kill_pid_group(pid: int) -> None:
    """SIGKILL the process group ``pid`` leads — a running job's child and everything it
    started (an Auto-fit run's own pool included). The group outlives its leader, so this
    reaches stragglers even when the child itself is already gone."""
    kill_group = getattr(os, "killpg", None)
    if kill_group is None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        return
    with contextlib.suppress(ProcessLookupError):
        kill_group(pid, signal.SIGKILL)
