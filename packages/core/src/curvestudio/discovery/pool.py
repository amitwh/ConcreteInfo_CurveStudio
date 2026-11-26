"""Running candidate jobs: in this process, or in a spawn process pool that can be reused.

The pool side is built for jobs that misbehave. A worker that dies (out of memory, a crash in
native code) costs only the candidate it was fitting: the in-flight candidates are re-run one
at a time to find it, it is reported ``failed``, and the run continues in a fresh pool. A
candidate that hangs (a native call that never returns, a model that loops without
evaluating) is stopped by a hard watchdog: its deadline is ``2 * time_limit + 5`` seconds, and
when it passes the pool is terminated, that candidate is reported ``timed_out`` and the other
in-flight candidates are simply re-queued. ``time_limit`` itself is cooperative; the watchdog
is what makes the run finish. Workers also exit when their parent process disappears (see
:func:`~curvestudio.discovery.worker.init_worker`).

Workers are started with ``spawn`` by default, so Linux, macOS and Windows behave alike. A
process that has already imported the fitting stack (a warm worker that forks a child per
job) may pass ``start_method="fork"`` instead: the pool's workers then fork from that
already-imported memory in milliseconds, with the same results (a candidate's seed depends
only on the run seed and its id, never on which process fitted it).
"""

import atexit
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

from curvestudio.discovery.candidates import resolve_spec
from curvestudio.discovery.result import CandidateOutcome
from curvestudio.discovery.worker import CandidateJob, describe, init_worker, run_candidate

POLL_SECONDS = 0.05
"""How often a waiting run looks at its cancel token and its watchdog deadlines."""

MAX_POOL_REBUILDS = 10
"""Fresh pools started after worker crashes before the rest of the run is given up."""

MAX_WATCHDOG_KILLS = 10
"""Candidates stopped by the watchdog before the rest of the run is given up."""

WATCHDOG_BASE_SECONDS = 5.0


def default_deadline(job: CandidateJob) -> float:
    """Seconds a candidate may run before the watchdog stops it: ``2 * time_limit + 5``."""
    return 2.0 * job.time_limit + WATCHDOG_BASE_SECONDS


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


def default_workers() -> int:
    """One worker per core but one, at least one."""
    return max(1, (os.process_cpu_count() or 2) - 1)


def _noop() -> None:
    """A candidate-free job :meth:`AutoFitPool.prewarm` runs to start a worker."""
    return None


class AutoFitPool:
    """A pool of worker processes that can serve several :func:`autofit` runs.

    Starting worker interpreters costs about two seconds (imports of NumPy, SciPy, SymPy and
    the model catalogues). A long-lived process that runs Auto-fit repeatedly (the desktop
    application) keeps one pool and passes it as ``pool=``; the workers stay warm between
    runs. Use it as a context manager or call :meth:`shutdown`; :func:`get_shared_pool` hands
    out one that is also shut down at interpreter exit. A pool runs one search at a time (a
    second concurrent run waits for the first). A cancelled or failed run terminates the
    workers, and the next run starts fresh ones.
    """

    def __init__(self, workers: int | None = None, *, start_method: str | None = None) -> None:
        self.workers = default_workers() if workers is None else max(1, workers)
        self.start_method = start_method if start_method is not None else "spawn"
        get_context(self.start_method)  # an unknown name fails here, not at the first job
        self._executor: ProcessPoolExecutor | None = None
        self.busy = threading.Lock()

    def __enter__(self) -> AutoFitPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    @property
    def running(self) -> bool:
        """Whether worker processes currently exist."""
        return self._executor is not None

    def executor(self) -> ProcessPoolExecutor:
        """The live executor, started on first use."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context(self.start_method),
                initializer=init_worker,
                initargs=(os.getpid(), threading.current_thread() is threading.main_thread()),
            )
        return self._executor

    def prewarm(self, timeout_s: float | None = None) -> None:
        """Start every worker now, so the first run pays none of their start-up.

        One no-op job per worker is enough: the executor starts a process per submission
        until it has ``workers`` of them. Waits for the workers to come up (a caller that
        wants the cost off its own critical path runs this on a background thread); a
        ``timeout_s`` bounds that wait.
        """
        if self._executor is not None:
            return  # already started: every worker exists or is being replaced on demand
        futures = [self.executor().submit(_noop) for _ in range(self.workers)]
        for future in futures:
            future.result(timeout=timeout_s)

    def worker_pids(self) -> list[int]:
        """Process ids of the live workers (empty before the first job)."""
        executor = self._executor
        return sorted(getattr(executor, "_processes", {})) if executor is not None else []

    def discard(self) -> None:
        """Terminate the workers now; the next use starts new ones."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.terminate_workers()

    def shutdown(self) -> None:
        """Stop the workers after the jobs in progress (none, between runs)."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)


_SHARED: dict[int, AutoFitPool] = {}
_SHARED_LOCK = threading.Lock()


def get_shared_pool(workers: int | None = None) -> AutoFitPool:
    """The process-wide warm pool for ``workers`` workers, created on first request.

    Meant for a long-lived process that runs :func:`autofit` more than once. It is shut down
    at interpreter exit (and by :func:`shutdown_shared_pools`); do not use it as a context
    manager unless you want to end it.
    """
    count = default_workers() if workers is None else max(1, workers)
    with _SHARED_LOCK:
        pool = _SHARED.get(count)
        if pool is None:
            pool = _SHARED[count] = AutoFitPool(count)
        return pool


def shutdown_shared_pools() -> None:
    """Shut down every pool handed out by :func:`get_shared_pool`."""
    with _SHARED_LOCK:
        pools = list(_SHARED.values())
        _SHARED.clear()
    for pool in pools:
        pool.shutdown()


atexit.register(shutdown_shared_pools)


def lost_outcome(job: CandidateJob, reason: str, kind: str = "failed") -> CandidateOutcome:
    """An outcome for a job whose worker never returned one (``failed`` or ``timed_out``)."""
    try:
        name, category = describe(job.ref, resolve_spec(job.ref))
    except Exception:  # naming the candidate is a courtesy; the reason is what matters
        name, category = job.ref.id, ""
    return CandidateOutcome.model_validate(
        {
            "candidate_id": job.ref.id,
            "name": name,
            "category": category,
            "outcome": kind,
            "reason": reason,
            "stage": job.stage,
        }
    )


def stream_inline(
    jobs: list[CandidateJob],
    cancel: CancelToken | None,
    run: Callable[[CandidateJob], CandidateOutcome] = run_candidate,
) -> Generator[CandidateOutcome]:
    """Run ``jobs`` one after another in this process, checking ``cancel`` between them.

    There is no watchdog here: a candidate running in this process cannot be stopped, only
    asked to stop at its cooperative ``time_limit``.
    """
    for job in jobs:
        if cancel is not None and cancel.cancelled:
            return
        yield run(job)


def stream_pool(
    jobs: list[CandidateJob],
    workers: int,
    cancel: CancelToken | None,
    run: Callable[[CandidateJob], CandidateOutcome] = run_candidate,
    *,
    pool: AutoFitPool | None = None,
    deadline_of: Callable[[CandidateJob], float] = default_deadline,
) -> Generator[CandidateOutcome]:
    """Run ``jobs`` in a spawn process pool, yielding outcomes in completion order.

    At most ``workers`` jobs are in flight (so each starts at once and its watchdog deadline,
    :func:`default_deadline` unless ``deadline_of`` says otherwise, counts from its start),
    which also drops the rest of the queue at a cancel and bounds memory for large data.
    ``run`` must be a picklable module-level function. ``pool`` supplies a warm pool (its
    own worker count applies); without it a temporary one is created and shut down here.
    A crash, a watchdog stop or a cancel terminates the pool's workers.
    """
    own = pool is None
    active = AutoFitPool(workers) if pool is None else pool
    window = active.workers
    queue = deque(jobs)
    suspects: deque[CandidateJob] = deque()
    suspect_ids: set[str] = set()
    in_flight: dict[Future[CandidateOutcome], tuple[CandidateJob, float]] = {}
    rebuilds = kills = 0
    finished = False
    with active.busy:
        try:
            while queue or suspects or in_flight:
                if cancel is not None and cancel.cancelled:
                    return
                isolating = bool(suspects) or any(
                    job.ref.id in suspect_ids for job, _ in in_flight.values()
                )
                limit = 1 if isolating else window
                broken = False
                while len(in_flight) < limit and (suspects or (queue and not isolating)):
                    job = suspects.popleft() if suspects else queue.popleft()
                    try:
                        submitted = active.executor().submit(run, job)
                    except BrokenProcessPool:
                        # A worker died asynchronously, so this job was never queued: isolate it
                        # like the in-flight jobs below and replace the dead pool.
                        suspect_ids.add(job.ref.id)
                        suspects.append(job)
                        broken = True
                        break
                    in_flight[submitted] = (job, time.monotonic())

                done: set[Future[CandidateOutcome]]
                if broken:
                    done = set()  # nothing was submitted here: go straight to the dead pool
                else:
                    done, _ = wait(in_flight, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
                for future in done:
                    job, _started = in_flight.pop(future)
                    try:
                        outcome = future.result()
                    except BrokenProcessPool:
                        broken = True
                        if job.ref.id in suspect_ids:  # it was running alone: the culprit
                            yield lost_outcome(
                                job,
                                "the worker process died while fitting this candidate "
                                "(crash or out of memory)",
                            )
                        else:
                            suspect_ids.add(job.ref.id)
                            suspects.append(job)
                    except Exception as exc:  # e.g. a result that cannot be unpickled
                        yield lost_outcome(job, f"{type(exc).__name__}: {exc}")
                    else:
                        yield outcome
                if broken:
                    for job, _started in in_flight.values():  # the pool is dead: all will fail
                        suspect_ids.add(job.ref.id)
                        suspects.append(job)
                    in_flight.clear()
                    active.discard()
                    rebuilds += 1
                    if rebuilds > MAX_POOL_REBUILDS:
                        for job in (*suspects, *queue):
                            yield lost_outcome(
                                job, "not fitted: worker processes kept dying during this run"
                            )
                        return
                    continue

                now = time.monotonic()
                expired = [
                    (future, job)
                    for future, (job, started) in in_flight.items()
                    if now - started > deadline_of(job)
                ]
                if expired:
                    kills += len(expired)
                    stopped = {job.ref.id for _, job in expired}
                    for _, job in expired:
                        yield lost_outcome(
                            job,
                            f"stopped by the watchdog after {deadline_of(job):.0f} s (twice the "
                            f"time limit of {job.time_limit:g} s plus {WATCHDOG_BASE_SECONDS:g} "
                            "s): the worker did not come back and was terminated",
                            kind="timed_out",
                        )
                    for job, _started in in_flight.values():
                        if job.ref.id not in stopped:
                            queue.appendleft(job)  # an innocent bystander: run it again
                    in_flight.clear()
                    active.discard()
                    if kills > MAX_WATCHDOG_KILLS:
                        for job in (*suspects, *queue):
                            yield lost_outcome(
                                job, "not fitted: too many candidates had to be stopped"
                            )
                        return
            finished = True
        finally:
            if own:
                if finished:
                    active.shutdown()
                else:
                    active.discard()
            elif not finished:  # workers may be busy or hung: do not leave them in the pool
                active.discard()
