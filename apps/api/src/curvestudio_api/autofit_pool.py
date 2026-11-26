"""One warm Auto-fit pool per job process: fork-started workers, recycled by count and memory.

Under the warm job workers (Task 5) every Auto-fit job runs in a fresh child forked from a
worker that has already imported the whole fitting stack — and a ``ProcessPoolExecutor``
cannot cross that fork: its result futures are set only by the management thread of the
process that created the executor, so a child inheriting a live one would wait forever. The
warmth the desktop gets from a shared pool in a long-lived process is therefore delivered
one level down: this process's ONE pool starts its workers with ``fork`` (Linux; ``spawn``
elsewhere, where the gain would not repay the fork-safety caveats), so they fork from the
already-imported child in milliseconds instead of every run paying an interpreter start plus
the NumPy/SciPy/SymPy imports. The workers die with the job child — the pool initialiser's
``PR_SET_PDEATHSIG`` and the ``killpg`` the manager already issues for the job's process
group both reach them — so no pool outlives its run.

The holder recycles by the discipline ``worker_pool.py`` established for the job workers:
after ``max_runs`` runs served, or with the pool's workers together above
``max_total_rss_mb`` (read from ``/proc`` between runs), the pool is shut down and the next
run gets a fresh one — a run never fails for it. A worker that dies mid-run is replaced by
the engine itself (``stream_pool`` discards the broken pool and rebuilds one), never
surfaced to the job.
"""

import sys
import threading
from dataclasses import dataclass

from curvestudio.discovery.pool import AutoFitPool

__all__ = [
    "AUTOFIT_POOL_MAX_RUNS",
    "AUTOFIT_POOL_MAX_TOTAL_RSS_MB",
    "WARM_AUTOFIT_START_METHOD",
    "job_autofit_pool",
    "shutdown_job_pools",
]

WARM_AUTOFIT_START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"
"""How a job process starts its pool workers. On Linux the job child forks them straight
out of its already-imported memory; elsewhere (Windows; macOS, where fork with libraries
loaded is not worth the risk) they spawn exactly as they always did."""

AUTOFIT_POOL_MAX_RUNS = 20
"""Runs a pool serves before it is replaced, mirroring the job workers' job-count recycle."""

AUTOFIT_POOL_MAX_TOTAL_RSS_MB = 1536
"""Resident MB the pool's workers may hold together before the pool is replaced."""


@dataclass
class _HeldPool:
    """One pool and how many runs it has served."""

    pool: AutoFitPool
    runs: int = 0


_POOLS: dict[int, _HeldPool] = {}
_POOLS_LOCK = threading.Lock()


def _total_rss_mb(pool: AutoFitPool) -> int:
    """Resident MB across the pool's live workers (0 where ``/proc`` cannot be read)."""
    total_kb = 0
    for pid in pool.worker_pids():
        try:
            with open(f"/proc/{pid}/status", encoding="ascii") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        total_kb += int(line.split()[1])
                        break
        except OSError, ValueError, IndexError:
            continue  # a worker that just died is the recycle decision's business, not a crash
    return total_kb // 1024


def job_autofit_pool(
    workers: int,
    *,
    max_runs: int = AUTOFIT_POOL_MAX_RUNS,
    max_total_rss_mb: int = AUTOFIT_POOL_MAX_TOTAL_RSS_MB,
) -> AutoFitPool:
    """The ONE warm Auto-fit pool of this process for ``workers`` workers.

    The pool is created on the first call at a count and reused for every later one; at
    ``max_runs`` runs served, or above ``max_total_rss_mb`` of resident memory across its
    workers, it is shut down and the next call builds a fresh one, so a run never inherits
    a spent pool. ``run_autofit_job`` is the caller; tests override the limits to exercise
    the recycle.
    """
    count = max(1, workers)
    with _POOLS_LOCK:
        held = _POOLS.get(count)
        over = held is not None and (
            held.runs >= max(1, max_runs)
            or (max_total_rss_mb > 0 and _total_rss_mb(held.pool) > max_total_rss_mb)
        )
        if held is None or over:
            if held is not None:
                held.pool.shutdown()  # between runs its workers are idle: this is quick
            held = _POOLS[count] = _HeldPool(
                AutoFitPool(count, start_method=WARM_AUTOFIT_START_METHOD)
            )
        held.runs += 1
        return held.pool


def shutdown_job_pools() -> None:
    """Shut down every pool this process holds. A job's exit needs no call — the pool's
    workers die with their job child — but a process that keeps running (the tests) does."""
    with _POOLS_LOCK:
        held = list(_POOLS.values())
        _POOLS.clear()
    for entry in held:
        entry.pool.shutdown()
