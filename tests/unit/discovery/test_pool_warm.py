"""The Auto-fit pool's start method and pre-warm hook (P6 Task 6).

A pool started with ``start_method="fork"`` (a long-lived process that has already imported
the fitting stack — the API's warm job workers) creates its workers without paying the
interpreter and import cost again, and gives the same results the default ``spawn`` pool
gives. :meth:`AutoFitPool.prewarm` starts every worker now, so the first run pays nothing.
"""

import sys

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, CancelToken, autofit
from curvestudio.discovery.pool import AutoFitPool
from curvestudio.discovery.worker import run_candidate

FORK = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the fork start method is POSIX"
)


def tiny_xy() -> XY:
    x = np.linspace(1.0, 28.0, 20)
    return XY(
        x=x,
        y=10.0 * np.exp(-0.15 * x) + 2.0,
        weights=None,
        x_name="age_days",
        y_name="strength_mpa",
        fingerprint="pool-warm",
    )


def config() -> AutoFitConfig:
    return AutoFitConfig(
        families=("exponential",),
        n_starts=2,
        time_limit=10.0,
        cv_folds=None,
        workers=2,
        curve_top_n=0,
        staged_starts=False,
    )


def ranked_ids(result) -> list[str]:
    return [row.model_id for row in result.ranked]


def metrics(result) -> list[tuple[str, float]]:
    return [(row.model_id, row.metrics.aicc) for row in result.ranked if row.metrics]


def test_the_default_pool_still_spawns() -> None:
    pool = AutoFitPool(2)
    assert pool.start_method == "spawn"


def test_an_unknown_start_method_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        AutoFitPool(2, start_method="no-such-method")


def test_prewarm_starts_every_worker_without_a_run() -> None:
    pool = AutoFitPool(2)
    try:
        pool.prewarm()
        assert len(pool.worker_pids()) == 2
    finally:
        pool.shutdown()


@FORK
def test_a_fork_pool_runs_candidates_and_matches_spawn() -> None:
    one, two = tiny_xy(), tiny_xy()
    spawn_pool = AutoFitPool(2)
    fork_pool = AutoFitPool(2, start_method="fork")
    try:
        from_spawn = autofit(one, config(), cancel=CancelToken(), pool=spawn_pool)
        from_fork = autofit(two, config(), cancel=CancelToken(), pool=fork_pool)
    finally:
        spawn_pool.shutdown()
        fork_pool.shutdown()
    assert ranked_ids(from_fork) == ranked_ids(from_spawn)
    assert metrics(from_fork) == metrics(from_spawn)


@FORK
def test_a_fork_pool_round_trips_a_job_and_keeps_its_workers() -> None:
    pool = AutoFitPool(2, start_method="fork")
    try:
        pool.prewarm()
        pids = pool.worker_pids()
        future = pool.executor().submit(run_candidate, None)
        # a None job fails before run_candidate's own try block: the point is that the
        # exception comes back through the pool instead of hanging or killing it
        with pytest.raises(AttributeError):
            future.result(timeout=30)
        assert pool.worker_pids() == pids  # the same warm workers
    finally:
        pool.shutdown()
