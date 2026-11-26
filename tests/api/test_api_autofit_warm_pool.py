"""The warm Auto-fit pool of a job process (P6 Task 6, half A).

Every Auto-fit job runs in a fresh child of a warm job worker, so its ONE pool starts its
workers with ``fork`` out of that already-imported memory instead of spawning interpreters
per run. These tests hold the plan's promises: two different runs through the warm path
equal the same runs with fresh per-run pools (rank order and the stored numbers); a pool
recycled at its run-count limit is replaced and the next run succeeds; a killed pool worker
is replaced by the engine and never surfaces as a run failure.
"""

import os
import signal
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import linear_csv, michaelis_csv, signup, upload
from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, CancelToken, autofit
from curvestudio.discovery.pool import AutoFitPool
from curvestudio_api import autofit_pool, autofit_worker
from curvestudio_api.orm import Autofit, Dataset, Job
from curvestudio_api.settings import Settings
from curvestudio_api.storage import decode_payload, encode_payload

API = "/api/v1"
LINUX = Path("/proc/self/stat").exists()
"""The fork-started pool is Linux; elsewhere the holder spawns exactly as before."""

pytestmark = pytest.mark.real_plugins
"""The in-process references must see the same model catalogue as the spawned job workers
(the shared fixtures hide the installed entry points in the parent process otherwise)."""

RUN_BODY: dict[str, Any] = {
    "families": ["polynomial", "exponential"],
    "max_parameters": 5,
    "n_starts": 1,
}
"""Enough candidates that the engine uses the pool, few enough to stay quick."""


@pytest.fixture(autouse=True)
def _no_pools_left_behind() -> Any:
    yield
    autofit_pool.shutdown_job_pools()


def wait_finished(
    client: TestClient, headers: dict[str, str], job_id: str, timeout: float = 120
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = dict(client.get(f"{API}/jobs/{job_id}", headers=headers).json())
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stayed {job.get('status')!r}")


def stored_ranking(client: TestClient, job: dict[str, Any]) -> list[tuple[str, float | None]]:
    """The stored run's rank order with each row's AICc."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        row = db.get(Autofit, uuid.UUID(str(job["result_ref"])))
        assert row is not None
        return [
            (entry["model_id"], entry["metrics"]["aicc"])
            if entry["metrics"]
            else (entry["model_id"], None)
            for entry in row.result["ranked"]
        ]


def fresh_reference(client: TestClient, job_id: str) -> list[tuple[str, float | None]]:
    """The same run with a fresh per-run pool (the engine's default, no ``pool=``)."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        job = db.get(Job, uuid.UUID(job_id))
        dataset = db.get(Dataset, job.dataset_id)  # type: ignore[arg-defined]
        assert job is not None and dataset is not None
        xy = decode_payload(
            dataset.payload,
            x_name=dataset.x_column,
            y_name=dataset.y_column,
            fingerprint=dataset.fingerprint,
        )
        config = AutoFitConfig.model_validate(job.spec["config"])
    result = autofit(xy, config, cancel=CancelToken())
    return [(row.model_id, row.metrics.aicc if row.metrics else None) for row in result.ranked]


def tiny_xy() -> XY:
    x = np.linspace(1.0, 28.0, 20)
    return XY(
        x=x,
        y=10.0 * np.exp(-0.15 * x) + 2.0,
        weights=None,
        x_name="age_days",
        y_name="strength_mpa",
        fingerprint="warm-pool-tiny",
    )


def tiny_config() -> AutoFitConfig:
    return AutoFitConfig(
        families=("exponential",),
        n_starts=2,
        time_limit=10.0,
        cv_folds=None,
        workers=2,
        curve_top_n=0,
        staged_starts=False,
    )


def tiny_run(pool: AutoFitPool) -> list[tuple[str, float | None]]:
    result = autofit(tiny_xy(), tiny_config(), cancel=CancelToken(), pool=pool)
    return [(row.model_id, row.metrics.aicc if row.metrics else None) for row in result.ranked]


# -- the warm path equals fresh per-run pools -----------------------------------------------


def test_two_different_runs_through_the_warm_path_match_fresh_pools(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(autofit_workers=2))
    headers = signup(client)
    for csv in (linear_csv(40), michaelis_csv(40)):
        dataset = upload(client, headers, csv).json()
        response = client.post(
            f"{API}/datasets/{dataset['id']}/autofit", headers=headers, json=RUN_BODY
        )
        assert response.status_code == 202, response.text
        job_id = str(response.json()["job_id"])
        job = wait_finished(client, headers, job_id)
        assert job["status"] == "succeeded", job
        assert stored_ranking(client, job) == fresh_reference(client, job_id)


def test_the_job_passes_the_process_pool_to_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_autofit_job`` hands the engine this process's ONE warm pool, not a temporary."""
    seen: dict[str, Any] = {}
    real = autofit_worker.autofit

    def spy(data: XY, config: AutoFitConfig, **kwargs: Any):
        seen["pool"] = kwargs.get("pool")
        return real(data, config, **kwargs)

    monkeypatch.setattr(autofit_worker, "autofit", spy)
    result = autofit_worker.run_autofit_job(
        payload=encode_payload(tiny_xy()),
        x_name="age_days",
        y_name="strength_mpa",
        fingerprint="warm-pool-tiny",
        spec={"config": tiny_config().model_dump(mode="json"), "budget_s": 60},
        report=lambda _snapshot: None,
    )
    assert result["ok"], result
    pool = seen["pool"]
    assert isinstance(pool, AutoFitPool)
    assert pool.workers == 2
    assert pool is autofit_pool.job_autofit_pool(2)
    if LINUX:
        assert pool.start_method == "fork"


# -- recycle and replacement ------------------------------------------------------------------


def test_one_pool_per_count_is_reused_until_its_run_limit_replaces_it() -> None:
    first = autofit_pool.job_autofit_pool(2, max_runs=10)
    assert autofit_pool.job_autofit_pool(2, max_runs=10) is first  # the ONE pool
    assert tiny_run(first)
    second = autofit_pool.job_autofit_pool(2, max_runs=1)  # first served its one run
    assert second is not first
    assert not first.running  # replaced cleanly, not left to leak workers
    assert tiny_run(second) == tiny_run(autofit_pool.job_autofit_pool(2, max_runs=1))


def test_a_pool_above_its_memory_limit_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = autofit_pool.job_autofit_pool(2, max_runs=100)
    assert autofit_pool.job_autofit_pool(2, max_runs=100) is pool
    monkeypatch.setattr(autofit_pool, "_total_rss_mb", lambda _pool: 4096)
    assert autofit_pool.job_autofit_pool(2, max_runs=100) is not pool
    assert not pool.running


def test_a_killed_pool_worker_is_replaced_and_the_run_still_succeeds() -> None:
    pool = autofit_pool.job_autofit_pool(2)
    before = tiny_run(pool)
    workers = pool.worker_pids()
    if not LINUX or not workers:  # spawn pools replace workers their own way; still must run
        assert tiny_run(pool) == before
        return
    os.kill(workers[0], signal.SIGKILL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and pool.worker_pids() == workers:
        time.sleep(0.05)
    assert tiny_run(pool) == before  # not a run failure: the engine rebuilt the pool
