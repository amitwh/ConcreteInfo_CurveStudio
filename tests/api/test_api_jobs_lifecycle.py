"""Job lifecycle with injected workers: cancel, timeout, crash recovery, restart recovery."""

import multiprocessing
import time
import uuid
from collections.abc import Callable
from typing import Any

import api_job_workers
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from conftest import linear_csv, signup, upload
from curvestudio_api.app import create_app
from curvestudio_api.migrations import upgrade_to_head
from curvestudio_api.orm import Dataset, Job, User
from curvestudio_api.timeutil import utcnow

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}


def submit(client: TestClient, headers: dict[str, str], dataset_id: str) -> str:
    response = client.post(f"{API}/datasets/{dataset_id}/fits", headers=headers, json=POLY1)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def wait_for(
    client: TestClient,
    headers: dict[str, str],
    job_id: str,
    statuses: tuple[str, ...],
    timeout: float = 60,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}", headers=headers).json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stayed {job.get('status')!r}, wanted one of {statuses}")


def test_cancel_a_running_job_kills_its_worker_and_frees_the_pool(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(job_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    first = submit(client, headers, dataset["id"])
    wait_for(client, headers, first, ("running",))

    cancelled = client.post(f"{API}/jobs/{first}/cancel", headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["finished_at"]
    assert (
        client.post(f"{API}/jobs/{first}/cancel", headers=headers).status_code == 200
    )  # idempotent

    # the single worker is free again: the next job is dispatched
    second = submit(client, headers, dataset["id"])
    wait_for(client, headers, second, ("running",))
    assert client.get(f"{API}/jobs/{first}", headers=headers).json()["status"] == "cancelled"


def test_cancel_a_queued_job_leaves_the_running_one_alone(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(job_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    running = submit(client, headers, dataset["id"])
    queued = submit(client, headers, dataset["id"])
    wait_for(client, headers, running, ("running",))
    assert client.get(f"{API}/jobs/{queued}", headers=headers).json()["status"] == "queued"
    assert (
        client.post(f"{API}/jobs/{queued}/cancel", headers=headers).json()["status"] == "cancelled"
    )
    time.sleep(0.5)
    assert client.get(f"{API}/jobs/{running}", headers=headers).json()["status"] == "running"
    assert client.get(f"{API}/jobs/{queued}", headers=headers).json()["status"] == "cancelled"


def test_cancelling_a_finished_job_is_a_409(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory()
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    job_id = submit(client, headers, dataset["id"])
    assert wait_for(client, headers, job_id, ("succeeded", "failed"))["status"] == "succeeded"
    response = client.post(f"{API}/jobs/{job_id}/cancel", headers=headers)
    assert response.status_code == 409
    assert response.json()["error"] == "CONFLICT"
    assert client.get(f"{API}/jobs/{job_id}", headers=headers).json()["status"] == "succeeded"


def test_a_stuck_job_times_out_and_its_worker_is_killed(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_timeout_s=0.5, job_grace_s=0.5),
        job_worker=api_job_workers.sleeping_worker,
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    first = submit(client, headers, dataset["id"])
    job = wait_for(client, headers, first, ("failed",))
    assert job["error_code"] == "TIMEOUT"
    assert "0.5" in job["error_message"]
    # the pool was recycled: the next job runs (and times out) too, instead of waiting forever
    second = submit(client, headers, dataset["id"])
    assert wait_for(client, headers, second, ("failed",))["error_code"] == "TIMEOUT"
    # no worker process outlives the recycle
    deadline = time.monotonic() + 10
    while len(multiprocessing.active_children()) > 1 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert len(multiprocessing.active_children()) <= 1


def test_a_crashed_worker_fails_the_job_and_the_pool_is_rebuilt(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(job_worker=api_job_workers.crashing_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    first = submit(client, headers, dataset["id"])
    job = wait_for(client, headers, first, ("failed",))
    assert job["error_code"] == "WORKER_CRASHED"
    assert job["error_message"]
    # a rebuilt pool takes the next job (which crashes too, but is dispatched and reported)
    second = submit(client, headers, dataset["id"])
    assert wait_for(client, headers, second, ("failed",))["error_code"] == "WORKER_CRASHED"
    assert client.get(f"{API}/health").status_code == 200  # the API itself is fine


def test_a_crash_does_not_take_innocent_jobs_down(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_workers=2), job_worker=api_job_workers.crash_on_boom_worker
    )
    headers = signup(client)
    good = upload(client, headers, linear_csv()).json()
    bad = upload(client, headers, linear_csv(header="boom,y")).json()
    good_job = submit(client, headers, good["id"])
    bad_job = submit(client, headers, bad["id"])
    survivor = wait_for(client, headers, good_job, ("succeeded", "failed"), timeout=120)
    culprit = wait_for(client, headers, bad_job, ("succeeded", "failed"), timeout=120)
    assert culprit["status"] == "failed"
    assert culprit["error_code"] == "WORKER_CRASHED"
    assert survivor["status"] == "succeeded", survivor
    assert client.get(f"{API}/fits/{survivor['result_ref']}", headers=headers).status_code == 200


def test_deleting_a_dataset_stops_its_running_job(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(job_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    doomed = upload(client, headers, linear_csv()).json()
    other = upload(client, headers, linear_csv()).json()
    running = submit(client, headers, doomed["id"])
    wait_for(client, headers, running, ("running",))
    assert client.delete(f"{API}/datasets/{doomed['id']}", headers=headers).status_code == 204
    assert client.get(f"{API}/jobs/{running}", headers=headers).status_code == 404
    follow_up = submit(client, headers, other["id"])
    wait_for(client, headers, follow_up, ("running",))


def test_a_user_cannot_queue_unbounded_jobs(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory(job_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    for _ in range(20):
        submit(client, headers, dataset["id"])
    response = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=POLY1)
    assert response.status_code == 429
    assert response.json()["error"] == "RATE_LIMITED"
    assert response.json()["detail"]["code"] == "TOO_MANY_JOBS"
    # another user is not affected
    other = signup(client)
    mine = upload(client, other, linear_csv()).json()
    submit(client, other, mine["id"])


def test_jobs_left_over_from_a_previous_run_are_failed_at_startup(
    settings_factory: Callable[..., Any],
) -> None:
    settings = settings_factory()
    app = create_app(settings)
    upgrade_to_head(app.state.ctx.engine)
    user_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    ids = {name: uuid.uuid4() for name in ("queued", "running", "succeeded", "failed")}
    with app.state.ctx.session_factory() as db:
        db.add(User(id=user_id, email="old@example.org", password_hash="x", name=""))
        db.flush()
        db.add(
            Dataset(
                id=dataset_id,
                user_id=user_id,
                name="d",
                n_rows=3,
                columns=["x", "y"],
                x_column="x",
                y_column="y",
                fingerprint="f",
                payload=b"x,y\n1,2\n",
            )
        )
        db.flush()
        for status, job_id in ids.items():
            db.add(
                Job(
                    id=job_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    kind="fit",
                    status=status,
                    spec={},
                    started_at=utcnow() if status == "running" else None,
                )
            )
        db.commit()
    with TestClient(app):
        pass
    with app.state.ctx.session_factory() as db:
        rows = {j.id: j for j in db.scalars(select(Job))}
    for name in ("queued", "running"):
        job = rows[ids[name]]
        assert job.status == "failed", name
        assert job.error_code == "SERVER_RESTART"
        assert job.finished_at is not None
    assert rows[ids["succeeded"]].status == "succeeded"
    assert rows[ids["failed"]].status == "failed"
    assert rows[ids["failed"]].error_code is None


@pytest.mark.parametrize("workers", [1, 2])
def test_queued_jobs_run_in_order_within_the_worker_limit(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any], workers: int
) -> None:
    client = client_factory(
        settings_factory(job_workers=workers), job_worker=api_job_workers.slow_worker
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    jobs = [submit(client, headers, dataset["id"]) for _ in range(3)]
    deadline = time.monotonic() + 20
    peak = 0
    while time.monotonic() < deadline:
        states = [client.get(f"{API}/jobs/{j}", headers=headers).json()["status"] for j in jobs]
        peak = max(peak, states.count("running"))
        if all(s == "succeeded" for s in states):
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"jobs did not finish: {states}")
    assert peak <= workers
