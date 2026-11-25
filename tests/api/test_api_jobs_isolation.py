"""Per-job processes: one tenant cannot disturb another's jobs; fair scheduling; limits."""

import multiprocessing
import time
import uuid
from collections.abc import Callable
from typing import Any

import api_job_workers
import pytest
from fastapi.testclient import TestClient

from conftest import linear_csv, signup, upload
from curvestudio_api.jobs import FairQueue, Task

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}


def submit(client: TestClient, headers: dict[str, str], dataset_id: str) -> str:
    response = client.post(f"{API}/datasets/{dataset_id}/fits", headers=headers, json=POLY1)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def get_job(client: TestClient, headers: dict[str, str], job_id: str) -> dict[str, Any]:
    return dict(client.get(f"{API}/jobs/{job_id}", headers=headers).json())


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
        job = get_job(client, headers, job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stayed {job.get('status')!r}, wanted one of {statuses}")


def test_an_attacker_cancelling_every_second_cannot_disturb_a_victims_job(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_workers=2), job_worker=api_job_workers.victim_or_sleeper_worker
    )
    victim = signup(client)
    attacker = signup(client)
    victim_data = upload(client, victim, linear_csv(header="victim,y")).json()
    attacker_data = upload(client, attacker, linear_csv()).json()

    victim_job = submit(client, victim, victim_data["id"])
    first = wait_for(client, victim, victim_job, ("running",))
    for _ in range(5):
        mine = submit(client, attacker, attacker_data["id"])
        time.sleep(1)
        assert client.post(f"{API}/jobs/{mine}/cancel", headers=attacker).status_code == 200
    done = wait_for(client, victim, victim_job, ("succeeded", "failed", "cancelled"), timeout=60)
    assert done["status"] == "succeeded", done
    assert done["started_at"] == first["started_at"]  # never restarted


def test_cancel_kills_only_that_jobs_process(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_workers=2), job_worker=api_job_workers.sleeping_worker
    )
    a, b = signup(client), signup(client)
    data_a = upload(client, a, linear_csv()).json()
    data_b = upload(client, b, linear_csv()).json()
    job_a = submit(client, a, data_a["id"])
    job_b = submit(client, b, data_b["id"])
    wait_for(client, a, job_a, ("running",))
    wait_for(client, b, job_b, ("running",))
    assert client.post(f"{API}/jobs/{job_a}/cancel", headers=a).json()["status"] == "cancelled"
    time.sleep(1.5)
    assert get_job(client, b, job_b)["status"] == "running"
    # only warm workers remain — job_a's forked child was killed and reaped by its worker
    # (one worker if the cancel beat the child's hello and replaced it, two if not)
    assert {p.name for p in multiprocessing.active_children()} <= {"curvestudio-job-worker"}


def test_no_zombie_processes_after_timeouts_crashes_and_cancels(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_timeout_s=0.5, job_grace_s=0.5, job_workers=2),
        job_worker=api_job_workers.sleeping_worker,
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    jobs = [submit(client, headers, dataset["id"]) for _ in range(3)]
    for job_id in jobs:
        assert wait_for(client, headers, job_id, ("failed",))["error_code"] == "TIMEOUT"
    # a timed-out job recycles its worker, so every process exits on its own; nothing is
    # left unreaped (a live warm worker would be named, a zombie would not answer at all)
    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert multiprocessing.active_children() == []


def test_the_global_queue_is_bounded(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_queue_max=3, job_workers=1), job_worker=api_job_workers.sleeping_worker
    )
    users = [signup(client) for _ in range(5)]
    datasets = [upload(client, h, linear_csv()).json() for h in users]
    first = submit(client, users[0], datasets[0]["id"])
    wait_for(client, users[0], first, ("running",))  # now only waiting jobs count
    for h, d in zip(users[1:4], datasets[1:4], strict=True):
        submit(client, h, d["id"])
    response = client.post(f"{API}/datasets/{datasets[4]['id']}/fits", headers=users[4], json=POLY1)
    assert response.status_code == 503
    assert response.json()["error"] == "QUEUE_FULL"
    assert response.json()["detail"]["code"] == "QUEUE_FULL"


def test_an_oversized_result_fails_the_job_not_the_server(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_max_result_bytes=1000), job_worker=api_job_workers.big_result_worker
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    job = wait_for(client, headers, submit(client, headers, dataset["id"]), ("failed", "succeeded"))
    assert job["status"] == "failed"
    assert job["error_code"] == "RESULT_TOO_LARGE"
    assert client.get(f"{API}/health").status_code == 200


def task(user: uuid.UUID, n: int) -> Task:
    return Task(job_id=uuid.UUID(int=n), user_id=user, dataset_id=uuid.UUID(int=0))


def test_fair_queue_round_robins_across_users() -> None:
    a, b, c = uuid.UUID(int=101), uuid.UUID(int=102), uuid.UUID(int=103)
    queue = FairQueue()
    for n in (1, 2, 3, 4):
        queue.push(task(a, n))
    queue.push(task(b, 10))
    queue.push(task(c, 20))
    queue.push(task(b, 11))
    order = [queue.pop().job_id.int for _ in range(len(queue))]
    assert order == [1, 10, 20, 2, 11, 3, 4]  # one per user in turn, FIFO within a user
    assert len(queue) == 0
    with pytest.raises(IndexError):
        queue.pop()


def test_fair_queue_remove_and_size() -> None:
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    queue = FairQueue()
    tasks = [task(a, 1), task(b, 2), task(a, 3)]
    for t in tasks:
        queue.push(t)
    assert len(queue) == 3
    assert queue.remove(tasks[0].job_id) is True
    assert queue.remove(tasks[0].job_id) is False
    assert [queue.pop().job_id.int for _ in range(2)] == [3, 2]  # user a still has its turn
