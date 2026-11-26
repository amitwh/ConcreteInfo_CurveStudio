"""The warm job-worker pool: isolation proven, not promised (P6 Task 5).

A warm worker forks a fresh child per job and never runs job code itself. These tests hold
that to the letter of the plan: two different jobs through one worker must equal the same
jobs in fresh processes byte for byte; a timeout must kill the job and replace the worker;
a cancel must leave nothing behind for the next job in the same worker; a job that dies the
way the OOM killer dies must not poison the pool; and the recycle limits must replace a
worker transparently.
"""

import json
import multiprocessing
import os
import time
import uuid
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import api_job_workers
from api_test_helpers import linear_csv, michaelis_csv, signup, upload
from fastapi.testclient import TestClient

from curvestudio_api.audit_worker import run_cross_validation_job
from curvestudio_api.orm import Dataset, Fit, Job
from curvestudio_api.worker import run_fit_job

API = "/api/v1"
MICHAELIS = {"model": {"kind": "registered", "id": "michaelis_menten"}}
POLY2 = {"model": {"kind": "polynomial", "degree": 2}}
TEST_DIR = "CURVESTUDIO_WARM_POOL_TEST_DIR"


def wait_for(
    client: TestClient, headers: dict[str, str], job_id: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    deadline = time.monotonic() + 90
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = dict(client.get(f"{API}/jobs/{job_id}", headers=headers).json())
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stayed {job.get('status')!r}, wanted one of {statuses}")


def submit_fit(client: TestClient, headers: dict[str, str], dataset_id: str, spec: dict) -> str:
    response = client.post(f"{API}/datasets/{dataset_id}/fits", headers=headers, json=spec)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def canonical(payload: Any) -> str:
    """Byte comparison of a job's computation. The one field blanked is the bundle's
    ``result.provenance.created_at`` — the wall-clock moment the fit was computed, which a
    fresh process seconds later differs in by design. Everything else must match exactly."""
    import copy

    normalised = copy.deepcopy(payload)
    provenance = normalised.get("bundle", {}).get("result", {}).get("provenance")
    if isinstance(provenance, dict) and "created_at" in provenance:
        provenance["created_at"] = "<when>"
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fresh_run(
    connection: Connection, worker: Callable[..., Any], arguments: dict[str, Any]
) -> None:
    """In a freshly spawned process: run ``worker`` and send its JSON result back."""
    import json

    try:
        result = worker(**arguments)
    except BaseException as exc:
        result = {"ok": False, "code": type(exc).__name__, "message": str(exc)}
    connection.send_bytes(json.dumps(result, allow_nan=False).encode())


def fresh_process_result(worker: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """What one job's worker produces in a fresh ``spawn`` process — the reference every
    warm-pool result is compared against."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_fresh_run, args=(sender, worker, arguments), daemon=True)
    try:
        process.start()
    finally:
        sender.close()
    data = receiver.recv_bytes()
    process.join(30)
    process.close()
    receiver.close()
    return dict(json.loads(data))


def job_arguments(client: TestClient, job_id: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    """The worker function and keyword arguments the manager handed the job, rebuilt from
    the stored rows (the same shape ``JobManager._arguments_for`` loads)."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        job = db.get(Job, uuid.UUID(job_id))
        dataset = db.get(Dataset, job.dataset_id)  # type: ignore[arg-type]
        assert job is not None and dataset is not None
        arguments: dict[str, Any] = {
            "payload": dataset.payload,
            "x_name": dataset.x_column,
            "y_name": dataset.y_column,
            "fingerprint": dataset.fingerprint,
            "spec": job.spec,
        }
        if job.kind == "audit":
            fit = db.get(Fit, uuid.UUID(job.spec["fit_id"]))
            assert fit is not None
            return run_cross_validation_job, {
                **arguments,
                "bundle": fit.bundle,
                "fit_spec": fit.spec,
            }
        return run_fit_job, arguments


def stored_result(client: TestClient, job: dict[str, Any]) -> Any:
    """The result a succeeded job stored: the fit's bundle, or the cross-validation (stored
    on the fit the job's spec names — an audit job's ``result_ref`` is its dataset)."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        if job["kind"] == "audit":
            fit = db.get(Fit, uuid.UUID(job["spec"]["fit_id"]))
            assert fit is not None and fit.cv is not None
            return {"ok": True, "cv": fit.cv}
        fit = db.get(Fit, uuid.UUID(str(job["result_ref"])))
        assert fit is not None
        return {"ok": True, "bundle": fit.bundle}


def manager_of(client: TestClient) -> Any:
    return client.app.state.ctx.jobs  # type: ignore[attr-defined]


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat:
            fields = stat.read()
        return fields[fields.rindex(")") + 2 :].split()[0] != "Z"
    except OSError:
        return False


# -- isolation: one worker, two different jobs -------------------------------------


def test_two_different_jobs_in_one_worker_match_fresh_processes(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()  # job_workers=1: both jobs share ONE warm worker
    headers = signup(client)
    dataset = upload(client, headers, michaelis_csv(40)).json()

    fit_job = wait_for(
        client, headers, submit_fit(client, headers, dataset["id"], MICHAELIS), ("succeeded",)
    )
    audit = client.post(
        f"{API}/fits/{fit_job['result_ref']}/audit/cross-validation", headers=headers, json={}
    )
    assert audit.status_code == 202, audit.text
    cv_job = wait_for(client, headers, str(audit.json()["job_id"]), ("succeeded",))

    assert manager_of(client).spawned_workers == 1, "both jobs must have run through one worker"

    for job in (fit_job, cv_job):
        worker, arguments = job_arguments(client, str(job["id"]))
        assert canonical(fresh_process_result(worker, arguments)) == canonical(
            stored_result(client, job)
        ), f"the warm worker's {job['kind']} result differs from a fresh process's"


# -- the kill guarantees ----------------------------------------------------------


def test_a_timed_out_job_is_killed_and_the_pool_replaces_the_worker(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(job_timeout_s=0.5, job_grace_s=0.5),
        job_worker=api_job_workers.hang_or_fit_worker,
    )
    headers = signup(client)
    stuck = upload(client, headers, linear_csv(header="sleepy,y")).json()
    sane = upload(client, headers, michaelis_csv()).json()
    first = wait_for(client, headers, submit_fit(client, headers, stuck["id"], POLY2), ("failed",))
    assert first["error_code"] == "TIMEOUT"

    # the stopped worker is replaced; the next job runs clean and its result still equals
    # a fresh process's
    second = wait_for(
        client, headers, submit_fit(client, headers, sane["id"], MICHAELIS), ("succeeded",)
    )
    worker, arguments = job_arguments(client, str(second["id"]))
    assert canonical(fresh_process_result(worker, arguments)) == canonical(
        stored_result(client, second)
    )
    assert manager_of(client).spawned_workers == 2, "the timed-out job's worker was not replaced"


def test_a_cancelled_job_leaves_nothing_behind(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Any],
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv(TEST_DIR, str(tmp_path))
    before = os.getcwd()
    client = client_factory(job_worker=api_job_workers.dirty_or_probe_worker)
    headers = signup(client)
    dirty_set = upload(client, headers, linear_csv(header="dirty,y")).json()
    probe_set = upload(client, headers, linear_csv(header="probe,y")).json()
    dirty = submit_fit(client, headers, dirty_set["id"], POLY2)

    marker = tmp_path / "grandchild.pid"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists(), "the dirty job never ran"

    cancelled = client.post(f"{API}/jobs/{dirty}/cancel", headers=headers)
    assert cancelled.json()["status"] == "cancelled"
    grandchild = int(marker.read_text())
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _alive(grandchild):
        time.sleep(0.1)
    assert not _alive(grandchild), "the cancelled job's grandchild survived the kill"

    # the next job runs in the SAME worker and sees exactly what a fresh process sees
    probe = wait_for(
        client, headers, submit_fit(client, headers, probe_set["id"], POLY2), ("succeeded",)
    )
    assert manager_of(client).spawned_workers == 1, "a cancel must not cost the worker"
    after_cancel = stored_result(client, probe)["bundle"]
    fresh = fresh_process_result(api_job_workers.dirty_or_probe_worker, {"x_name": "probe"})[
        "bundle"
    ]
    assert canonical(after_cancel) == canonical(fresh)
    assert after_cancel["leak"] is None  # no environment carried over
    assert after_cancel["scratch"] == "clean"  # no globals carried over
    assert after_cancel["cwd"] == fresh["cwd"] == before  # no working-directory carry-over


def test_cancelling_before_the_hello_is_read_still_kills_the_child(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Any],
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """The racy window: the job's child has started (its grandchild exists) but the
    scheduler has not read the child's hello yet. The manager's lock is held so the
    scheduler cannot tick, the hello sits unread in the pipe, and the cancel must still
    find it — killing the child's whole group — and keep the worker."""
    monkeypatch.setenv(TEST_DIR, str(tmp_path))
    client = client_factory(job_worker=api_job_workers.dirty_or_probe_worker)
    headers = signup(client)
    dirty_set = upload(client, headers, linear_csv(header="dirty,y")).json()
    probe_set = upload(client, headers, linear_csv(header="probe,y")).json()
    manager = manager_of(client)
    dirty = submit_fit(client, headers, dirty_set["id"], POLY2)

    marker = tmp_path / "grandchild.pid"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists()
    grandchild = int(marker.read_text())
    assert _alive(grandchild)
    with manager._lock:  # type: ignore[attr-defined]  # the scheduler cannot read the hello
        time.sleep(0.3)  # the hello is written; the tick that would read it is blocked
        assert manager.cancel(uuid.UUID(dirty)) == "cancelled"

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _alive(grandchild):
        time.sleep(0.1)
    assert not _alive(grandchild), "the child's group was not killed from its own hello"
    assert manager.spawned_workers == 1, "the pre-hello cancel must not cost the worker"
    probe = wait_for(
        client, headers, submit_fit(client, headers, probe_set["id"], POLY2), ("succeeded",)
    )
    assert canonical(stored_result(client, probe)["bundle"]) == canonical(
        fresh_process_result(api_job_workers.dirty_or_probe_worker, {"x_name": "probe"})["bundle"]
    )


def test_a_job_killed_for_memory_does_not_poison_the_pool(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(job_worker=api_job_workers.bomb_or_fit_worker)
    headers = signup(client)
    doomed = upload(client, headers, linear_csv(header="boom,y")).json()
    sane = upload(client, headers, michaelis_csv()).json()
    bombed = wait_for(
        client, headers, submit_fit(client, headers, doomed["id"], POLY2), ("failed",)
    )
    assert bombed["error_code"] == "WORKER_CRASHED"

    # the same worker takes the next job, which succeeds with a fresh process's result
    second = wait_for(
        client, headers, submit_fit(client, headers, sane["id"], MICHAELIS), ("succeeded",)
    )
    assert manager_of(client).spawned_workers == 1, "a job's death must not cost its worker"
    worker, arguments = job_arguments(client, str(second["id"]))
    assert canonical(fresh_process_result(worker, arguments)) == canonical(
        stored_result(client, second)
    )


# -- recycling ---------------------------------------------------------------------


def test_a_worker_is_replaced_after_its_job_count_limit(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(job_worker_max_jobs=1))
    headers = signup(client)
    dataset = upload(client, headers, michaelis_csv()).json()
    for _ in range(2):
        job = wait_for(
            client, headers, submit_fit(client, headers, dataset["id"], POLY2), ("succeeded",)
        )
        assert job["error_code"] is None
    assert manager_of(client).spawned_workers == 2, "the worker was not recycled after one job"
    assert len(multiprocessing.active_children()) <= 1  # the retired worker is gone


def test_a_worker_is_replaced_above_its_memory_limit(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    # 64 MB is below any warm worker's post-import size, so every job recycles it
    client = client_factory(settings_factory(job_worker_memory_mb=64))
    headers = signup(client)
    dataset = upload(client, headers, michaelis_csv()).json()
    for _ in range(2):
        job = wait_for(
            client, headers, submit_fit(client, headers, dataset["id"], POLY2), ("succeeded",)
        )
        assert job["error_code"] is None
    assert manager_of(client).spawned_workers == 2
