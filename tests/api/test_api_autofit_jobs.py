"""Auto-fit jobs: progress, cancel (no orphaned worker processes), the time budget, the hard
timeout, a parent that dies, and one realistic run on the whole catalogue (slow)."""

import json
import multiprocessing
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import api_job_workers
import numpy as np
import pytest
from api_test_helpers import linear_csv, signup, upload
from fastapi.testclient import TestClient

from curvestudio.data import XY
from curvestudio.discovery import AutoFitConfig, AutoFitResult, CandidateOutcome, RankBy
from curvestudio.workflow import FitBundle
from curvestudio_api import autofit_worker
from curvestudio_api.settings import Settings
from curvestudio_api.storage import encode_payload

API = "/api/v1"
POOL_RUN = {"families": ["polynomial"], "max_parameters": 6, "cv_folds": 3, "n_starts": 1}
"""A dozen candidates: more than the engine fits in-process, so it starts its own pool."""
LINUX_ONLY = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
HERE = Path(__file__).parent


def poll_job(client: TestClient, headers: dict[str, str], job_id: str) -> dict[str, Any]:
    return dict(client.get(f"{API}/jobs/{job_id}", headers=headers).json())


def wait_until(condition: Callable[[], Any], timeout: float, what: str) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def wait_finished(
    client: TestClient, headers: dict[str, str], job_id: str, timeout: float = 120
) -> dict[str, Any]:
    return dict(
        wait_until(
            lambda: (
                job
                if (job := poll_job(client, headers, job_id))["status"]
                in ("succeeded", "failed", "cancelled")
                else None
            ),
            timeout,
            "the job to finish",
        )
    )


def start(client: TestClient, headers: dict[str, str], dataset_id: str, **body: Any) -> str:
    response = client.post(f"{API}/datasets/{dataset_id}/autofit", headers=headers, json=body)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rindex(")") + 2 :].split()[0] != "Z"  # a zombie is dead, just not reaped


def group_members(pgid: int) -> list[int]:
    """Live processes of process group ``pgid``."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        state, _ppid, group = stat[stat.rindex(")") + 2 :].split()[:3]
        if int(group) == pgid and state != "Z":
            found.append(int(entry.name))
    return found


def job_process_pid(client: TestClient, job_id: str) -> int | None:
    """The pid of the process running the job: the child its warm worker forked."""
    return client.app.state.ctx.jobs.child_pid_of(uuid.UUID(job_id))  # type: ignore[attr-defined]


# -- progress --------------------------------------------------------------------------------------


def test_progress_is_visible_while_a_run_is_in_progress_and_kept_after(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(autofit_workers=2))
    headers = signup(client)
    dataset = upload(client, headers, linear_csv(40)).json()
    job_id = start(client, headers, dataset["id"], **POOL_RUN)

    seen: list[dict[str, Any]] = []

    def finished() -> dict[str, Any] | None:
        job = poll_job(client, headers, job_id)
        if job["status"] == "running" and job["progress"] is not None:
            seen.append(job["progress"])
        return job if job["status"] in ("succeeded", "failed", "cancelled") else None

    job = wait_until(finished, 120, "the run to finish")
    assert job["status"] == "succeeded", job
    assert seen, "no progress was visible while the job was running"
    assert all(p["total"] >= 5 for p in seen)
    assert [p["done"] for p in seen] == sorted(p["done"] for p in seen)  # never goes back
    assert all(p["done"] < p["total"] or p["partial"] is False for p in seen)

    final = job["progress"]
    assert final["done"] == final["total"] and final["converged"] + final["failed"] == final["done"]
    assert 1 <= len(final["top"]) <= 5
    assert [r["rank"] for r in final["top"]] == sorted(r["rank"] for r in final["top"])
    assert job["result_id"] and job["result_ref"] == job["result_id"]


def test_progress_snapshots_are_throttled_to_two_a_second(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(autofit_worker.time, "monotonic", lambda: clock[0])
    sent: list[dict[str, Any]] = []
    progress = autofit_worker._Progress(sent.append, total=50, skipped=7, config=AutoFitConfig())
    progress.start()
    assert [s["done"] for s in sent] == [0] and sent[0]["total"] == 50 and sent[0]["skipped"] == 7

    def outcome(i: int, kind: str) -> CandidateOutcome:
        return CandidateOutcome(candidate_id=f"m{i}", outcome=kind, reason="x")  # type: ignore[arg-type]

    for i in range(5):
        clock[0] += 0.05
        progress.add(outcome(i, "failed"))
    assert len(sent) == 1  # five outcomes inside half a second: no new snapshot
    clock[0] += 0.5
    progress.add(outcome(5, "timed_out"))
    assert len(sent) == 2 and sent[1]["done"] == 6 and sent[1]["failed"] == 6
    assert sent[1]["converged"] == 0 and sent[1]["top"] == []
    assert progress.snapshot(partial=True)["partial"] is True


# -- cancel ----------------------------------------------------------------------------------------


@LINUX_ONLY
def test_cancelling_a_run_leaves_no_worker_processes_behind(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(autofit_workers=3))
    headers = signup(client)
    dataset = upload(client, headers, linear_csv(40)).json()
    job_id = start(client, headers, dataset["id"], **POOL_RUN)

    pid = wait_until(lambda: job_process_pid(client, job_id), 60, "the job process")
    assert pid is not None
    # the job process leads its own group, so everything it starts is in it
    members = wait_until(
        lambda: [m for m in group_members(pid) if m != pid], 90, "the engine's worker processes"
    )
    assert len(members) >= 1

    cancelled = client.post(f"{API}/jobs/{job_id}/cancel", headers=headers)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    wait_until(lambda: not group_members(pid), 15, "the whole process group to die")
    assert not alive(pid) and not any(alive(m) for m in members)
    assert multiprocessing.parent_process() is None  # (we are the API process in this test)
    assert job_process_pid(client, job_id) is None
    listed = client.get(f"{API}/datasets/{dataset['id']}/autofits", headers=headers).json()
    assert listed == []  # a cancelled run stores nothing
    assert poll_job(client, headers, job_id)["status"] == "cancelled"


@LINUX_ONLY
def test_a_warm_worker_its_job_and_grandchildren_die_when_the_api_process_is_killed(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import multiprocessing, os, signal, sys, time, uuid\n"
        "import api_job_workers\n"
        "from curvestudio_api.worker_pool import send_ticket, warm_worker_main\n"
        "if __name__ == '__main__':\n"
        "    ctx = multiprocessing.get_context('spawn')\n"
        "    ours, theirs = ctx.Pipe(duplex=True)\n"
        "    worker = ctx.Process(target=warm_worker_main, args=(theirs,), daemon=False)\n"
        "    worker.start()\n"
        "    theirs.close()  # the worker holds the only far end\n"
        "    assert send_ticket(ours, api_job_workers.child_spawning_worker,\n"
        "                       {'pid_file': sys.argv[1]}, False, uuid.uuid4())\n"
        "    print(worker.pid, flush=True)\n"
        "    while not (os.path.exists(sys.argv[1]) and open(sys.argv[1]).read()):\n"
        "        time.sleep(0.05)\n"
        "    print('ready', flush=True)\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE), *sys.path])}
    parent = subprocess.Popen(
        [sys.executable, str(script), str(pid_file)],
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )
    worker_pid = 0
    grandchild_pid = 0
    try:
        assert parent.stdout is not None
        worker_pid = int(parent.stdout.readline())
        assert parent.stdout.readline().strip() == "ready"
        grandchild_pid = int(pid_file.read_text())
        parent.wait(timeout=10)
        assert parent.returncode == -9  # it was killed, it did not exit cleanly
        wait_until(lambda: not alive(worker_pid), 15, "the orphaned warm worker to exit")
        wait_until(lambda: not alive(grandchild_pid), 15, "its job's child to be killed too")
    finally:
        for pid in (worker_pid, grandchild_pid):
            if pid and alive(pid):
                os.kill(pid, 9)
        parent.kill()
        parent.wait()


# -- budget and timeout ----------------------------------------------------------------------------


def test_a_run_that_reaches_its_budget_keeps_a_partial_result(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(
        settings_factory(autofit_budget_s=1.0, autofit_timeout_s=90, autofit_workers=2)
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv(40)).json()
    # the whole catalogue (hundreds of candidates) cannot be fitted in one second
    job_id = start(client, headers, dataset["id"], cv_folds=3, n_starts=1)
    job = wait_finished(client, headers, job_id)
    assert job["status"] == "succeeded" and job["error_code"] is None, job
    assert job["progress"]["partial"] is True
    assert job["progress"]["done"] < job["progress"]["total"]

    stored = client.get(f"{API}/autofits/{job['result_id']}", headers=headers).json()
    assert stored["partial"] is True
    result = AutoFitResult.model_validate(stored["result"])
    assert result.cancelled is True
    assert any("cancelled" in w for w in result.warnings)
    assert any("never fitted" in o.reason or "cancelled" in o.reason for o in result.outcomes)
    # still a usable, re-rankable result
    assert [r.model_id for r in result.ranked_by(RankBy.RMSE)] == [
        r["model_id"]
        for r in client.get(
            f"{API}/autofits/{job['result_id']}?rank_by=rmse", headers=headers
        ).json()["result"]["ranked"]
    ]


def test_a_run_that_overruns_the_hard_timeout_fails_and_stores_nothing(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(
        settings_factory(autofit_budget_s=0.5, autofit_timeout_s=1.0, job_grace_s=0.2),
        autofit_worker=api_job_workers.sleeping_autofit_worker,
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    job_id = start(client, headers, dataset["id"], families=["polynomial"], max_parameters=2)
    job = wait_finished(client, headers, job_id, timeout=60)
    assert job["status"] == "failed" and job["error_code"] == "TIMEOUT"
    assert "Auto-fit" in job["error_message"] and job["result_id"] is None
    assert client.get(f"{API}/datasets/{dataset['id']}/autofits", headers=headers).json() == []


def test_a_fit_job_is_not_held_to_the_autofit_timeout(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(
        settings_factory(
            job_timeout_s=1.0, job_grace_s=0.2, autofit_timeout_s=90, autofit_budget_s=60
        ),
        job_worker=api_job_workers.sleeping_worker,
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    response = client.post(
        f"{API}/datasets/{dataset['id']}/fits",
        headers=headers,
        json={"model": {"kind": "polynomial", "degree": 1}},
    )
    job = wait_finished(client, headers, response.json()["job_id"], timeout=30)
    assert job["status"] == "failed" and job["error_code"] == "TIMEOUT"
    assert "the fit did not finish" in job["error_message"]


# -- the worker functions, called directly ---------------------------------------------------------


def line_payload(n: int = 20) -> dict[str, Any]:
    xy = XY(
        x=np.arange(1.0, n + 1),
        y=2.0 * np.arange(1.0, n + 1) + 1.0 + 0.05 * np.sin(np.arange(n)),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="fp-line",
    )
    return {
        "payload": encode_payload(xy),
        "x_name": "x",
        "y_name": "y",
        "fingerprint": "fp-line",
    }


def test_the_autofit_worker_reports_progress_and_returns_a_json_result() -> None:
    config = AutoFitConfig(families=("polynomial",), max_params=2, n_starts=1, cv_folds=3)
    reports: list[dict[str, Any]] = []
    result = autofit_worker.run_autofit_job(
        **line_payload(),
        spec={"config": config.model_dump(mode="json"), "budget_s": 60.0},
        report=reports.append,
    )
    assert result["ok"] is True and result["progress"]["partial"] is False
    run = AutoFitResult.model_validate(result["autofit"])
    assert run.cancelled is False and run.ranked[0].model_id == "poly1"
    assert reports[0]["done"] == 0 and reports[0]["total"] == len(run.outcomes)
    assert result["progress"]["done"] == result["progress"]["total"] == len(run.outcomes)
    json.dumps(result, allow_nan=False)  # it can cross the pipe


def test_the_autofit_worker_stops_at_its_budget_and_says_so() -> None:
    config = AutoFitConfig(families=("polynomial",), max_params=3, n_starts=1, cv_folds=3)
    result = autofit_worker.run_autofit_job(
        **line_payload(),
        spec={"config": config.model_dump(mode="json"), "budget_s": 0.0},
        report=lambda _snapshot: None,
    )
    assert result["ok"] is True
    assert result["autofit"]["cancelled"] is True and result["progress"]["partial"] is True


def test_the_autofit_worker_reports_a_refused_run_as_an_error_result() -> None:
    xy = XY(
        x=np.arange(1.0, 4.0),
        y=np.arange(1.0, 4.0),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint="fp-tiny",
    )
    result = autofit_worker.run_autofit_job(
        payload=encode_payload(xy),
        x_name="x",
        y_name="y",
        fingerprint="fp-tiny",
        spec={"config": AutoFitConfig().model_dump(mode="json"), "budget_s": 60.0},
        report=lambda _snapshot: None,
    )
    assert result["ok"] is False and result["code"] == "TOO_FEW_POINTS"


def test_the_open_row_worker_refits_with_the_runs_settings_and_checks_the_data() -> None:
    config = AutoFitConfig(families=("polynomial",), max_params=2, n_starts=1, seed=5)
    arguments = line_payload()
    spec = {
        "config": config.model_dump(mode="json"),
        "candidate_id": "poly1",
        "dataset_fingerprint": "fp-line",
    }
    opened = autofit_worker.run_open_row_job(**arguments, spec=spec)
    assert opened["ok"] is True
    assert FitBundle.model_validate(opened["bundle"]).result.model_id == "poly1"

    other = autofit_worker.run_open_row_job(
        **arguments, spec={**spec, "dataset_fingerprint": "another"}
    )
    assert other["ok"] is False and other["code"] == "DATA_MISMATCH"
    unknown = autofit_worker.run_open_row_job(**arguments, spec={**spec, "candidate_id": "nope"})
    assert unknown["ok"] is False and unknown["code"] == "UNKNOWN_MODEL"


# -- a realistic run -------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_true_model_is_among_the_top_three_on_the_whole_catalogue(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(autofit_workers=4))
    headers = signup(client)
    lines = ["t,signal"]
    for i in range(40):
        t = 0.25 * i
        noise = 0.02 * (((i * 37) % 11) - 5) / 5
        lines.append(f"{t!r},{5.0 * 2.718281828459045 ** (-0.35 * t) + noise!r}")
    dataset = upload(client, headers, ("\n".join(lines) + "\n").encode()).json()

    job_id = start(client, headers, dataset["id"], cv_folds=5, n_starts=3)
    job = wait_finished(client, headers, job_id, timeout=175)
    assert job["status"] == "succeeded", job
    assert job["progress"]["partial"] is False, "the run did not finish inside its budget"
    stored = client.get(f"{API}/autofits/{job['result_id']}", headers=headers).json()
    assert stored["n_candidates"] > 100
    top = [r["model_id"] for r in stored["result"]["ranked"][:3]]
    assert any(name in top for name in ("exponential_decay", "exp_decay_offset")), top
    best = stored["result"]["ranked"][0]
    assert best["metrics"]["r2"] > 0.99

    # open the best row: it reproduces the row's own parameters
    opened = client.post(
        f"{API}/autofits/{job['result_id']}/rows/{best['model_id']}/open", headers=headers
    )
    fit_job = wait_finished(client, headers, opened.json()["job_id"])
    assert fit_job["status"] == "succeeded", fit_job
    fit = client.get(f"{API}/fits/{fit_job['result_ref']}", headers=headers).json()
    for refit, row in zip(fit["result"]["parameters"], best["parameters"], strict=True):
        assert refit["value"] == pytest.approx(row["value"], rel=1e-6)
