"""Uploads are parsed in their own short-lived process: bounded time, memory and concurrency."""

import io
import multiprocessing
import sys
import threading
import time
import zipfile
from collections.abc import Callable
from typing import Any

import api_job_workers
import pytest
from api_test_helpers import linear_csv, signup, upload
from fastapi.testclient import TestClient
from openpyxl import Workbook

API = "/api/v1"


def test_xlsx_declaring_more_than_100_mb_is_refused_before_any_process_starts(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    headers = signup(client)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * 150_000_000)
    bomb = buffer.getvalue()
    assert len(bomb) < 1_000_000  # a 150 MB expansion in under 1 MB
    response = upload(client, headers, bomb, "bomb.xlsx")
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "TOO_LARGE"
    assert "expands" in response.json()["message"]
    assert client.app.state.ctx.imports.spawned == 0  # type: ignore[attr-defined]


def test_a_flood_of_zip_entries_is_refused_unread(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    headers = signup(client)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for i in range(10_500):
            archive.writestr(f"f{i}", b"")
    response = upload(client, headers, buffer.getvalue(), "many.xlsx")
    assert response.status_code == 413
    assert client.app.state.ctx.imports.spawned == 0  # type: ignore[attr-defined]


def test_too_many_csv_lines_are_refused_before_parsing(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(max_rows=50))
    headers = signup(client)
    response = upload(client, headers, linear_csv(3000))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "TOO_LARGE"
    assert client.app.state.ctx.imports.spawned == 0  # type: ignore[attr-defined]


def test_a_parse_that_runs_too_long_is_killed(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(import_timeout_s=1.0), import_worker=api_job_workers.sleeping_import_worker
    )
    headers = signup(client)
    started = time.monotonic()
    response = upload(client, headers, linear_csv())
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "IMPORT_TIMEOUT"
    assert time.monotonic() - started < 15
    assert multiprocessing.active_children() == []  # killed and reaped, no zombie
    assert client.get(f"{API}/datasets", headers=headers).json() == []


def test_a_parser_that_dies_is_a_clean_422(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory(import_worker=api_job_workers.crashing_import_worker)
    headers = signup(client)
    response = upload(client, headers, linear_csv())
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "IMPORT_FAILED"
    assert "Traceback" not in response.text
    assert multiprocessing.active_children() == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: RLIMIT_AS")
def test_the_address_space_cap_turns_a_memory_bomb_into_a_413(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(import_memory_mb=256),
        import_worker=api_job_workers.memory_hog_import_worker,
    )
    headers = signup(client)
    response = upload(client, headers, linear_csv())
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "TOO_LARGE"
    assert "memory" in response.json()["message"]


def test_import_concurrency_is_two_overall_and_one_per_user(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(import_timeout_s=4.0), import_worker=api_job_workers.sleeping_import_worker
    )
    a, b, c = signup(client), signup(client), signup(client)
    results: dict[str, Any] = {}

    def slow(name: str, headers: dict[str, str]) -> None:
        results[name] = upload(client, headers, linear_csv())

    threads = [threading.Thread(target=slow, args=(n, h)) for n, h in (("a", a), ("b", b))]
    for thread in threads:
        thread.start()
    runner = client.app.state.ctx.imports  # type: ignore[attr-defined]
    deadline = time.monotonic() + 10
    while runner.spawned < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.spawned == 2
    again = upload(client, a, linear_csv())  # user a already has one running
    third = upload(client, c, linear_csv())  # the server is at its limit of two
    for response in (again, third):
        assert response.status_code == 429
        assert response.json()["error"] == "TOO_MANY_UPLOADS"
    for thread in threads:
        thread.join(timeout=30)
    assert results["a"].status_code == results["b"].status_code == 422  # both timed out
    # the slots are free again
    assert runner.spawned == 2


def test_the_event_loop_stays_responsive_during_concurrent_heavy_uploads(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    users = [signup(client) for _ in range(6)]
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet()
    sheet.append(["x", "y"])
    for i in range(120_000):
        sheet.append([i * 0.01, i * 0.02 + 1.0])
    buffer = io.BytesIO()
    workbook.save(buffer)
    heavy = buffer.getvalue()
    results: list[int] = []

    def send(headers: dict[str, str]) -> None:
        results.append(upload(client, headers, heavy, "big.xlsx").status_code)

    threads = [threading.Thread(target=send, args=(h,)) for h in users]
    for thread in threads:
        thread.start()
    slowest = 0.0
    probes = 0
    while any(t.is_alive() for t in threads):
        started = time.perf_counter()
        assert client.get(f"{API}/health").status_code == 200
        slowest = max(slowest, time.perf_counter() - started)
        probes += 1
        time.sleep(0.1)
    for thread in threads:
        thread.join()
    assert probes >= 5
    assert slowest < 1.0, f"/health took {slowest:.2f} s while uploads were parsing"
    assert 201 in results  # the heavy file really was parsed
    assert set(results) <= {201, 429}
    assert multiprocessing.active_children() == []
