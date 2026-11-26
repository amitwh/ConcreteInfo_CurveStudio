"""The fit-centric tools through the API: the three job kinds, their routes and the
statistical results they store.

Each kind is derived from one stored fit: the discernibility report from a fit opened from an
Auto-fit row, the feasibility simulation from any fit's specification, and the
next-measurement advice from any converged fit's Jacobian. All three store one
``stat_results`` row the finished job points at.
"""

import time
import uuid
from collections.abc import Callable
from typing import Any

import api_job_workers
import pytest
from api_test_helpers import linear_csv, signup, upload
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from curvestudio_api.export_openapi import generate
from curvestudio_api.orm import StatResult
from curvestudio_api.settings import Settings

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}
SMALL_RUN = {"families": ["polynomial"], "max_parameters": 2, "cv_folds": 3, "n_starts": 1}
"""Few enough candidates for an Auto-fit run to stay inside its job process."""
SIMULATION = {
    "truth": {"b0": 1.0, "b1": 2.0},
    "design": {"low": 0.0, "high": 10.0, "count": 5},
    "noise": {"kind": "sd", "value": 0.05},
    "runs": 5,
    "seed": 11,
}
"""A small simulation of a straight line: five datasets, five design points."""


def wait_for_job(
    client: TestClient,
    headers: dict[str, str],
    job_id: str,
    statuses: tuple[str, ...] = ("succeeded", "failed", "cancelled"),
    timeout: float = 90,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}", headers=headers).json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stayed {job.get('status')!r}, wanted one of {statuses}")


def fit_through_api(
    client: TestClient, headers: dict[str, str], dataset_id: str, spec: dict[str, Any]
) -> dict[str, Any]:
    """Run one fit job to the end and return its job; ``result_ref`` is the stored fit's id."""
    response = client.post(f"{API}/datasets/{dataset_id}/fits", headers=headers, json=spec)
    assert response.status_code == 202, response.text
    job = wait_for_job(client, headers, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    return job


def stat_count(client: TestClient) -> int:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return int(db.scalar(select(func.count()).select_from(StatResult)) or 0)


def start_tool(
    client: TestClient, headers: dict[str, str], fit_id: str, name: str, body: Any = None
) -> str:
    response = client.post(f"{API}/fits/{fit_id}/{name}", headers=headers, json=body)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def stored_result(client: TestClient, headers: dict[str, str], result_id: str) -> dict[str, Any]:
    response = client.get(f"{API}/stats/results/{result_id}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def fitted(shared_client: TestClient) -> dict[str, Any]:
    """One account's straight-line dataset, an ordinary fit, and a fit opened from the best
    row of a finished Auto-fit run over the same data."""
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv(30)).json()
    plain = fit_through_api(shared_client, headers, dataset["id"], POLY1)
    started = shared_client.post(
        f"{API}/datasets/{dataset['id']}/autofit", headers=headers, json=SMALL_RUN
    )
    assert started.status_code == 202, started.text
    run = wait_for_job(shared_client, headers, started.json()["job_id"])
    assert run["status"] == "succeeded", run
    autofit_id = run["result_id"]
    ranking = shared_client.get(f"{API}/autofits/{autofit_id}", headers=headers).json()["result"]
    row = ranking["ranked"][0]["model_id"]
    opened = shared_client.post(f"{API}/autofits/{autofit_id}/rows/{row}/open", headers=headers)
    assert opened.status_code == 202, opened.text
    opened_job = wait_for_job(shared_client, headers, opened.json()["job_id"])
    assert opened_job["status"] == "succeeded", opened_job
    return {
        "client": shared_client,
        "headers": headers,
        "dataset": dataset,
        "plain_fit": plain["result_ref"],
        "ranked_fit": opened_job["result_ref"],
        "autofit_id": autofit_id,
    }


# -- one run of each kind --------------------------------------------------------------------------


def test_the_discernibility_report_runs_as_a_job_and_stores_one_result(
    fitted: dict[str, Any],
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = fitted["ranked_fit"]
    before = stat_count(client)
    started = start_tool(client, headers, fit_id, "discernibility", {"max_models": 1})
    job = wait_for_job(client, headers, started)
    assert job["status"] == "succeeded", job
    assert job["kind"] == "discern"
    assert job["result_id"] is not None and job["result_id"] == job["result_ref"]
    assert stat_count(client) == before + 1

    body = stored_result(client, headers, job["result_id"])
    assert body["tool"] == "discernibility"
    assert body["dataset_id"] == fitted["dataset"]["id"]
    assert body["request"]["fit_id"] == fit_id
    assert body["request"]["max_models"] == 1
    assert body["result"]["tool"] == "discernibility"
    assert body["result"]["title"] == "Model discernibility"
    titles = {table["title"] for table in body["result"]["tables"]}
    assert {"Models", "Predictive groups", "The simplest model that can be defended"} <= titles

    # the body is optional: the request's own defaults are stored
    started = start_tool(client, headers, fit_id, "discernibility")
    job = wait_for_job(client, headers, started)
    assert job["status"] == "succeeded", job
    body = stored_result(client, headers, job["result_id"])
    assert body["request"]["grid_points"] == 200 and body["request"]["profile"] is True
    assert stat_count(client) == before + 2


def test_the_feasibility_simulation_runs_as_a_job_and_stores_one_result(
    fitted: dict[str, Any],
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = fitted["plain_fit"]
    before = stat_count(client)
    job = wait_for_job(client, headers, start_tool(client, headers, fit_id, "simulate", SIMULATION))
    assert job["status"] == "succeeded", job
    assert job["kind"] == "simulate"
    assert job["result_id"] is not None
    assert stat_count(client) == before + 1

    body = stored_result(client, headers, job["result_id"])
    assert body["tool"] == "feasibility"
    assert body["dataset_id"] == fitted["dataset"]["id"]
    assert body["request"]["fit_id"] == fit_id
    assert body["request"]["truth"] == SIMULATION["truth"]
    assert body["request"]["runs"] == SIMULATION["runs"]
    assert body["result"]["tool"] == "feasibility"
    assert body["result"]["title"].startswith("Experiment feasibility for")
    # five datasets were simulated, and every parameter's estimate recovered
    assert body["result"]["counts"]["n_offered"] == 5
    titles = {table["title"] for table in body["result"]["tables"]}
    assert {"Parameter recovery", "Design"} <= titles


def test_the_advisor_runs_as_a_job_and_stores_one_result(fitted: dict[str, Any]) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = fitted["plain_fit"]
    before = stat_count(client)
    started = start_tool(client, headers, fit_id, "advisor", {"points": 2, "candidates": 50})
    job = wait_for_job(client, headers, started)
    assert job["status"] == "succeeded", job
    assert job["kind"] == "advise"
    assert job["result_id"] is not None
    assert stat_count(client) == before + 1

    body = stored_result(client, headers, job["result_id"])
    assert body["tool"] == "advisor"
    assert body["request"]["fit_id"] == fit_id
    assert body["request"]["points"] == 2
    assert body["result"]["tool"] == "advisor"
    assert body["result"]["title"].startswith("Where to measure next")
    points = next(t for t in body["result"]["tables"] if t["title"] == "Recommended points")
    assert len(points["rows"]) == 2


# -- who may ask, and how often --------------------------------------------------------------------


def test_the_fit_tool_routes_need_a_token_and_hide_another_users_fit(
    fitted: dict[str, Any],
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = fitted["plain_fit"]
    routes = (
        ("discernibility", None),
        ("simulate", SIMULATION),
        ("advisor", None),
    )
    stranger = signup(client)
    for name, body in routes:
        path = f"{API}/fits/{fit_id}/{name}"
        anonymous = client.post(path, json=body)
        assert anonymous.status_code == 401, (name, anonymous.text)
        assert anonymous.json()["error"] == "UNAUTHORIZED"
        foreign = client.post(path, headers=stranger, json=body)
        assert foreign.status_code == 404, (name, foreign.text)
        assert foreign.json()["error"] == "NOT_FOUND"
        missing = client.post(f"{API}/fits/{uuid.uuid4()}/{name}", headers=headers, json=body)
        assert missing.status_code == 404, (name, missing.text)


def test_a_fit_without_a_ranking_has_no_discernibility_report(fitted: dict[str, Any]) -> None:
    client, headers = fitted["client"], fitted["headers"]
    job = wait_for_job(
        client, headers, start_tool(client, headers, fitted["plain_fit"], "discernibility")
    )
    assert job["status"] == "failed", job
    assert job["error_code"] == "NO_RANKING"
    assert "open a row from an Auto-fit result first" in job["error_message"]
    assert job["result_id"] is None


def test_a_second_request_of_a_kind_is_a_409_and_a_cancelled_job_stores_nothing(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(), simulate_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    fit_id = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]

    holding = start_tool(client, headers, fit_id, "simulate", SIMULATION)
    wait_for_job(client, headers, holding, statuses=("running",))
    before = stat_count(client)
    again = client.post(f"{API}/fits/{fit_id}/simulate", headers=headers, json=SIMULATION)
    assert again.status_code == 409, again.text
    assert again.json()["error"] == "CONFLICT"
    assert again.json()["detail"] == {"code": "SIMULATION_RUNNING", "job_id": holding}
    # another kind of job for the same fit is not in the way
    assert client.post(f"{API}/fits/{fit_id}/advisor", headers=headers).status_code == 202

    assert client.post(f"{API}/jobs/{holding}/cancel", headers=headers).status_code == 200
    job = wait_for_job(client, headers, holding, statuses=("cancelled",))
    assert job["status"] == "cancelled" and job["result_id"] is None
    assert stat_count(client) == before  # the cancelled job wrote no row


def test_a_fit_deleted_before_its_job_starts_fails_fit_deleted(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(), simulate_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    holding_fit = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    other_fit = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]

    # the only worker is held, so the second fit's job waits in the queue
    holding = start_tool(client, headers, holding_fit, "simulate", SIMULATION)
    wait_for_job(client, headers, holding, statuses=("running",))
    waiting = start_tool(client, headers, other_fit, "simulate", SIMULATION)
    assert client.get(f"{API}/jobs/{waiting}", headers=headers).json()["status"] == "queued"
    assert client.delete(f"{API}/fits/{other_fit}", headers=headers).status_code == 204

    assert client.post(f"{API}/jobs/{holding}/cancel", headers=headers).status_code == 200
    job = wait_for_job(client, headers, waiting)
    assert job["status"] == "failed", job
    assert job["error_code"] == "FIT_DELETED"
    assert job["error_message"] == "the fit was deleted before its feasibility simulation started"
    assert job["result_id"] is None


def test_openapi_documents_the_fit_tool_endpoints() -> None:
    document = generate()
    fit = "/api/v1/fits/{fit_id}"
    for name in ("discernibility", "simulate", "advisor"):
        started = document["paths"][f"{fit}/{name}"]
        assert set(started) == {"post"}, name
        assert "202" in started["post"]["responses"], name
    assert "discern" in str(document["components"]["schemas"]["JobOut"]["properties"]["kind"])
