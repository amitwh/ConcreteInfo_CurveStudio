"""Auto-fit endpoints: validation, a run end to end, re-ranking, opening a row, export, quotas,
isolation between users, and the migration. Runs use small candidate sets (polynomials) so the
engine stays in the job process."""

import csv
import io
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from conftest import linear_csv, make_settings, signup, upload
from curvestudio.discovery import AutoFitResult, RankBy
from curvestudio.workflow import FitBundle, FitSpec, bundle_for_candidate, predict
from curvestudio_api.app import create_app
from curvestudio_api.export_openapi import generate
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.orm import Autofit, Dataset, Fit
from curvestudio_api.settings import Settings
from curvestudio_api.storage import decode_payload
from curvestudio_api.workflow_errors import autofit_api_error

API = "/api/v1"
SMALL = {"families": ["polynomial"], "max_parameters": 2, "cv_folds": 3, "n_starts": 1}
"""A handful of candidates (poly1, poly1_origin, poly2_origin and a registered model of the
polynomial family): few enough for the engine to fit them in the job process itself."""


def wait_for(
    client: TestClient, headers: dict[str, str], job_id: str, timeout: float = 120
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {job}")


def start(
    client: TestClient, headers: dict[str, str], dataset_id: str, **body: Any
) -> dict[str, Any]:
    response = client.post(
        f"{API}/datasets/{dataset_id}/autofit", headers=headers, json={**SMALL, **body}
    )
    assert response.status_code == 202, response.text
    job = wait_for(client, headers, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    return job


@pytest.fixture(scope="module")
def run(shared_client: TestClient) -> dict[str, Any]:
    """One finished run on a noisy straight line, shared by the read-only tests."""
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv(30)).json()
    job = start(shared_client, headers, dataset["id"])
    return {"client": shared_client, "headers": headers, "dataset": dataset, "job": job}


# -- validation ------------------------------------------------------------------------------------


def test_request_bounds_are_enforced(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    url = f"{API}/datasets/{dataset['id']}/autofit"
    bad_bodies: list[dict[str, Any]] = [
        {"n_starts": 9},
        {"n_starts": 0},
        {"cv_folds": 1},
        {"cv_folds": 11},
        {"curve_top_n": 0},
        {"curve_top_n": 11},
        {"seed": 2**53},
        {"seed": -1},
        {"rank_by": "magic"},
        {"families": []},
        {"max_parameters": 0},
        {"custom_models": [{"id": f"m{i}", "name": "m", "expression": "a*x"} for i in range(11)]},
        {"unknown_field": 1},
    ]
    for body in bad_bodies:
        response = shared_client.post(url, headers=headers, json=body)
        assert response.status_code == 422, (body, response.text)
        assert response.json()["error"] == "VALIDATION_ERROR"
    assert (
        shared_client.post(url, headers=headers, json={"seed": 2**53 - 1, **SMALL}).status_code
        == 202
    )


def test_too_few_points_is_a_422_with_its_code(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, b"x,y\n1,2\n2,4\n3,6\n").json()
    response = shared_client.post(
        f"{API}/datasets/{dataset['id']}/autofit", headers=headers, json={}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["detail"]["code"] == "TOO_FEW_POINTS"


def test_unknown_families_and_unusable_custom_models_are_422s(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    url = f"{API}/datasets/{dataset['id']}/autofit"

    unknown = shared_client.post(url, headers=headers, json={"families": ["polynomial", "nope"]})
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "UNKNOWN_FAMILY"
    assert "'nope'" in unknown.json()["message"]

    broken = {"id": "broken", "name": "Broken", "expression": "a*x**"}
    custom = shared_client.post(url, headers=headers, json={"custom_models": [broken]})
    assert custom.status_code == 422
    assert custom.json()["detail"]["code"] == "INVALID_CUSTOM_MODEL"

    clash = {"id": "poly1", "name": "Clash", "expression": "a*x + b"}
    duplicate = shared_client.post(url, headers=headers, json={"custom_models": [clash]})
    assert duplicate.status_code == 422
    assert duplicate.json()["detail"]["code"] == "DUPLICATE_MODEL_ID"

    # A family that exists but has no model within the parameter limit is no run either (the
    # smallest sigmoidal model has three parameters).
    empty = shared_client.post(
        url, headers=headers, json={"families": ["sigmoidal"], "max_parameters": 2}
    )
    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "NO_CANDIDATES"


def test_a_dataset_that_is_not_yours_is_not_found(shared_client: TestClient) -> None:
    owner = signup(shared_client)
    other = signup(shared_client)
    dataset = upload(shared_client, owner, linear_csv()).json()
    response = shared_client.post(
        f"{API}/datasets/{dataset['id']}/autofit", headers=other, json=SMALL
    )
    assert response.status_code == 404 and response.json()["error"] == "NOT_FOUND"
    assert (
        shared_client.get(f"{API}/datasets/{dataset['id']}/autofits", headers=other).status_code
        == 404
    )


# -- a run, read back ------------------------------------------------------------------------------


def test_a_run_is_stored_and_listed(run: dict[str, Any]) -> None:
    client, headers, dataset, job = run["client"], run["headers"], run["dataset"], run["job"]
    autofit_id = job["result_id"]
    assert job["kind"] == "autofit" and job["result_ref"] == autofit_id
    progress = job["progress"]
    n = progress["total"]
    assert 3 <= n <= 4 and progress["done"] == n and progress["skipped"] == 0
    assert progress["converged"] == n and progress["failed"] == 0 and progress["partial"] is False
    assert 1 <= len(progress["top"]) <= 5 and progress["top"][0]["rank"] == 1

    listed = client.get(f"{API}/datasets/{dataset['id']}/autofits", headers=headers).json()
    assert [(a["id"], a["n_candidates"], a["rank_by"]) for a in listed] == [
        (autofit_id, n, "best_overall")
    ]
    body = client.get(f"{API}/autofits/{autofit_id}", headers=headers).json()
    assert body["partial"] is False and body["best_model_id"] == listed[0]["best_model_id"]
    result = AutoFitResult.model_validate(body["result"])
    assert result.ranked[0].model_id == body["best_model_id"]
    assert {"poly1", "poly1_origin", "poly2_origin"} <= {o.candidate_id for o in result.outcomes}
    assert result.cancelled is False and len(result.outcomes) == n
    assert result.ranked[0].model_id == "poly1"  # the data are a straight line with an offset
    # per-point vectors are not stored, curves of the top rows are
    assert all(o.fit is None or o.fit.fitted == [] for o in result.outcomes)
    assert result.ranked[0].model_id in result.curves


@pytest.mark.parametrize("rank_by", list(RankBy))
def test_ranking_again_equals_ranked_by_without_refitting(
    run: dict[str, Any], rank_by: RankBy
) -> None:
    client, headers = run["client"], run["headers"]
    autofit_id = run["job"]["result_id"]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        stored = AutoFitResult.model_validate(db.get(Autofit, uuid.UUID(autofit_id)).result)  # type: ignore[union-attr]
    body = client.get(
        f"{API}/autofits/{autofit_id}?rank_by={rank_by.value}", headers=headers
    ).json()
    assert body["rank_by"] == rank_by.value
    assert [r["model_id"] for r in body["result"]["ranked"]] == [
        r.model_id for r in stored.ranked_by(rank_by)
    ]
    assert body["result"]["config"]["rank_by"] == rank_by.value
    again = AutoFitResult.model_validate(body["result"])
    assert again.ranked == stored.ranked_by(rank_by)


def test_an_unknown_rank_by_is_a_422(run: dict[str, Any]) -> None:
    client, headers = run["client"], run["headers"]
    for path in ("", "/export"):
        response = client.get(
            f"{API}/autofits/{run['job']['result_id']}{path}?rank_by=magic", headers=headers
        )
        assert response.status_code == 422 and response.json()["error"] == "VALIDATION_ERROR"


def test_deleting_a_run_removes_it(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    autofit_id = start(shared_client, headers, dataset["id"])["result_id"]
    assert shared_client.delete(f"{API}/autofits/{autofit_id}", headers=headers).status_code == 204
    assert shared_client.get(f"{API}/autofits/{autofit_id}", headers=headers).status_code == 404
    assert shared_client.delete(f"{API}/autofits/{autofit_id}", headers=headers).status_code == 404
    assert (
        shared_client.get(f"{API}/datasets/{dataset['id']}/autofits", headers=headers).json() == []
    )


def test_deleting_the_dataset_removes_its_runs(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    autofit_id = start(shared_client, headers, dataset["id"])["result_id"]
    assert (
        shared_client.delete(f"{API}/datasets/{dataset['id']}", headers=headers).status_code == 204
    )
    assert shared_client.get(f"{API}/autofits/{autofit_id}", headers=headers).status_code == 404


# -- opening a row ---------------------------------------------------------------------------------


def stored_xy(client: TestClient, dataset_id: str) -> Any:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        dataset = db.get(Dataset, uuid.UUID(dataset_id))
        return decode_payload(
            dataset.payload,
            x_name=dataset.x_column,
            y_name=dataset.y_column,
            fingerprint=dataset.fingerprint,
        )


def open_row(client: TestClient, headers: dict[str, str], autofit_id: str, row: str) -> Any:
    return client.post(f"{API}/autofits/{autofit_id}/rows/{row}/open", headers=headers)


def test_opening_a_row_stores_the_same_fit_the_engine_gives(run: dict[str, Any]) -> None:
    client, headers, dataset = run["client"], run["headers"], run["dataset"]
    autofit_id = run["job"]["result_id"]
    opened = open_row(client, headers, autofit_id, "poly1")
    assert opened.status_code == 202 and opened.json()["fit_id"] is None
    job = wait_for(client, headers, opened.json()["job_id"])
    assert job["status"] == "succeeded" and job["kind"] == "open_row"
    fit_id = job["result_ref"]
    assert job["result_id"] == fit_id

    fit = client.get(f"{API}/fits/{fit_id}", headers=headers).json()
    stored = AutoFitResult.model_validate(
        client.get(f"{API}/autofits/{autofit_id}", headers=headers).json()["result"]
    )
    expected = bundle_for_candidate(stored_xy(client, dataset["id"]), stored, "poly1", compact=True)
    got = FitBundle.model_validate(fit)
    assert [p.value for p in got.result.parameters] == [p.value for p in expected.result.parameters]
    row = next(r for r in stored.ranked if r.model_id == "poly1")
    assert [p.value for p in got.result.parameters] == [p.value for p in row.parameters]

    summary = client.get(f"{API}/datasets/{dataset['id']}/fits", headers=headers).json()
    assert fit_id in [f["id"] for f in summary]

    # the stored fit is a normal one: predict and export work
    prediction = client.post(
        f"{API}/fits/{fit_id}/predict",
        headers=headers,
        json={"x": [1.0, 2.0], "band": "confidence"},
    )
    assert prediction.status_code == 200, prediction.text
    direct = predict(
        expected,
        FitSpec.model_validate({"model": {"kind": "polynomial", "degree": 1}}),
        [1.0, 2.0],
        "confidence",
        data=stored_xy(client, dataset["id"]),
    )
    assert prediction.json()["y"] == direct.y
    assert client.get(f"{API}/fits/{fit_id}/export?format=csv", headers=headers).status_code == 200


def test_opening_a_row_twice_returns_the_same_fit(run: dict[str, Any]) -> None:
    client, headers = run["client"], run["headers"]
    autofit_id = run["job"]["result_id"]
    first = open_row(client, headers, autofit_id, "poly1_origin")
    job = wait_for(client, headers, first.json()["job_id"])
    second = open_row(client, headers, autofit_id, "poly1_origin")
    assert second.status_code == 202
    assert second.json()["fit_id"] == job["result_ref"]
    again = client.get(f"{API}/jobs/{second.json()['job_id']}", headers=headers).json()
    assert again["status"] == "succeeded" and again["result_ref"] == job["result_ref"]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        fits = db.scalars(select(Fit).where(Fit.candidate_id == "poly1_origin")).all()
    assert [f.id for f in fits if str(f.autofit_id) == autofit_id] == [uuid.UUID(job["result_ref"])]


def test_a_polynomial_through_the_origin_can_be_opened_and_predicted(run: dict[str, Any]) -> None:
    client, headers = run["client"], run["headers"]
    job = wait_for(
        client,
        headers,
        open_row(client, headers, run["job"]["result_id"], "poly2_origin").json()["job_id"],
    )
    assert job["status"] == "succeeded", job
    prediction = client.post(
        f"{API}/fits/{job['result_ref']}/predict", headers=headers, json={"x": [3.0]}
    )
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["y"][0] is not None


def test_only_fitted_rows_of_the_run_can_be_opened(run: dict[str, Any]) -> None:
    client, headers = run["client"], run["headers"]
    autofit_id = run["job"]["result_id"]
    assert open_row(client, headers, autofit_id, "no_such_row").status_code == 404
    assert open_row(client, headers, autofit_id, "poly9").status_code == 404
    skipped = open_row(client, headers, autofit_id, "poly3")  # never a candidate (max 2 parameters)
    assert skipped.status_code == 404


# -- export ----------------------------------------------------------------------------------------


def test_csv_export_is_one_table_of_candidates(run: dict[str, Any]) -> None:
    client, headers = run["client"], run["headers"]
    response = client.get(
        f"{API}/autofits/{run['job']['result_id']}/export?format=csv&rank_by=rmse", headers=headers
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == [
        "rank", "id", "name", "category", "equation", "k", "r2", "adj_r2", "rmse", "aicc", "bic",
        "cv_rmse", "status", "notes",
    ]  # fmt: skip
    body = rows[1:]
    assert len(body) >= 3 and {len(r) for r in body} == {14}
    assert [r[0] for r in body] == [str(i) for i in range(1, len(body) + 1)]
    rmse = [float(r[8]) for r in body]
    assert rmse == sorted(rmse)  # ranked by rmse
    assert body[0][12] in ("converged", "exact_fit")

    as_json = client.get(
        f"{API}/autofits/{run['job']['result_id']}/export?format=json", headers=headers
    )
    assert as_json.status_code == 200
    assert AutoFitResult.model_validate(as_json.json()).ranked[0].rank == 1


def test_csv_export_neutralises_formulas(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    evil = {"id": "evil_line", "name": '=HYPERLINK("http://x")', "expression": "a*x + b"}
    job = start(shared_client, headers, dataset["id"], custom_models=[evil])
    text = shared_client.get(
        f"{API}/autofits/{job['result_id']}/export?format=csv", headers=headers
    ).text
    cells = [cell for row in csv.reader(io.StringIO(text)) for cell in row]
    assert '\'=HYPERLINK("http://x")' in cells
    assert not any(cell.startswith(("=", "+", "@")) for cell in cells)


def test_a_custom_equation_takes_part_and_can_be_opened(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    mine = {
        "id": "my_line",
        "name": "My line",
        "expression": "a*x + b",
        "parameters": [{"name": "a", "initial": 1.0}, {"name": "b", "initial": 0.0, "lower": -5}],
    }
    job = start(shared_client, headers, dataset["id"], custom_models=[mine])
    result = AutoFitResult.model_validate(
        shared_client.get(f"{API}/autofits/{job['result_id']}", headers=headers).json()["result"]
    )
    assert "my_line" in {o.candidate_id for o in result.outcomes}
    opened = wait_for(
        shared_client,
        headers,
        open_row(shared_client, headers, job["result_id"], "my_line").json()["job_id"],
    )
    assert opened["status"] == "succeeded", opened
    fit = shared_client.get(f"{API}/fits/{opened['result_ref']}", headers=headers).json()
    assert fit["result"]["model_id"] == "my_line"
    assert abs(fit["result"]["parameters"][0]["value"] - 2.0) < 0.1
    # Known limit: a fit specification cannot name a custom model by id, so predict says so.
    prediction = shared_client.post(
        f"{API}/fits/{opened['result_ref']}/predict", headers=headers, json={"x": [1.0]}
    )
    assert prediction.status_code == 422
    assert prediction.json()["detail"]["code"] == "MODEL_MISMATCH"


# -- quotas, isolation -----------------------------------------------------------------------------


def test_the_daily_autofit_quota_is_enforced(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(quota_max_autofits_per_day=1))
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    url = f"{API}/datasets/{dataset['id']}/autofit"
    first = client.post(url, headers=headers, json=SMALL)
    assert first.status_code == 202
    second = client.post(url, headers=headers, json=SMALL)
    assert second.status_code == 429
    body = second.json()
    assert body["error"] == "QUOTA_EXCEEDED"
    assert body["detail"]["quota"] == "autofits_per_day" and body["detail"]["limit"] == 1
    assert 1 <= int(second.headers["retry-after"]) <= 86_400
    # another user has their own allowance
    other = signup(client)
    other_data = upload(client, other, linear_csv()).json()
    assert (
        client.post(
            f"{API}/datasets/{other_data['id']}/autofit", headers=other, json=SMALL
        ).status_code
        == 202
    )


def test_autofit_runs_do_not_use_up_the_fit_quota(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(quota_max_fits=1))
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    url = f"{API}/datasets/{dataset['id']}/autofit"
    ids = [client.post(url, headers=headers, json=SMALL).json()["job_id"] for _ in range(2)]
    for job_id in ids:
        assert wait_for(client, headers, job_id)["status"] == "succeeded"


def test_every_autofit_route_hides_other_users_runs(run: dict[str, Any]) -> None:
    client = run["client"]
    stranger = signup(client)
    autofit_id = run["job"]["result_id"]
    base = f"{API}/autofits/{autofit_id}"
    responses = [
        client.get(base, headers=stranger),
        client.get(f"{base}?rank_by=rmse", headers=stranger),
        client.delete(base, headers=stranger),
        client.get(f"{base}/export?format=csv", headers=stranger),
        client.get(f"{base}/export?format=json", headers=stranger),
        client.post(f"{base}/rows/poly1/open", headers=stranger),
        client.get(f"{API}/jobs/{run['job']['id']}", headers=stranger),
        client.post(f"{API}/jobs/{run['job']['id']}/cancel", headers=stranger),
        client.get(f"{API}/datasets/{run['dataset']['id']}/autofits", headers=stranger),
        client.post(f"{API}/datasets/{run['dataset']['id']}/autofit", headers=stranger, json=SMALL),
    ]
    assert [r.status_code for r in responses] == [404] * len(responses)
    assert all(r.json()["error"] == "NOT_FOUND" for r in responses)
    # ...and without a token
    for path in (base, f"{base}/export", f"{API}/datasets/{run['dataset']['id']}/autofits"):
        assert client.get(path).status_code == 401
    assert client.get(base, headers=run["headers"]).status_code == 200  # still there


# -- settings, errors, schema, migration -----------------------------------------------------------


def test_the_budget_must_be_below_the_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BUDGET"):
        make_settings(tmp_path, autofit_budget_s=60, autofit_timeout_s=60)
    assert make_settings(tmp_path).autofit_budget_s < make_settings(tmp_path).autofit_timeout_s


def test_autofit_error_codes_reach_the_envelope() -> None:
    from curvestudio.discovery import AutoFitError

    error = autofit_api_error(AutoFitError("TOO_FEW_POINTS", "needs 4"))
    assert (error.status_code, error.code, error.extra) == (
        422,
        "VALIDATION_ERROR",
        {"code": "TOO_FEW_POINTS"},
    )


def test_openapi_documents_the_autofit_endpoints() -> None:
    document = generate()
    paths = document["paths"]
    expected = {
        "/api/v1/datasets/{dataset_id}/autofit": {"post"},
        "/api/v1/datasets/{dataset_id}/autofits": {"get"},
        "/api/v1/autofits/{autofit_id}": {"get", "delete"},
        "/api/v1/autofits/{autofit_id}/export": {"get"},
        "/api/v1/autofits/{autofit_id}/rows/{candidate_id}/open": {"post"},
    }
    for path, methods in expected.items():
        assert methods <= set(paths[path]), path
    schemas = document["components"]["schemas"]
    assert {"AutoFitRequest", "AutofitOut", "AutoFitResult", "RankedFit", "OpenRowOut"} <= set(
        schemas
    )
    job = schemas["JobOut"]["properties"]
    assert "progress" in job and "result_id" in job
    assert "open_row" in str(job["kind"])


def test_migration_0003_upgrades_and_downgrades_on_sqlite(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    engine = app.state.ctx.engine
    upgrade_to_head(engine)
    inspector = inspect(engine)
    assert "autofits" in inspector.get_table_names()
    assert "progress" in {c["name"] for c in inspector.get_columns("jobs")}
    assert {"autofit_id", "candidate_id"} <= {c["name"] for c in inspector.get_columns("fits")}
    progress = next(c for c in inspector.get_columns("jobs") if c["name"] == "progress")
    assert progress["nullable"] is True

    command.downgrade(_config(engine), "0002")
    inspector = inspect(engine)
    assert "autofits" not in inspector.get_table_names()
    assert "progress" not in {c["name"] for c in inspector.get_columns("jobs")}
    assert not {"autofit_id", "candidate_id"} & {c["name"] for c in inspector.get_columns("fits")}
    assert "fits" in inspector.get_table_names()

    command.upgrade(_config(engine), "head")
    assert "autofits" in inspect(engine).get_table_names()
