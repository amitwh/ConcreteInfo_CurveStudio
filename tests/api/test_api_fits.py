"""Fit jobs end to end through the real worker pool: results equal ``fit_dataset``; the
fit, predict, export and delete endpoints; per-handler isolation between users."""

import csv
import io
import time
import uuid
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from conftest import linear_csv, michaelis_csv, signup, upload
from curvestudio.data import XY
from curvestudio.fitting import ParameterEstimate
from curvestudio.workflow import FitBundle, FitSpec, fit_dataset, predict
from curvestudio_api.export import csv_chunks
from curvestudio_api.orm import AuditLog, Job

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}


def table(content: bytes) -> np.ndarray:
    return np.loadtxt(io.BytesIO(content), delimiter=",", skiprows=1, ndmin=2)


def direct_xy(content: bytes, x_name: str = "x", y_name: str = "y") -> XY:
    values = table(content)
    return XY(values[:, 0], values[:, 1], None, x_name, y_name, "direct")


def wait_for_job(
    client: TestClient, headers: dict[str, str], job_id: str, timeout: float = 90
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def submit(
    client: TestClient, headers: dict[str, str], dataset_id: str, spec: dict[str, Any]
) -> str:
    response = client.post(f"{API}/datasets/{dataset_id}/fits", headers=headers, json=spec)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def fit_through_api(
    client: TestClient, headers: dict[str, str], dataset_id: str, spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    job = wait_for_job(client, headers, submit(client, headers, dataset_id, spec))
    assert job["status"] == "succeeded", job
    bundle = client.get(f"{API}/fits/{job['result_ref']}", headers=headers)
    assert bundle.status_code == 200
    return job, bundle.json()


def same_numbers(api: dict[str, Any], direct: FitBundle) -> None:
    expected = direct.model_dump(mode="json")
    assert api["result"]["status"] == expected["result"]["status"]
    assert api["result"]["parameters"] == expected["result"]["parameters"]
    assert api["result"]["stats"] == expected["result"]["stats"]
    assert api["result"]["fitted"] == expected["result"]["fitted"]
    assert api["result"]["residuals"] == expected["result"]["residuals"]
    assert api["summary"] == expected["summary"]
    assert api["curve"] == expected["curve"]
    assert api["confidence_band"] == expected["confidence_band"]
    assert api["prediction_band"] == expected["prediction_band"]
    assert api["equation_text"] == expected["equation_text"]


def test_polynomial_fit_equals_the_direct_call(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    content = linear_csv(40)
    dataset = upload(shared_client, headers, content).json()
    spec = {"model": {"kind": "polynomial", "degree": 2}, "seed": 5}
    job, bundle = fit_through_api(shared_client, headers, dataset["id"], spec)
    assert job["kind"] == "fit"
    assert job["started_at"] and job["finished_at"] and job["error_code"] is None
    direct = fit_dataset(direct_xy(content), FitSpec.model_validate(job["spec"]), compact=True)
    same_numbers(bundle, direct)
    assert bundle["result"]["status"] == "converged"


def test_registered_model_fit_equals_the_direct_call(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    content = michaelis_csv()
    dataset = upload(shared_client, headers, content).json()
    spec = {"model": {"kind": "registered", "id": "michaelis_menten"}}
    job, bundle = fit_through_api(shared_client, headers, dataset["id"], spec)
    direct = fit_dataset(
        direct_xy(content, "s", "v"), FitSpec.model_validate(job["spec"]), compact=True
    )
    same_numbers(bundle, direct)
    names = {p["name"]: p["value"] for p in bundle["result"]["parameters"]}
    assert names["vmax"] == pytest.approx(10.0, rel=0.05)
    assert names["km"] == pytest.approx(2.5, rel=0.1)


def test_custom_equation_fit_equals_the_direct_call(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    content = michaelis_csv()
    dataset = upload(shared_client, headers, content).json()
    spec = {
        "model": {
            "kind": "custom",
            "equation": "y = a*s/(b+s)",
            "parameters": ["a", "b"],
            "variable": "s",
        },
        "initial": {"a": 8.0, "b": 2.0},
    }
    job, bundle = fit_through_api(shared_client, headers, dataset["id"], spec)
    direct = fit_dataset(
        direct_xy(content, "s", "v"), FitSpec.model_validate(job["spec"]), compact=True
    )
    same_numbers(bundle, direct)


def test_the_job_budget_is_written_into_the_stored_spec(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    job_id = submit(shared_client, headers, dataset["id"], POLY1)
    assert (
        shared_client.get(f"{API}/jobs/{job_id}", headers=headers).json()["spec"]["time_limit"]
        == 120.0
    )
    wait_for_job(shared_client, headers, job_id)
    asked = {**POLY1, "time_limit": 5.0}
    job_id = submit(shared_client, headers, dataset["id"], asked)
    assert (
        shared_client.get(f"{API}/jobs/{job_id}", headers=headers).json()["spec"]["time_limit"]
        == 5.0
    )
    wait_for_job(shared_client, headers, job_id)


def test_invalid_equation_is_a_422_with_the_hint(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    response = shared_client.post(
        f"{API}/datasets/{dataset['id']}/fits",
        headers=headers,
        json={"model": {"kind": "custom", "equation": "a*Exp(x)", "parameters": ["a"]}},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["detail"]["code"] == "INVALID_EQUATION"
    assert "did you mean 'exp'" in body["message"]
    # nothing was queued
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.scalar(select(Job).where(Job.dataset_id == uuid.UUID(dataset["id"]))) is None


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        ({"model": {"kind": "registered", "id": "no_such_model"}}, "UNKNOWN_MODEL"),
        ({"model": {"kind": "polynomial", "degree": 2}, "method": "bogus"}, "INVALID_SPEC"),
        ({"model": {"kind": "polynomial", "degree": 2}, "use_weights": True}, "INVALID_SPEC"),
    ],
)
def test_configuration_mistakes_are_422_with_the_workflow_code(
    shared_client: TestClient, spec: dict[str, Any], code: str
) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    response = shared_client.post(
        f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=spec
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == code


def test_malformed_specs_are_422_validation_errors(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    for body in ({}, {"model": {"kind": "polynomial", "degree": 99}}, {"model": {"kind": "magic"}}):
        response = shared_client.post(
            f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=body
        )
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"


def test_non_converged_fit_is_delivered_as_such(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, b"x,y\n1,2\n2,4\n3,6\n").json()
    job, bundle = fit_through_api(
        shared_client, headers, dataset["id"], {"model": {"kind": "polynomial", "degree": 5}}
    )
    assert job["status"] == "succeeded"  # the job ran; the numbers are not a fit
    assert bundle["result"]["status"] != "converged"
    assert bundle["curve"] is None and bundle["curve_reason"]
    assert bundle["summary"] is None and bundle["summary_reason"]
    assert bundle["confidence_band"] is None and bundle["bands_reason"]
    assert bundle["result"]["fitted"] == []
    # nothing can be predicted from it
    fit_id = job["result_ref"]
    predicted = shared_client.post(
        f"{API}/fits/{fit_id}/predict", headers=headers, json={"x": [1.0]}
    )
    assert predicted.status_code == 422
    assert predicted.json()["detail"]["code"] == "FIT_NOT_CONVERGED"
    # the export is still honest: empty cells, flagged
    exported = shared_client.get(f"{API}/fits/{fit_id}/export?format=csv", headers=headers)
    assert exported.status_code == 200
    assert "no parameters: the fit did not converge" in exported.text
    assert "fitted/residual not defined" in exported.text
    summaries = shared_client.get(f"{API}/datasets/{dataset['id']}/fits", headers=headers).json()
    assert [s["status"] for s in summaries] == [bundle["result"]["status"]]


def test_predict_equals_the_direct_call(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    content = michaelis_csv()
    dataset = upload(shared_client, headers, content).json()
    spec = {"model": {"kind": "registered", "id": "michaelis_menten"}}
    job, _ = fit_through_api(shared_client, headers, dataset["id"], spec)
    points = [0.0, 1.0, 5.0, 9.5, 40.0]
    xy = direct_xy(content, "s", "v")
    direct_bundle = fit_dataset(xy, FitSpec.model_validate(job["spec"]), compact=True)
    for band in (None, "confidence", "prediction"):
        response = shared_client.post(
            f"{API}/fits/{job['result_ref']}/predict",
            headers=headers,
            json={"x": points, "band": band},
        )
        assert response.status_code == 200, response.text
        expected = predict(
            direct_bundle, FitSpec.model_validate(job["spec"]), points, band, data=xy
        ).model_dump(mode="json")
        assert response.json() == expected
    assert response.json()["extrapolated"] == [True, False, False, False, True]


def test_predict_limits(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    fit_through_api(shared_client, headers, dataset["id"], POLY1)
    fit_id = shared_client.get(f"{API}/datasets/{dataset['id']}/fits", headers=headers).json()[0][
        "id"
    ]
    ok = shared_client.post(
        f"{API}/fits/{fit_id}/predict", headers=headers, json={"x": [0.5] * 100_000}
    )
    assert ok.status_code == 200
    assert len(ok.json()["y"]) == 100_000
    too_many = shared_client.post(
        f"{API}/fits/{fit_id}/predict", headers=headers, json={"x": [0.5] * 100_001}
    )
    assert too_many.status_code == 422
    assert too_many.json()["error"] == "VALIDATION_ERROR"
    bad_band = shared_client.post(
        f"{API}/fits/{fit_id}/predict", headers=headers, json={"x": [1.0], "band": "wide"}
    )
    assert bad_band.status_code == 422


def test_export_json_and_csv(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    content = linear_csv(12)
    dataset = upload(shared_client, headers, content).json()
    job, bundle = fit_through_api(shared_client, headers, dataset["id"], POLY1)
    fit_id = job["result_ref"]
    as_json = shared_client.get(f"{API}/fits/{fit_id}/export?format=json", headers=headers)
    assert as_json.status_code == 200
    assert "attachment" in as_json.headers["content-disposition"]
    assert as_json.json() == bundle

    as_csv = shared_client.get(f"{API}/fits/{fit_id}/export?format=csv", headers=headers)
    assert as_csv.status_code == 200
    assert as_csv.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(as_csv.text)))
    start = rows.index(["name", "estimate", "std_err", "ci_low", "ci_high", "note"])
    names = [r[0] for r in rows[start + 1 : start + 3]]
    assert names == ["b0", "b1"]
    estimates = [float(r[1]) for r in rows[start + 1 : start + 3]]
    assert estimates == [p["value"] for p in bundle["result"]["parameters"]]
    observations = rows.index(["x", "y", "weight", "fitted", "residual", "note"])
    body = rows[observations + 1 :]
    assert len(body) == 12
    values = table(content)
    assert [float(r[0]) for r in body] == values[:, 0].tolist()
    assert [float(r[3]) for r in body] == bundle["result"]["fitted"]
    assert [float(r[4]) for r in body] == bundle["result"]["residuals"]
    assert all(r[5] == "" for r in body)
    assert (
        shared_client.get(f"{API}/fits/{fit_id}/export?format=xml", headers=headers).status_code
        == 422
    )


def test_csv_export_marks_undefined_cells() -> None:
    """Parameters without a standard error or interval export as empty, flagged cells."""
    xy = direct_xy(linear_csv(10))
    bundle = fit_dataset(xy, FitSpec.model_validate(POLY1))
    bare = tuple(ParameterEstimate(name=p.name, value=p.value) for p in bundle.result.parameters)
    degraded = bundle.model_copy(
        update={"result": bundle.result.model_copy(update={"parameters": bare})}
    )
    rows = list(csv.reader(io.StringIO("".join(csv_chunks(degraded, xy)))))
    start = rows.index(["name", "estimate", "std_err", "ci_low", "ci_high", "note"])
    first = rows[start + 1]
    assert first[2:5] == ["", "", ""]
    assert first[5] == "not defined: std_err, ci_low, ci_high"


def test_delete_fit_and_dataset_cascade(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    first, _ = fit_through_api(shared_client, headers, dataset["id"], POLY1)
    second, _ = fit_through_api(shared_client, headers, dataset["id"], POLY1)
    fits = shared_client.get(f"{API}/datasets/{dataset['id']}/fits", headers=headers).json()
    assert {f["id"] for f in fits} == {first["result_ref"], second["result_ref"]}
    assert (
        shared_client.delete(f"{API}/fits/{first['result_ref']}", headers=headers).status_code
        == 204
    )
    assert (
        shared_client.get(f"{API}/fits/{first['result_ref']}", headers=headers).status_code == 404
    )
    assert (
        shared_client.delete(f"{API}/fits/{first['result_ref']}", headers=headers).status_code
        == 404
    )
    assert (
        shared_client.delete(f"{API}/datasets/{dataset['id']}", headers=headers).status_code == 204
    )
    assert (
        shared_client.get(f"{API}/fits/{second['result_ref']}", headers=headers).status_code == 404
    )
    assert shared_client.get(f"{API}/jobs/{second['id']}", headers=headers).status_code == 404
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        actions = {
            r.action
            for r in db.scalars(select(AuditLog).where(AuditLog.entity_id == first["result_ref"]))
        }
    assert actions == {"fit.create", "fit.delete"}


def test_user_b_cannot_use_user_a_jobs_or_fits(shared_client: TestClient) -> None:
    a = signup(shared_client)
    b = signup(shared_client)
    dataset = upload(shared_client, a, linear_csv()).json()
    job, _ = fit_through_api(shared_client, a, dataset["id"], POLY1)
    fit_id = job["result_ref"]
    attempts = [
        ("GET", f"/jobs/{job['id']}", None),
        ("POST", f"/jobs/{job['id']}/cancel", None),
        ("GET", f"/fits/{fit_id}", None),
        ("DELETE", f"/fits/{fit_id}", None),
        ("POST", f"/fits/{fit_id}/predict", {"x": [1.0]}),
        ("GET", f"/fits/{fit_id}/export?format=json", None),
        ("GET", f"/fits/{fit_id}/export?format=csv", None),
    ]
    for method, path, body in attempts:
        response = shared_client.request(method, API + path, headers=b, json=body)
        assert response.status_code == 404, (method, path, response.text)
        assert response.json()["error"] == "NOT_FOUND"
    # still intact for the owner
    assert shared_client.get(f"{API}/fits/{fit_id}", headers=a).status_code == 200
    assert shared_client.get(f"{API}/jobs/{job['id']}", headers=a).json()["status"] == "succeeded"


def test_models_endpoints_are_public(shared_client: TestClient) -> None:
    everything = shared_client.get(f"{API}/models")
    assert everything.status_code == 200
    ids = {m["id"] for m in everything.json()}
    assert {"michaelis_menten", "polynomial", "gaussian"} <= ids
    growth = shared_client.get(f"{API}/models", params={"category": "polynomial"}).json()
    assert [m["id"] for m in growth] == ["polynomial"]
    found = shared_client.get(f"{API}/models", params={"q": "michaelis"}).json()
    assert [m["id"] for m in found] == ["michaelis_menten"]
    one = shared_client.get(f"{API}/models/michaelis_menten").json()
    assert one["equation"] == "y = vmax*x/(km+x)"
    missing = shared_client.get(f"{API}/models/nope")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "UNKNOWN_MODEL"
