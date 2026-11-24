"""The fit audit through the API: the column a cross-validation is stored in, what a stored
fit's audit and methods say, and the job that cross-validates a fit."""

import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import api_job_workers
import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import Engine, insert, inspect, select
from sqlalchemy.orm import Session

from conftest import linear_csv, make_settings, signup, upload
from curvestudio.discovery import AutoFitResult, subsample
from curvestudio.discovery.engine import SUBSAMPLE_N
from curvestudio.tools import (
    ToolError,
    all_methods_texts,
    audit_fit,
    cross_validate,
)
from curvestudio.tools.audit import DW_EXACT_MAX_N
from curvestudio.workflow import FitBundle, FitSpec, fit_dataset
from curvestudio.workflow.bundle import COMPACT_MAX_POINTS
from curvestudio_api.app import create_app
from curvestudio_api.export_openapi import generate
from curvestudio_api.migrations import _config, current_revision, upgrade_to_head
from curvestudio_api.orm import AuditLog, Dataset, Fit, Job, User
from curvestudio_api.settings import Settings
from curvestudio_api.storage import decode_payload

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}
SMALL_RUN = {"families": ["polynomial"], "max_parameters": 2, "cv_folds": 3, "n_starts": 1}
"""Few enough candidates for an Auto-fit run to stay inside its job process."""
NUMBER = r"\d+(?:\.\d+)?(?:e[+-]?\d+)?"
EXACT_STATISTIC = re.compile(
    rf"Durbin-Watson d = {NUMBER}; p = {NUMBER} \(positive autocorrelation\), {NUMBER} "
    rf"\(negative\)"
)
"""The independence check's ``statistic`` as the exact Durbin-Watson test writes it."""
DECIDED = "Ljung-Box decided: the exact Durbin-Watson test was not run because "


def audit_path(fit_id: str) -> str:
    return f"{API}/fits/{fit_id}/audit"


def methods_path(fit_id: str) -> str:
    return f"{API}/fits/{fit_id}/methods"


def check_of(audit: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(check for check in audit["checks"] if check["id"] == check_id)


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


def stored_fit(client: TestClient, fit_id: str) -> dict[str, Any]:
    """A fit as the API stores it: its bundle and specification as the core's types, the
    cross-validation on it (``None`` until one has run) and its dataset's table."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        fit = db.get(Fit, uuid.UUID(fit_id))
        dataset = db.get(Dataset, fit.dataset_id)
        return {
            "bundle": FitBundle.model_validate(fit.bundle),
            "spec": FitSpec.model_validate(fit.spec),
            "cv": fit.cv,
            "dataset_id": str(fit.dataset_id),
            "xy": decode_payload(
                dataset.payload,
                x_name=dataset.x_column,
                y_name=dataset.y_column,
                fingerprint=dataset.fingerprint,
            ),
        }


@pytest.fixture(scope="module")
def fitted(shared_client: TestClient) -> dict[str, Any]:
    """One account's straight-line dataset with two converged fits, and a fit that did not
    converge (a degree-5 polynomial through three points)."""
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv(40)).json()
    first = fit_through_api(shared_client, headers, dataset["id"], POLY1)
    second = fit_through_api(shared_client, headers, dataset["id"], POLY1)
    tiny = upload(shared_client, headers, b"x,y\n1,2\n2,4\n3,6\n").json()
    failed = fit_through_api(
        shared_client, headers, tiny["id"], {"model": {"kind": "polynomial", "degree": 5}}
    )
    return {
        "client": shared_client,
        "headers": headers,
        "dataset": dataset,
        "first": first,
        "second": second,
        "failed": failed,
    }


# -- a stored fit's audit and methods --------------------------------------------------------------


def test_the_audit_is_what_the_tool_gives_for_the_stored_fit(fitted: dict[str, Any]) -> None:
    fit_id = fitted["first"]["result_ref"]
    response = fitted["client"].get(audit_path(fit_id), headers=fitted["headers"])
    assert response.status_code == 200, response.text
    audit = response.json()
    stored = stored_fit(fitted["client"], fit_id)
    assert stored["cv"] is None  # nothing has been cross-validated yet
    assert audit == audit_fit(stored["bundle"], spec=stored["spec"]).model_dump(mode="json")
    assert audit["n_observations"] == 40 and audit["n_parameters"] == 2
    assert check_of(audit, "cross_validation")["verdict"] == "not_applicable"
    assert check_of(audit, "simpler_model")["verdict"] == "not_applicable"
    # every field of a check is always there, so a client never meets a missing key
    assert set(check_of(audit, "converged")) == {
        "id",
        "title",
        "verdict",
        "statistic",
        "threshold",
        "evidence",
        "explanation",
        "next_step",
    }


def test_the_methods_name_the_model_in_three_styles(fitted: dict[str, Any]) -> None:
    fit_id = fitted["first"]["result_ref"]
    response = fitted["client"].get(methods_path(fit_id), headers=fitted["headers"])
    assert response.status_code == 200, response.text
    texts = response.json()
    assert set(texts) == {"text", "markdown", "latex"}
    stored = stored_fit(fitted["client"], fit_id)
    for style, text in all_methods_texts(stored["bundle"], spec=stored["spec"]).items():
        assert texts[style] == text.model_dump(mode="json")
        assert "A polynomial of degree 1 was fitted to 40 observations" in texts[style]["paragraph"]
    # the same estimates, laid out for each style
    assert texts["latex"]["equation"].startswith("\\[")
    assert "95 \\%" in texts["latex"]["paragraph"]
    assert texts["markdown"]["table"].startswith("| Parameter | Estimate | Std. error |")
    assert "|" not in texts["text"]["table"]


def test_a_fit_that_did_not_converge_audits_to_fail_and_has_no_methods(
    fitted: dict[str, Any],
) -> None:
    fit_id = fitted["failed"]["result_ref"]
    response = fitted["client"].get(audit_path(fit_id), headers=fitted["headers"])
    assert response.status_code == 200, response.text
    audit = response.json()
    assert audit["overall"] == "fail"
    assert check_of(audit, "converged")["verdict"] == "fail"
    stored = stored_fit(fitted["client"], fit_id)
    assert audit == audit_fit(stored["bundle"], spec=stored["spec"]).model_dump(mode="json")

    refused = fitted["client"].get(methods_path(fit_id), headers=fitted["headers"])
    assert refused.status_code == 422, refused.text
    with pytest.raises(ToolError) as expected:
        all_methods_texts(stored["bundle"], spec=stored["spec"])
    body = refused.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["message"] == expected.value.message
    assert body["detail"] == {"code": "FIT_NOT_CONVERGED_INPUT"}


@pytest.fixture(scope="module")
def opened(fitted: dict[str, Any]) -> dict[str, Any]:
    """A fit opened from the best row of a finished Auto-fit run."""
    client, headers = fitted["client"], fitted["headers"]
    dataset = upload(client, headers, linear_csv(30)).json()
    started = client.post(
        f"{API}/datasets/{dataset['id']}/autofit", headers=headers, json=SMALL_RUN
    )
    assert started.status_code == 202, started.text
    run = wait_for_job(client, headers, started.json()["job_id"])
    assert run["status"] == "succeeded", run
    autofit_id = run["result_id"]
    ranking = AutoFitResult.model_validate(
        client.get(f"{API}/autofits/{autofit_id}", headers=headers).json()["result"]
    )
    row = ranking.ranked[0].model_id
    opened_row = client.post(f"{API}/autofits/{autofit_id}/rows/{row}/open", headers=headers)
    assert opened_row.status_code == 202, opened_row.text
    job = wait_for_job(client, headers, opened_row.json()["job_id"])
    assert job["status"] == "succeeded", job
    return {"autofit_id": autofit_id, "fit_id": job["result_ref"]}


def test_an_opened_row_is_audited_against_its_run_until_the_run_is_deleted(
    fitted: dict[str, Any], opened: dict[str, Any]
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id, autofit_id = opened["fit_id"], opened["autofit_id"]
    ranking = AutoFitResult.model_validate(
        client.get(f"{API}/autofits/{autofit_id}", headers=headers).json()["result"]
    )
    stored = stored_fit(client, fit_id)

    response = client.get(audit_path(fit_id), headers=headers)
    assert response.status_code == 200, response.text
    expected = audit_fit(stored["bundle"], spec=stored["spec"], ranking=ranking)
    assert response.json() == expected.model_dump(mode="json")
    simpler = check_of(response.json(), "simpler_model")
    assert simpler["verdict"] in ("pass", "caution")  # the ranking decides it
    assert (response.json()["simpler_model"] is not None) == (simpler["verdict"] == "caution")

    assert client.delete(f"{API}/autofits/{autofit_id}", headers=headers).status_code == 204
    response = client.get(audit_path(fit_id), headers=headers)
    assert response.status_code == 200, response.text
    expected = audit_fit(stored["bundle"], spec=stored["spec"])
    assert response.json() == expected.model_dump(mode="json")
    assert check_of(response.json(), "simpler_model")["verdict"] == "not_applicable"
    assert response.json()["simpler_model"] is None
    # the fit itself is untouched
    assert client.get(f"{API}/fits/{fit_id}", headers=headers).status_code == 200


# -- the cross-validation job ----------------------------------------------------------------------


def cross_validation_path(fit_id: str) -> str:
    return f"{API}/fits/{fit_id}/audit/cross-validation"


def start_cross_validation(
    client: TestClient, headers: dict[str, str], fit_id: str, body: dict[str, Any] | None = None
) -> str:
    response = client.post(cross_validation_path(fit_id), headers=headers, json=body or {})
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def cross_validation_specs(client: TestClient, fit_id: str) -> list[dict[str, Any]]:
    """The specs of every cross-validation job queued for ``fit_id``, finished or not."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        jobs = db.scalars(select(Job).where(Job.kind == "audit"))
        return [dict(job.spec) for job in jobs if job.spec["fit_id"] == fit_id]


def test_the_cross_validation_job_stores_its_result_on_the_fit(fitted: dict[str, Any]) -> None:
    client, headers, dataset = fitted["client"], fitted["headers"], fitted["dataset"]
    fit_id = fitted["first"]["result_ref"]
    assert stored_fit(client, fit_id)["cv"] is None

    started = start_cross_validation(client, headers, fit_id)  # five folds by default
    job = wait_for_job(client, headers, started)
    assert job["status"] == "succeeded", job
    assert job["kind"] == "audit"
    assert job["spec"] == {"fit_id": fit_id, "folds": 5}
    assert job["result_id"] == fit_id and job["result_ref"] == dataset["id"]

    stored = stored_fit(client, fit_id)
    expected = cross_validate(stored["bundle"], stored["spec"], data=stored["xy"], folds=5)
    assert stored["cv"] == expected.model_dump(mode="json")
    audited = client.get(audit_path(fit_id), headers=headers)
    assert audited.status_code == 200, audited.text
    expected_audit = audit_fit(stored["bundle"], spec=stored["spec"], cv=expected)
    assert audited.json() == expected_audit.model_dump(mode="json")
    assert check_of(audited.json(), "cross_validation")["verdict"] != "not_applicable"
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        actions = {
            row.action for row in db.scalars(select(AuditLog).where(AuditLog.entity_id == fit_id))
        }
    assert actions == {"fit.create", "fit.cross_validate"}

    # leave-one-out replaces it, once no job is active
    started = start_cross_validation(client, headers, fit_id, {"folds": "loo"})
    job = wait_for_job(client, headers, started)
    assert job["status"] == "succeeded", job
    assert job["spec"] == {"fit_id": fit_id, "folds": "loo"}
    stored = stored_fit(client, fit_id)
    expected = cross_validate(stored["bundle"], stored["spec"], data=stored["xy"], folds="loo")
    assert stored["cv"] == expected.model_dump(mode="json")
    assert stored["cv"]["folds"] == 40


def test_the_cross_validation_job_uses_no_quota(fitted: dict[str, Any]) -> None:
    client, headers = fitted["client"], fitted["headers"]
    second_fit = fitted["second"]["result_ref"]

    def fits_stored() -> int:
        return client.get(f"{API}/me", headers=headers).json()["usage"]["fits"]

    before = fits_stored()
    job = wait_for_job(client, headers, start_cross_validation(client, headers, second_fit))
    assert job["status"] == "succeeded", job
    assert fits_stored() == before


@pytest.fixture(scope="module")
def large(fitted: dict[str, Any]) -> dict[str, Any]:
    """A dataset past the compact limit, so that a stored fit holds only a subsample of it.

    ``whole`` is an ordinary fit of the whole table. ``on_subsample`` is what opening a row of
    an Auto-fit run on so large a dataset stores when the run fitted that row on the dataset's
    subsample: a fit whose own data is that subsample, carried whole.
    """
    client, headers = fitted["client"], fitted["headers"]
    dataset = upload(client, headers, linear_csv(COMPACT_MAX_POINTS + 500)).json()
    whole = fit_through_api(client, headers, dataset["id"], POLY1)
    stored = stored_fit(client, whole["result_ref"])
    assert stored["bundle"].data.subsampled
    spec = FitSpec.model_validate(POLY1)
    bundle = fit_dataset(subsample(stored["xy"]), spec, compact=True)
    user_id = uuid.UUID(client.get(f"{API}/me", headers=headers).json()["id"])
    row = Fit(
        user_id=user_id,
        dataset_id=uuid.UUID(dataset["id"]),
        spec=spec.model_dump(mode="json"),
        bundle=bundle.model_dump(mode="json"),
        status=bundle.result.status.value,
    )
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(row)
        db.commit()
        on_subsample = str(row.id)
    return {"whole": whole["result_ref"], "on_subsample": on_subsample}


def test_a_compact_bundle_is_cross_validated_on_the_whole_table(
    fitted: dict[str, Any], large: dict[str, Any]
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = large["whole"]
    stored = stored_fit(client, fit_id)
    assert len(stored["bundle"].data.x) == COMPACT_MAX_POINTS < len(stored["xy"].x)
    with pytest.raises(ToolError) as refused:  # what the bundle carries is only a subsample of it
        cross_validate(stored["bundle"], stored["spec"], folds=5)
    assert refused.value.code == "DATA_REQUIRED"

    job = wait_for_job(client, headers, start_cross_validation(client, headers, fit_id))
    assert job["status"] == "succeeded", job
    expected = cross_validate(stored["bundle"], stored["spec"], data=stored["xy"], folds=5)
    assert stored_fit(client, fit_id)["cv"] == expected.model_dump(mode="json")


def test_a_fit_made_on_a_subsample_is_cross_validated_on_that_subsample(
    fitted: dict[str, Any], large: dict[str, Any]
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = large["on_subsample"]
    stored = stored_fit(client, fit_id)
    assert not stored["bundle"].data.subsampled
    assert stored["bundle"].data.n_total == SUBSAMPLE_N < len(stored["xy"].x)
    with pytest.raises(ToolError) as refused:  # the whole table is not what it was computed from
        cross_validate(stored["bundle"], stored["spec"], data=stored["xy"], folds=5)
    assert refused.value.code == "DATA_MISMATCH"

    job = wait_for_job(client, headers, start_cross_validation(client, headers, fit_id))
    assert job["status"] == "succeeded", job
    expected = cross_validate(stored["bundle"], stored["spec"], folds=5)
    assert stored_fit(client, fit_id)["cv"] == expected.model_dump(mode="json")


@pytest.fixture(scope="module")
def marked_compact(fitted: dict[str, Any]) -> str:
    """A stored fit that carries all its points but is marked as holding only a subsample of
    them; its id.

    A compact bundle from the workflow has more than ``COMPACT_MAX_POINTS`` points, more than
    the exact Durbin-Watson test is computed for, so no fit the API stores shows what the mark
    alone does. It is set by hand on a small fit, as the core's own tests set it.
    """
    client, headers = fitted["client"], fitted["headers"]
    stored = stored_fit(client, fitted["first"]["result_ref"])
    bundle = stored["bundle"].model_copy(
        update={"data": stored["bundle"].data.model_copy(update={"subsampled": True})}
    )
    user_id = uuid.UUID(client.get(f"{API}/me", headers=headers).json()["id"])
    row = Fit(
        user_id=user_id,
        dataset_id=uuid.UUID(fitted["dataset"]["id"]),
        spec=stored["spec"].model_dump(mode="json"),
        bundle=bundle.model_dump(mode="json"),
        status=bundle.result.status.value,
    )
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(row)
        db.commit()
        return str(row.id)


def test_the_audit_route_needs_no_data_while_the_exact_test_stays_under_the_compact_limit() -> None:
    """The audit route hands the audit no data, which is right only while a compact fit (one
    above ``COMPACT_MAX_POINTS`` rows) is also past the exact Durbin-Watson test's own limit. If
    this fails, the route must pass a compact fit's dataset table, as the cross-validation job
    does (``audit_fit(..., data=...)``), with a test that shows the exact test then deciding."""
    assert DW_EXACT_MAX_N < COMPACT_MAX_POINTS


def test_an_impossible_fold_count_is_refused_before_a_job_is_queued(
    fitted: dict[str, Any],
) -> None:
    client, headers = fitted["client"], fitted["headers"]
    second_fit = fitted["second"]["result_ref"]
    queued = len(cross_validation_specs(client, second_fit))
    refused = client.post(cross_validation_path(second_fit), headers=headers, json={"folds": 1})
    assert refused.status_code == 422, refused.text
    body = refused.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["detail"] == {"code": "INVALID_FOLDS"}
    assert "folds" in body["message"]
    for body in ({"folds": 0}, {"folds": "many"}, {"folds": 2.5}, {"folds": 2, "seed": 1}):
        refused = client.post(cross_validation_path(second_fit), headers=headers, json=body)
        assert refused.status_code == 422, (body, refused.text)
    assert len(cross_validation_specs(client, second_fit)) == queued  # nothing was queued

    # two folds is the least a cross-validation has
    job = wait_for_job(
        client, headers, start_cross_validation(client, headers, second_fit, {"folds": 2})
    )
    assert job["status"] == "succeeded", job


def test_a_refusal_inside_the_job_is_the_jobs_failure(fitted: dict[str, Any]) -> None:
    client, headers = fitted["client"], fitted["headers"]
    second_fit = fitted["second"]["result_ref"]
    # a fit that did not converge has no model to cross-validate
    fit_id = fitted["failed"]["result_ref"]
    job = wait_for_job(client, headers, start_cross_validation(client, headers, fit_id))
    stored = stored_fit(client, fit_id)
    with pytest.raises(ToolError) as refused:
        cross_validate(stored["bundle"], stored["spec"], data=stored["xy"], folds=5)
    assert job["status"] == "failed"
    assert job["error_code"] == refused.value.code == "FIT_NOT_CONVERGED_INPUT"
    assert job["error_message"] == refused.value.message
    assert job["result_id"] is None and job["result_ref"] is None

    # more folds than points is the tool's refusal as well
    before = stored_fit(client, second_fit)["cv"]
    job = wait_for_job(
        client, headers, start_cross_validation(client, headers, second_fit, {"folds": 1000})
    )
    assert job["status"] == "failed" and job["error_code"] == "INVALID_FOLDS", job
    assert stored_fit(client, second_fit)["cv"] == before


def test_the_audit_routes_hide_another_users_fit_and_need_a_token(fitted: dict[str, Any]) -> None:
    client, headers = fitted["client"], fitted["headers"]
    fit_id = fitted["first"]["result_ref"]
    routes = (
        ("GET", audit_path(fit_id)),
        ("GET", methods_path(fit_id)),
        ("POST", cross_validation_path(fit_id)),
    )
    queued = len(cross_validation_specs(client, fit_id))
    stranger = signup(client)
    for method, path in routes:
        body = {"folds": 5} if method == "POST" else None
        anonymous = client.request(method, path)
        assert anonymous.status_code == 401, (method, path, anonymous.text)
        assert anonymous.json()["error"] == "UNAUTHORIZED"
        foreign = client.request(method, path, headers=stranger, json=body)
        assert foreign.status_code == 404, (method, path, foreign.text)
        assert foreign.json()["error"] == "NOT_FOUND"
        unknown = path.replace(fit_id, str(uuid.uuid4()))
        missing = client.request(method, unknown, headers=headers, json=body)
        assert missing.status_code == 404, (method, path, missing.text)
    # nothing was started for the owner, whose fit is untouched
    assert len(cross_validation_specs(client, fit_id)) == queued
    assert client.get(audit_path(fit_id), headers=headers).status_code == 200


def test_openapi_documents_the_audit_endpoints() -> None:
    document = generate()
    fit = "/api/v1/fits/{fit_id}"
    assert set(document["paths"][f"{fit}/audit"]) == {"get"}
    assert set(document["paths"][f"{fit}/methods"]) == {"get"}
    started = document["paths"][f"{fit}/audit/cross-validation"]
    assert set(started) == {"post"} and "202" in started["post"]["responses"]
    schemas = document["components"]["schemas"]
    assert {
        "AuditCheckOut",
        "ExcludedRowOut",
        "FitAuditOut",
        "FitMethodsOut",
        "MethodsTextOut",
        "SimplerModelOut",
    } <= set(schemas)
    folds = schemas["CrossValidationIn"]["properties"]["folds"]
    assert folds["default"] == 5 and "loo" in str(folds)
    assert "audit" in str(schemas["JobOut"]["properties"]["kind"])


# -- the job in flight, with workers that do not finish --------------------------------------------


def test_a_second_cross_validation_of_a_fit_is_a_409_while_one_is_queued_or_running(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(), audit_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    first = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    second = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]

    # the only worker is held by the first fit's job, so the second fit's waits in the queue
    holding = start_cross_validation(client, headers, first)
    wait_for_job(client, headers, holding, statuses=("running",))
    queued = start_cross_validation(client, headers, second)
    assert client.get(f"{API}/jobs/{queued}", headers=headers).json()["status"] == "queued"
    again = client.post(cross_validation_path(second), headers=headers, json={"folds": "loo"})
    assert again.status_code == 409, again.text
    assert again.json()["error"] == "CONFLICT"
    assert again.json()["detail"] == {"code": "CROSS_VALIDATION_RUNNING", "job_id": queued}
    running = client.post(cross_validation_path(first), headers=headers, json={})
    assert running.status_code == 409 and running.json()["detail"]["job_id"] == holding

    # once nothing waits, the job answered is still the one to follow
    assert client.post(f"{API}/jobs/{queued}/cancel", headers=headers).status_code == 200
    assert client.post(f"{API}/jobs/{holding}/cancel", headers=headers).status_code == 200
    active = start_cross_validation(client, headers, second)
    wait_for_job(client, headers, active, statuses=("running",))
    again = client.post(cross_validation_path(second), headers=headers, json={})
    assert again.status_code == 409 and again.json()["detail"]["job_id"] == active
    # another fit is not in the way
    start_cross_validation(client, headers, first)

    # with nothing active, a new one is taken
    assert client.post(f"{API}/jobs/{active}/cancel", headers=headers).status_code == 200
    accepted = client.post(cross_validation_path(second), headers=headers, json={})
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["job_id"] != active


def test_a_cross_validation_in_flight_does_not_count_against_the_fit_quota(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(
        settings_factory(job_workers=2, quota_max_fits=2),
        audit_worker=api_job_workers.sleeping_worker,
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    first = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    running = start_cross_validation(client, headers, first)
    wait_for_job(client, headers, running, statuses=("running",))

    # one fit stored and a job in flight that stores no fit: the second of two is still allowed
    second = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    # the quota is used up, and a cross-validation is still allowed: it is no fit
    refused = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=POLY1)
    assert refused.status_code == 409 and refused.json()["error"] == "QUOTA_EXCEEDED", refused.text
    start_cross_validation(client, headers, second)


def test_an_audit_job_that_overruns_its_timeout_says_cross_validation(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(
        settings_factory(job_timeout_s=1.0, job_grace_s=0.2),
        audit_worker=api_job_workers.sleeping_worker,
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    fit_id = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    job = wait_for_job(client, headers, start_cross_validation(client, headers, fit_id))
    assert job["status"] == "failed" and job["error_code"] == "TIMEOUT", job
    assert "the cross-validation did not finish within 1 s" in job["error_message"]


def failed_audit_job(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    """Start the cross-validation of a fresh fit and wait for its job to fail."""
    dataset = upload(client, headers, linear_csv()).json()
    fit_id = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    job = wait_for_job(client, headers, start_cross_validation(client, headers, fit_id))
    assert job["status"] == "failed", job
    return job


def test_a_crashed_audit_worker_fails_the_job_naming_the_cross_validation(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(), audit_worker=api_job_workers.crashing_worker)
    job = failed_audit_job(client, signup(client))
    assert job["error_code"] == "WORKER_CRASHED"
    assert job["error_message"] == (
        "the worker process died while cross-validating; the job did not complete"
    )


def test_an_oversized_cross_validation_result_fails_the_job_naming_it(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(
        settings_factory(job_max_result_bytes=200_000),
        audit_worker=api_job_workers.big_audit_result_worker,
    )
    job = failed_audit_job(client, signup(client))
    assert job["error_code"] == "RESULT_TOO_LARGE"
    assert job["error_message"] == (
        "the cross-validation result is larger than 200,000 bytes; use fewer folds"
    )


def test_a_cross_validation_result_that_cannot_be_stored_fails_the_job_naming_it(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    # the worker's result carries no cross-validation, so there is nothing to store on the fit
    client = client_factory(settings_factory(), audit_worker=api_job_workers.big_result_worker)
    job = failed_audit_job(client, signup(client))
    assert job["error_code"] == "INTERNAL_ERROR"
    assert (
        job["error_message"] == "the cross-validation finished but its result could not be stored"
    )


def test_deleting_a_fit_cancels_its_cross_validation(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    """The audit worker waits before it runs, so the job is still in flight when the fit goes."""
    client = client_factory(settings_factory(), audit_worker=api_job_workers.slow_audit_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    first = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]
    second = fit_through_api(client, headers, dataset["id"], POLY1)["result_ref"]

    # deleted while its job waits in the queue: the job is not started
    holding = start_cross_validation(client, headers, first)
    wait_for_job(client, headers, holding, statuses=("running",))
    waiting = start_cross_validation(client, headers, second)
    assert client.get(f"{API}/jobs/{waiting}", headers=headers).json()["status"] == "queued"
    assert client.delete(f"{API}/fits/{second}", headers=headers).status_code == 204
    job = wait_for_job(client, headers, waiting)
    assert job["status"] == "cancelled" and job["error_code"] is None, job
    assert job["result_id"] is None
    # the other fit's job, on the same dataset, is left alone and runs to its end
    assert wait_for_job(client, headers, holding)["status"] == "succeeded"

    # deleted while its job runs: the worker is stopped
    running = start_cross_validation(client, headers, first)
    wait_for_job(client, headers, running, statuses=("running",))
    assert client.delete(f"{API}/fits/{first}", headers=headers).status_code == 204
    job = wait_for_job(client, headers, running)
    assert job["status"] == "cancelled" and job["error_code"] is None, job
    assert job["result_id"] is None

    # no fit row remains
    for fit_id in (first, second):
        assert client.get(f"{API}/fits/{fit_id}", headers=headers).status_code == 404
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.get(Fit, uuid.UUID(first)) is None
        assert db.get(Fit, uuid.UUID(second)) is None


# -- the migration ---------------------------------------------------------------------------------


def fit_columns(engine: Engine) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns("fits")}


def test_migration_0012_round_trip(tmp_path: Path) -> None:
    """``fits.cv`` arrives nullable: a fit stored before the column has ``None``, which says
    that no cross-validation has been run. The downgrade takes the column away and nothing else."""
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    assert current_revision(engine) == "0014"
    assert "cv" in fit_columns(engine)

    command.downgrade(_config(engine), "0011")
    assert current_revision(engine) == "0011"
    assert "cv" not in fit_columns(engine)
    with Session(engine) as db:  # a fit of a database that has not got the column yet
        user = User(email="before@example.org", password_hash="not-a-hash")
        db.add(user)
        db.flush()
        # A statement, not an object of the session: an object names every column of the
        # model, the dataset columns this database has not got yet included.
        dataset_id = uuid.uuid4()
        db.execute(
            insert(Dataset).values(
                id=dataset_id,
                user_id=user.id,
                name="line",
                n_rows=0,
                columns=[],
                x_column="x",
                y_column="y",
                fingerprint="f" * 64,
                payload=b"",
            )
        )
        fit = Fit(user_id=user.id, dataset_id=dataset_id, spec={}, bundle={}, status="converged")
        db.add(fit)
        db.commit()
        fit_id = fit.id

    upgrade_to_head(engine)
    assert current_revision(engine) == "0014"
    column = next(c for c in inspect(engine).get_columns("fits") if c["name"] == "cv")
    assert column["nullable"] is True
    with Session(engine) as db:  # cv is a deferred column: read it while the session is open
        kept = db.get(Fit, fit_id)
        assert kept is not None and kept.status == "converged"
        assert kept.cv is None
