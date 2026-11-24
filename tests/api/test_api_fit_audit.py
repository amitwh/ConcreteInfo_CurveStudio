"""The fit audit through the API: the column a cross-validation is stored in, what a stored
fit's audit and methods say, and the job that cross-validates a fit."""

import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session

from conftest import linear_csv, make_settings, signup, upload
from curvestudio.discovery import AutoFitResult
from curvestudio.tools import ToolError, all_methods_texts, audit_fit
from curvestudio.workflow import FitBundle, FitSpec
from curvestudio_api.app import create_app
from curvestudio_api.migrations import _config, current_revision, upgrade_to_head
from curvestudio_api.orm import Dataset, Fit, User
from curvestudio_api.storage import decode_payload

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}
SMALL_RUN = {"families": ["polynomial"], "max_parameters": 2, "cv_folds": 3, "n_starts": 1}
"""Few enough candidates for an Auto-fit run to stay inside its job process."""


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


# -- the migration ---------------------------------------------------------------------------------


def fit_columns(engine: Engine) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns("fits")}


def test_migration_0012_round_trip(tmp_path: Path) -> None:
    """``fits.cv`` arrives nullable: a fit stored before the column has ``None``, which says
    that no cross-validation has been run. The downgrade takes the column away and nothing else."""
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    assert current_revision(engine) == "0012"
    assert "cv" in fit_columns(engine)

    command.downgrade(_config(engine), "0011")
    assert current_revision(engine) == "0011"
    assert "cv" not in fit_columns(engine)
    with Session(engine) as db:  # a fit of a database that has not got the column yet
        user = User(email="before@example.org", password_hash="not-a-hash")
        db.add(user)
        db.flush()
        dataset = Dataset(
            user_id=user.id,
            name="line",
            n_rows=0,
            columns=[],
            x_column="x",
            y_column="y",
            fingerprint="f" * 64,
            payload=b"",
        )
        db.add(dataset)
        db.flush()
        fit = Fit(user_id=user.id, dataset_id=dataset.id, spec={}, bundle={}, status="converged")
        db.add(fit)
        db.commit()
        fit_id = fit.id

    upgrade_to_head(engine)
    assert current_revision(engine) == "0012"
    column = next(c for c in inspect(engine).get_columns("fits") if c["name"] == "cv")
    assert column["nullable"] is True
    with Session(engine) as db:  # cv is a deferred column: read it while the session is open
        kept = db.get(Fit, fit_id)
        assert kept is not None and kept.status == "converged"
        assert kept.cv is None
