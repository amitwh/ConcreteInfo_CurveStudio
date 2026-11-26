"""The XLSX export of a stored fit: a real workbook (a zip, with its plot sheet), strict
format validation, and a loud failure — the error envelope, never an empty file — when the
stored bundle is unusable. JSON and CSV are pinned in test_api_fits.py.

The workbook also carries the fit's methods paragraph and its audit: read, as the audit
route reads them, from the stored specification, the Auto-fit ranking the fit was opened
from and the cross-validation stored on it."""

import io
import time
import uuid
from typing import Any

import numpy as np
from api_test_helpers import linear_csv, signup, upload
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from curvestudio.discovery import AutoFitResult
from curvestudio.tools import CrossValidation, audit_fit, methods_text
from curvestudio.workflow import FitBundle, FitSpec
from curvestudio_api.orm import Fit

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}
POLY5 = {"model": {"kind": "polynomial", "degree": 5}}
SMALL_RUN = {"families": ["polynomial"], "max_parameters": 2, "cv_folds": 3, "n_starts": 1}
"""Few enough candidates for an Auto-fit run to stay inside its job process."""
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def finished(client: TestClient, headers: dict[str, str], job_id: str) -> dict[str, Any]:
    """A job once it has succeeded, failed or been cancelled (90 seconds at most)."""
    deadline = time.monotonic() + 90
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stayed {job.get('status')!r}")


def fit_through_api(
    client: TestClient, headers: dict[str, str], content: bytes, spec: dict[str, Any]
) -> str:
    """Upload ``content``, fit it with ``spec`` through the real worker pool and return the
    stored fit's id; the job must succeed (a fit that does not converge is still a job that
    succeeded)."""
    dataset = upload(client, headers, content).json()
    created = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=spec)
    assert created.status_code == 202, created.text
    job = finished(client, headers, created.json()["job_id"])
    assert job["status"] == "succeeded", job
    return str(job["result_ref"])


def store_fit(client: TestClient, headers: dict[str, str]) -> str:
    """Fit a small linear dataset through the real worker pool; returns the fit id."""
    return fit_through_api(client, headers, linear_csv(12), POLY1)


def test_xlsx_export_is_a_workbook_with_the_right_name(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    fit_id = store_fit(shared_client, headers)
    response = shared_client.get(f"{API}/fits/{fit_id}/export?format=xlsx", headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"] == f'attachment; filename="fit-{fit_id}.xlsx"'
    assert response.headers["content-type"] == XLSX_MEDIA_TYPE
    assert len(response.content) > 0
    assert response.content[:2] == b"PK"  # a zip: the openpyxl container
    workbook = load_workbook(io.BytesIO(response.content))
    assert "Summary" in workbook.sheetnames
    assert "Plot" in workbook.sheetnames  # the plot sheet the reporting package ships


def test_unlisted_format_is_a_validation_error(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    fit_id = store_fit(shared_client, headers)
    response = shared_client.get(f"{API}/fits/{fit_id}/export?format=parquet", headers=headers)
    assert response.status_code == 422
    assert response.json()["error"] == "VALIDATION_ERROR"


def test_xlsx_of_a_fit_without_a_bundle_is_the_error_envelope(shared_client: TestClient) -> None:
    """A stored bundle the writers cannot read fails loudly: the same envelope the CSV path
    yields for the same damage, never a 200 with an empty file."""
    headers = signup(shared_client)
    fit_id = store_fit(shared_client, headers)
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        fit = db.get(Fit, uuid.UUID(fit_id))
        fit.bundle = {}  # type: ignore[assignment]
        db.commit()
    as_csv = shared_client.get(f"{API}/fits/{fit_id}/export?format=csv", headers=headers)
    as_xlsx = shared_client.get(f"{API}/fits/{fit_id}/export?format=xlsx", headers=headers)
    assert as_xlsx.status_code >= 400, as_xlsx.text
    assert as_xlsx.json()["error"] == as_csv.json()["error"]
    assert as_xlsx.status_code == as_csv.status_code  # xlsx mirrors the csv path


# -- what the workbook says of the fit: its methods paragraph and its audit ----------------------


def stored_fit(
    client: TestClient, fit_id: str
) -> tuple[FitBundle, FitSpec, CrossValidation | None]:
    """A stored fit as the core's types: its bundle, its specification and the cross-validation
    stored on it (``None`` until one has run)."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        fit = db.get(Fit, uuid.UUID(fit_id))
        assert fit is not None
        cv = None if fit.cv is None else CrossValidation.model_validate(fit.cv)
        return FitBundle.model_validate(fit.bundle), FitSpec.model_validate(fit.spec), cv


def summary_rows(workbook: bytes) -> list[list]:
    """The workbook's Summary sheet, one list per row."""
    sheet = load_workbook(io.BytesIO(workbook))["Summary"]
    return [list(row) for row in sheet.iter_rows(values_only=True)]


def summary_values(workbook: bytes) -> dict[str, object]:
    """The Summary sheet's labelled rows, label to value."""
    return {row[0]: row[1] for row in summary_rows(workbook) if row[0]}


def test_the_workbook_carries_the_methods_paragraph_and_the_audit(
    shared_client: TestClient,
) -> None:
    headers = signup(shared_client)
    fit_id = store_fit(shared_client, headers)
    response = shared_client.get(f"{API}/fits/{fit_id}/export?format=xlsx", headers=headers)
    assert response.status_code == 200, response.text
    bundle, spec, cv = stored_fit(shared_client, fit_id)
    assert cv is None
    audit = audit_fit(bundle, spec=spec)
    texts = methods_text(bundle, spec=spec, style="text")
    # the exact Durbin-Watson test needs the specification: its name in the fragment shows
    # that the export passed the stored one
    assert "Durbin-Watson" in audit.methods_fragment
    assert "Ljung-Box" in audit_fit(bundle).methods_fragment
    values = summary_values(response.content)
    assert values["methods paragraph"] == f"{texts.paragraph} {audit.methods_fragment}"
    assert values["Fit audit"] == audit.summary


def clean_line(seed: int) -> bytes:
    """A straight line through 30 points with Gaussian noise: every audit check passes."""
    rng = np.random.default_rng(seed)
    x = np.arange(30, dtype=float)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, 30)
    lines = ["x,y", *(f"{float(a)!r},{float(b)!r}" for a, b in zip(x, y, strict=True))]
    return ("\n".join(lines) + "\n").encode()


def test_the_audit_in_the_workbook_reads_the_ranking_and_the_stored_cross_validation(
    shared_client: TestClient,
) -> None:
    """The export's audit is the audit route's: it reads the Auto-fit ranking the fit was
    opened from and the cross-validation stored on the fit. On a clean fit each of the two
    turns a "could not be run" into a passed check, so the summary row tells them apart."""
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, clean_line(5)).json()
    started = shared_client.post(
        f"{API}/datasets/{dataset['id']}/autofit", headers=headers, json=SMALL_RUN
    )
    assert started.status_code == 202, started.text
    autofit_id = finished(shared_client, headers, started.json()["job_id"])["result_id"]
    ranking = AutoFitResult.model_validate(
        shared_client.get(f"{API}/autofits/{autofit_id}", headers=headers).json()["result"]
    )
    opened = shared_client.post(
        f"{API}/autofits/{autofit_id}/rows/{ranking.ranked[0].model_id}/open", headers=headers
    )
    assert opened.status_code == 202, opened.text
    fit_id = finished(shared_client, headers, opened.json()["job_id"])["result_ref"]
    cross_validated = shared_client.post(
        f"{API}/fits/{fit_id}/audit/cross-validation", headers=headers, json={}
    )
    assert cross_validated.status_code == 202, cross_validated.text
    job = finished(shared_client, headers, cross_validated.json()["job_id"])
    assert job["status"] == "succeeded", job

    response = shared_client.get(f"{API}/fits/{fit_id}/export?format=xlsx", headers=headers)
    assert response.status_code == 200, response.text
    bundle, spec, cv = stored_fit(shared_client, str(fit_id))
    assert cv is not None
    expected = audit_fit(bundle, spec=spec, ranking=ranking, cv=cv)
    assert expected.summary == "Every check that could run passed."
    assert "2 could not be run" in audit_fit(bundle, spec=spec).summary
    assert "1 could not be run" in audit_fit(bundle, spec=spec, ranking=ranking).summary
    assert "1 could not be run" in audit_fit(bundle, spec=spec, cv=cv).summary
    values = summary_values(response.content)
    assert values["Fit audit"] == expected.summary
    paragraph = methods_text(bundle, spec=spec, style="text").paragraph
    assert values["methods paragraph"] == f"{paragraph} {expected.methods_fragment}"
    assert "Cross-validation was not run" not in str(values["methods paragraph"])
    rows = summary_rows(response.content)
    after = [row[0] for row in rows].index("Fit audit") + 1
    assert rows[after][0] == "assumptions and limitations", "no check is flagged: one row"


def test_a_fit_that_did_not_converge_still_exports_its_workbook(shared_client: TestClient) -> None:
    """The tools write no methods text for such a fit: the export goes through without it."""
    headers = signup(shared_client)
    # a degree-5 polynomial through three points
    fit_id = fit_through_api(shared_client, headers, b"x,y\n1,2\n2,4\n3,6\n", POLY5)
    response = shared_client.get(f"{API}/fits/{fit_id}/export?format=xlsx", headers=headers)
    assert response.status_code == 200, response.text
    values = summary_values(response.content)
    assert values["status"] != "converged"
    assert "methods paragraph" not in values and "Fit audit" not in values
