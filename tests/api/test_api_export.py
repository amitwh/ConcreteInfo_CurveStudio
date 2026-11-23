"""The XLSX export of a stored fit: a real workbook (a zip, with its plot sheet), strict
format validation, and a loud failure — the error envelope, never an empty file — when the
stored bundle is unusable. JSON and CSV are pinned in test_api_fits.py."""

import io
import time
import uuid
from typing import Any

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from conftest import linear_csv, signup, upload
from curvestudio_api.orm import Fit

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def store_fit(client: TestClient, headers: dict[str, str]) -> str:
    """Fit a small linear dataset through the real worker pool; returns the fit id."""
    dataset = upload(client, headers, linear_csv(12)).json()
    created = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=POLY1)
    assert created.status_code == 202, created.text
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(
            f"{API}/jobs/{created.json()['job_id']}", headers=headers
        ).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert job["status"] == "succeeded", job
    return str(job["result_ref"])


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
