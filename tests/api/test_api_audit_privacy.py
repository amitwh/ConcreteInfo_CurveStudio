"""The audit trail keeps no dataset name. A dataset's name is usually a file name, which can
identify a person, and the trail outlives the account that owned the dataset: erasing an
account only clears ``user_id`` on its rows, so anything written into ``detail`` stays."""

import json
from collections.abc import Callable

from api_test_helpers import linear_csv, new_email, signup, upload
from fastapi.testclient import TestClient
from sqlalchemy import select

from curvestudio_api.orm import AuditLog
from curvestudio_api.settings import Settings

API = "/api/v1"
ME = f"{API}/me"
PATIENT_NAME = "patient-42-blood-pressure"


def audit_details(client: TestClient) -> list[str]:
    """Every audit row's ``detail`` as JSON text, oldest first."""
    with client.app.state.ctx.session_factory() as db:
        rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at))
        return [json.dumps(row.detail) for row in rows]


def delete_dataset(client: TestClient, headers: dict[str, str], dataset_id: str) -> None:
    response = client.delete(f"{API}/datasets/{dataset_id}", headers=headers)
    assert response.status_code == 204, response.text


def erase_account(client: TestClient, headers: dict[str, str], email: str) -> None:
    """``DELETE /me``: the body must repeat the part of the address before the ``@``."""
    phrase = email.partition("@")[0]
    response = client.request("DELETE", ME, headers=headers, json={"confirm": phrase})
    assert response.status_code == 204, response.text


def test_deleting_a_dataset_leaves_no_name_in_the_audit_trail(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    created = upload(shared_client, headers, linear_csv(), f"{PATIENT_NAME}.csv").json()
    assert created["name"] == PATIENT_NAME  # the file name is what the dataset would be called
    delete_dataset(shared_client, headers, created["id"])

    with shared_client.app.state.ctx.session_factory() as db:
        rows = list(
            db.scalars(
                select(AuditLog)
                .where(AuditLog.entity_id == created["id"])
                .order_by(AuditLog.created_at)
            )
        )
    details = [json.dumps(row.detail) for row in rows]

    # the deletion is still recorded, against the dataset's id, but without its name
    deletions = [row for row in rows if row.action == "dataset.delete"]
    assert len(deletions) == 1
    assert "name" not in deletions[0].detail

    assert PATIENT_NAME not in "".join(details)


def test_erasing_an_account_leaves_no_dataset_name_behind(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory()  # a database of this test's own, so "every row" means every row
    email = new_email()
    headers = signup(client, email)
    first = upload(client, headers, linear_csv(), "asha-verma-glucose.csv").json()
    second = upload(client, headers, linear_csv(), "ravi-patient-7-hba1c.csv").json()
    names = (first["name"], second["name"])
    delete_dataset(client, headers, first["id"])  # the name used to be written here
    erase_account(client, headers, email)

    details = audit_details(client)
    assert details  # the trail outlived the account, as it is meant to
    for name in names:
        assert name not in "".join(details), name
