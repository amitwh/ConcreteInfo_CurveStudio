"""The other columns of an upload: stored beside the fit input, counted against the quota,
listed by the detail route and never read back by it; a dataset from before keeps its slim
form, is never migrated and says so."""

import csv
import io
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import api_job_workers
import numpy as np
import pytest
from alembic import command
from api_test_helpers import linear_csv, make_settings, signup, upload
from fastapi.testclient import TestClient
from openpyxl import Workbook
from sqlalchemy import Engine, event, insert, inspect, select
from sqlalchemy.orm import Session, undefer

from curvestudio.data import XY
from curvestudio_api.app import create_app
from curvestudio_api.errors import ApiError
from curvestudio_api.importer import ImportRunner
from curvestudio_api.migrations import _config, current_revision, upgrade_to_head
from curvestudio_api.orm import Dataset, User
from curvestudio_api.settings import Settings
from curvestudio_api.storage import decode_payload, decode_table, encode_payload

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}
GROUPED = b"x,y,g,a\n1,2,alpha,1.5\n2,4,beta,\n3,6,alpha,2.5\n4,8,beta,3\n"
FIT_PAYLOAD = b"x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n4.0,8.0\n"
TABLE = b"g,a\nalpha,1.5\nbeta,\nalpha,2.5\nbeta,3.0\n"
X_AND_Y = [
    {"name": "x", "kind": "number", "n_missing": 0},
    {"name": "y", "kind": "number", "n_missing": 0},
]


def stored(client: TestClient, dataset_id: str) -> Dataset:
    """The dataset's row with its deferred blobs loaded (the session that loaded it is closed
    by then; what it loaded stays readable)."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        row = db.scalar(
            select(Dataset)
            .options(undefer(Dataset.payload), undefer(Dataset.table))
            .where(Dataset.id == uuid.UUID(dataset_id))
        )
        assert row is not None
        return row


def detail(client: TestClient, headers: dict[str, str], dataset_id: str) -> dict[str, Any]:
    response = client.get(f"{API}/datasets/{dataset_id}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def finished(client: TestClient, headers: dict[str, str], job_id: str) -> dict[str, Any]:
    """A fit job once it has succeeded, failed or been cancelled (90 seconds at most)."""
    deadline = time.monotonic() + 90
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = client.get(f"{API}/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stayed {job.get('status')!r}")


def add_from_before(
    client: TestClient, headers: dict[str, str], content: bytes = linear_csv(20), **columns: Any
) -> bytes:
    """Store ``content`` (a 20-row line by default) for the account of ``headers`` as the code
    of before wrote it: the fit payload alone. Returns that payload."""
    values = np.loadtxt(io.BytesIO(content), delimiter=",", skiprows=1, ndmin=2)
    xy = XY(values[:, 0], values[:, 1], None, "x", "y", "before")
    payload = encode_payload(xy)
    account = uuid.UUID(client.get(f"{API}/me", headers=headers).json()["id"])
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(
            Dataset(
                id=uuid.uuid4(),
                user_id=account,
                name="before",
                n_rows=int(xy.x.size),
                columns=["x", "y"],
                x_column="x",
                y_column="y",
                fingerprint=xy.fingerprint,
                payload=payload,
                **columns,
            )
        )
        db.commit()
    return payload


# -- what an upload stores ------------------------------------------------------------------------


def test_an_upload_keeps_the_other_columns(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory()
    headers = signup(client)
    created = upload(client, headers, GROUPED, x_column="x", y_column="y")
    assert created.status_code == 201, created.text
    assert "table_stored" not in created.json()  # the detail alone says it

    row = stored(client, created.json()["id"])
    assert row.payload == FIT_PAYLOAD
    assert row.table == TABLE
    columns = decode_table(row.table)
    assert list(columns) == ["g", "a"]
    assert columns["g"].dtype.kind == "O"  # an object array of Python strings
    assert columns["g"].tolist() == ["alpha", "beta", "alpha", "beta"]
    assert columns["a"].dtype == np.float64
    assert np.isnan(columns["a"][1])
    assert np.delete(columns["a"], 1).tolist() == [1.5, 2.5, 3.0]

    described = detail(client, headers, created.json()["id"])
    assert described["table_stored"] is True
    assert described["table_columns"] == [
        *X_AND_Y,
        {"name": "g", "kind": "text", "n_missing": 0},
        {"name": "a", "kind": "number", "n_missing": 1},
    ]


def test_a_spreadsheet_keeps_its_other_columns_too(
    client_factory: Callable[..., TestClient],
) -> None:
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet()
    sheet.append(["x", "y", "batch", "a"])
    for x, batch, a in ((1.0, "one", 1.5), (2.0, "two", None), (3.0, "one", 2.5), (4.0, "two", 3)):
        sheet.append([x, 2.0 * x, batch, a])
    buffer = io.BytesIO()
    workbook.save(buffer)
    client = client_factory()
    headers = signup(client)
    created = upload(client, headers, buffer.getvalue(), "run.xlsx", x_column="x", y_column="y")
    assert created.status_code == 201, created.text
    row = stored(client, created.json()["id"])
    assert row.table == b"batch,a\none,1.5\ntwo,\none,2.5\ntwo,3.0\n"
    assert detail(client, headers, created.json()["id"])["table_columns"] == [
        *X_AND_Y,
        {"name": "batch", "kind": "text", "n_missing": 0},
        {"name": "a", "kind": "number", "n_missing": 1},
    ]


def test_an_upload_with_only_the_fit_columns_stores_no_table(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    headers = signup(client)
    plain = upload(client, headers, linear_csv(5)).json()
    row = stored(client, plain["id"])
    assert row.table is None
    described = detail(client, headers, plain["id"])
    assert described["table_stored"] is True
    assert described["table_columns"] == X_AND_Y

    weighted = upload(
        client,
        headers,
        b"x,y,w\n1,2,1\n2,4,2\n3,6,1\n",
        x_column="x",
        y_column="y",
        weight_column="w",
    ).json()
    row = stored(client, weighted["id"])
    assert row.table is None
    three = [*X_AND_Y, {"name": "w", "kind": "number", "n_missing": 0}]
    assert row.table_columns == three
    described = detail(client, headers, weighted["id"])
    assert described["table_stored"] is True
    assert described["table_columns"] == three


def test_the_rows_line_up(client_factory: Callable[..., TestClient]) -> None:
    """Blank rows are dropped when the file is read: the stored table has a row for each row
    of the fit's input, in the same order."""
    client = client_factory()
    headers = signup(client)
    blank_rows = b"x,y,g\n1,10,a\n\n2,20,b\n   ,   ,   \n\n3,30,c\n\n"
    created = upload(client, headers, blank_rows, x_column="x", y_column="y")
    assert created.status_code == 201, created.text
    row = stored(client, created.json()["id"])
    assert row.n_rows == 3
    assert row.payload == b"x,y\n1.0,10.0\n2.0,20.0\n3.0,30.0\n"
    assert row.table == b"g\na\nb\nc\n"
    again = decode_payload(row.payload, x_name="x", y_name="y", fingerprint=row.fingerprint)
    labels = decode_table(row.table)["g"]
    assert again.x.size == labels.size == row.n_rows
    assert again.x.tolist() == [1.0, 2.0, 3.0] and labels.tolist() == ["a", "b", "c"]
    assert detail(client, headers, created.json()["id"])["table_columns"] == [
        *X_AND_Y,
        {"name": "g", "kind": "text", "n_missing": 0},
    ]


def test_the_list_and_detail_routes_never_read_the_stored_table(
    client_factory: Callable[..., TestClient],
) -> None:
    """Both routes take the column list from the row and leave the blobs where they are: a
    stored table may be tens of megabytes."""
    client = client_factory()
    headers = signup(client)
    created = upload(client, headers, GROUPED, x_column="x", y_column="y").json()
    statements: list[str] = []
    engine: Engine = client.app.state.ctx.engine  # type: ignore[attr-defined]

    def record(_conn: Any, _cursor: Any, statement: str, *_: Any) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        assert client.get(f"{API}/datasets", headers=headers).status_code == 200
        listed = statements[:]
        statements.clear()
        assert client.get(f"{API}/datasets/{created['id']}", headers=headers).status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", record)

    def reads(made: list[str]) -> list[str]:
        return [s for s in made if s.lstrip().upper().startswith("SELECT") and "datasets" in s]

    assert len(reads(listed)) == 1
    assert "datasets.table_columns" in reads(listed)[0]  # the list is read, not the table
    assert "datasets.table_columns" in reads(statements)[0]
    assert not any(re.search(r"datasets\.table\b", s) for s in reads(listed + statements))
    assert not any(re.search(r"datasets\.payload\b", s) for s in reads(listed))  # as before


# -- a dataset from before -------------------------------------------------------------------------


def test_a_dataset_from_before_still_works_and_says_so(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    payload = add_from_before(shared_client, headers, table=None, table_columns=None)
    add_from_before(shared_client, headers)  # written without naming either column
    listed = shared_client.get(f"{API}/datasets", headers=headers).json()
    assert len(listed) == 2
    for dataset in listed:
        assert "table_stored" not in dataset
        described = detail(shared_client, headers, dataset["id"])
        assert described["table_stored"] is False
        assert described["table_columns"] == X_AND_Y
        assert described["points"]["n_total"] == 20

    # a fit, a prediction and an export work on it as on any dataset
    created = shared_client.post(
        f"{API}/datasets/{listed[0]['id']}/fits", headers=headers, json=POLY1
    )
    assert created.status_code == 202, created.text
    job = finished(shared_client, headers, created.json()["job_id"])
    assert job["status"] == "succeeded", job
    fit_id = job["result_ref"]
    predicted = shared_client.post(
        f"{API}/fits/{fit_id}/predict", headers=headers, json={"x": [1.0, 2.0]}
    )
    assert predicted.status_code == 200, predicted.text
    assert predicted.json()["x"] == [1.0, 2.0] and len(predicted.json()["y"]) == 2
    exported = shared_client.get(f"{API}/fits/{fit_id}/export?format=csv", headers=headers)
    assert exported.status_code == 200, exported.text
    rows = list(csv.reader(io.StringIO(exported.text)))
    start = rows.index(["x", "y", "weight", "fitted", "residual", "note"])
    assert [r[0] for r in rows[start + 1 :]] == [str(float(i)) for i in range(20)]

    # and its bytes still count
    usage = shared_client.get(f"{API}/me", headers=headers).json()["usage"]
    assert usage["datasets"] == 2 and usage["fits"] == 1
    assert usage["dataset_bytes"] == 2 * len(payload)


# -- the quota and the cap -------------------------------------------------------------------------


def test_the_quota_counts_both_payloads(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    old = encode_payload(XY(np.array([1.0]), np.array([2.0]), None, "x", "y", "before"))
    fit, table = b"x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n", b"g\na\nb\na\n"
    grouped = b"x,y,g\n1,2,a\n2,4,b\n3,6,a\n"
    room = len(old) + len(fit) + len(table)  # exactly what the two datasets take
    client = client_factory(settings_factory(quota_max_dataset_bytes=room))
    headers = signup(client)
    add_from_before(client, headers, b"x,y\n1,2\n", table=None, table_columns=None)
    first = upload(client, headers, grouped, x_column="x", y_column="y")
    assert first.status_code == 201, first.text
    row = stored(client, first.json()["id"])
    assert (row.payload, row.table) == (fit, table)
    usage = client.get(f"{API}/me", headers=headers).json()["usage"]
    assert usage["datasets"] == 2 and usage["dataset_bytes"] == room

    refused = upload(client, headers, grouped, x_column="x", y_column="y")
    assert refused.status_code == 413
    body = refused.json()
    assert body["error"] == "QUOTA_EXCEEDED"
    assert body["detail"]["quota"] == "dataset_bytes"
    assert body["detail"]["usage"]["dataset_bytes"] == room
    assert len(client.get(f"{API}/datasets", headers=headers).json()) == 2  # nothing was stored
    usage = client.get(f"{API}/me", headers=headers).json()["usage"]
    assert usage["datasets"] == 2 and usage["dataset_bytes"] == room


def test_the_cap_covers_both_payloads(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    capped = ImportRunner(
        concurrency=1,
        timeout_s=60.0,
        memory_mb=1024,
        max_payload_bytes=1_000,
        worker=api_job_workers.bulky_import_worker,
    )
    who = uuid.uuid4()
    # the 8-byte length in front of the two payloads makes the cap exactly
    within = capped.run(who, {"fit_bytes": 592, "table_bytes": 400})
    assert (within.payload, within.table) == (b"x" * 592, b"g" * 400)
    for fit_bytes, table_bytes in ((600, 400), (1_001, 0)):  # together over; one of them over
        with pytest.raises(ApiError) as refused:
            capped.run(who, {"fit_bytes": fit_bytes, "table_bytes": table_bytes})
        assert refused.value.status_code == 422
        assert refused.value.extra == {"code": "IMPORT_FAILED"}
    # a last message of another shape is refused the same way: nothing of it is stored
    old_shape = ImportRunner(
        concurrency=1,
        timeout_s=60.0,
        memory_mb=1024,
        max_payload_bytes=1_000,
        worker=api_job_workers.prefixless_import_worker,
    )
    with pytest.raises(ApiError) as refused:
        old_shape.run(who, {})
    assert refused.value.status_code == 422
    assert refused.value.extra == {"code": "IMPORT_FAILED"}

    # Through the application's own settings. The cap is at least 192 MiB, so a table whose
    # stored form is larger than max_rows x 100 bytes imports.
    client = client_factory(settings_factory(max_rows=500))
    headers = signup(client)
    note = "note " * 30
    rows = "\n".join(f"{i},{2 * i + 1},{note}" for i in range(500))
    created = upload(client, headers, f"x,y,note\n{rows}\n".encode(), x_column="x", y_column="y")
    assert created.status_code == 201, created.text
    row = stored(client, created.json()["id"])
    assert row.table is not None and len(row.table) > 500 * 100
    assert decode_table(row.table)["note"].tolist() == [note.strip()] * 500


def test_the_import_cap_is_192_mib_or_a_hundred_bytes_a_row_whichever_is_larger(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    """What the application builds its import runner with: the cap on one import's result,
    which carries the fit's and the table's payload together."""
    for max_rows in (500, 2_000_000, 3_000_000):  # the floor, the floor by a hair, the rows
        settings = settings_factory(max_rows=max_rows)
        client = client_factory(settings)
        runner = client.app.state.ctx.imports  # type: ignore[attr-defined]
        assert runner._max_payload_bytes == max(192 * 1024 * 1024, settings.max_rows * 100)


# -- the migrations --------------------------------------------------------------------------------


def dataset_columns(engine: Engine) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns("datasets")}


def add_dataset_of_an_older_schema(db: Session, user: User, **stored: Any) -> uuid.UUID:
    """A dataset row for a database that is at an older revision.

    The statement names the columns it is given and none other: an object added to the
    session would name every column of the model, the new ones included."""
    dataset_id = uuid.uuid4()
    db.execute(
        insert(Dataset)
        .values(
            id=dataset_id,
            user_id=user.id,
            name="line",
            n_rows=2,
            columns=["x", "y"],
            x_column="x",
            y_column="y",
            fingerprint="f" * 64,
            payload=b"x,y\n1,2\n",
        )
        .values(**stored)
    )
    db.commit()
    return dataset_id


def test_migration_0013_round_trip(tmp_path: Path) -> None:
    """``datasets.table`` arrives nullable: a dataset stored before the column has ``None``
    for it, which says it keeps its slim form. The downgrade takes the column away and
    nothing else."""
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    assert current_revision(engine) == "0016"
    assert {"table", "table_columns"} <= dataset_columns(engine)

    command.downgrade(_config(engine), "0012")
    assert current_revision(engine) == "0012"
    assert not {"table", "table_columns"} & dataset_columns(engine)
    with Session(engine) as db:  # a dataset of a database that has not got the columns yet
        user = User(email="before@example.org", password_hash="not-a-hash")
        db.add(user)
        db.flush()
        dataset_id = add_dataset_of_an_older_schema(db, user)

    upgrade_to_head(engine)
    assert current_revision(engine) == "0016"
    column = next(c for c in inspect(engine).get_columns("datasets") if c["name"] == "table")
    assert column["nullable"] is True
    with Session(engine) as db:  # the blobs are deferred columns: read them while it is open
        kept = db.get(Dataset, dataset_id)
        assert kept is not None and kept.payload == b"x,y\n1,2\n"
        assert kept.table is None and kept.table_columns is None


def test_migration_0014_round_trip(tmp_path: Path) -> None:
    """``datasets.table_columns`` arrives nullable: a dataset of a database at 0013, with a
    stored table but no list of its columns, keeps ``None`` for it, which says so. The
    downgrade takes the column away and nothing else."""
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    assert current_revision(engine) == "0016"
    assert "table_columns" in dataset_columns(engine)

    command.downgrade(_config(engine), "0013")
    assert current_revision(engine) == "0013"
    assert "table" in dataset_columns(engine) and "table_columns" not in dataset_columns(engine)
    with Session(engine) as db:  # a dataset of a database that has the table, not the list
        user = User(email="before@example.org", password_hash="not-a-hash")
        db.add(user)
        db.flush()
        dataset_id = add_dataset_of_an_older_schema(db, user, table=b"g\na\nb\n")

    upgrade_to_head(engine)
    assert current_revision(engine) == "0016"
    column = next(
        c for c in inspect(engine).get_columns("datasets") if c["name"] == "table_columns"
    )
    assert column["nullable"] is True
    with Session(engine) as db:
        kept = db.get(Dataset, dataset_id)
        assert kept is not None and kept.table == b"g\na\nb\n"
        assert kept.table_columns is None
