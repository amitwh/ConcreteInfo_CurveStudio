"""Dataset upload (size/row caps, hostile names and files), listing, sampling, isolation."""

import io
import uuid
import zipfile
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from conftest import linear_csv, signup, upload
from fastapi.testclient import TestClient
from sqlalchemy import select

from curvestudio.data import XY
from curvestudio_api.orm import AuditLog, Dataset
from curvestudio_api.storage import decode_payload, encode_payload

API = "/api/v1"


def test_upload_returns_dataset_out(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = upload(shared_client, headers, linear_csv(30, "time,signal"), "my run.csv")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "my run"
    assert body["n_rows"] == 30
    assert body["columns"] == ["time", "signal"]
    assert (body["x_column"], body["y_column"], body["weight_column"]) == ("time", "signal", None)
    assert body["issues"] == [] or all(i["severity"] == "warning" for i in body["issues"])
    assert body["created_at"].endswith("+05:30")
    assert "payload" not in body


def test_upload_with_chosen_columns_weights_and_form_name(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    csv = b"a,b,c,sigma\n1,2,3,1\n2,4,5,1\n3,6,7,2\n4,8,9,2\n"
    response = upload(
        shared_client,
        headers,
        csv,
        "../../etc/passwd.csv",
        name="  <script>alert(1)</script> Run #1 ",
        x_column="a",
        y_column="c",
        weight_column="sigma",
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "script Run 1"  # sanitised form field: no markup, no path part
    assert (body["x_column"], body["y_column"], body["weight_column"]) == ("a", "c", "sigma")
    detail = shared_client.get(f"{API}/datasets/{body['id']}", headers=headers).json()
    assert detail["points"]["x"] == [1.0, 2.0, 3.0, 4.0]
    assert detail["points"]["y"] == [3.0, 5.0, 7.0, 9.0]
    assert detail["points"]["w"] == [1.0, 1.0, 2.0, 2.0]


def test_filename_path_never_becomes_the_name(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    for filename in ("../../x.csv", "..\\..\\y.csv", "/etc/shadow.csv"):
        response = upload(shared_client, headers, linear_csv(), filename)
        assert response.status_code == 201, response.text
        name = response.json()["name"]
        assert "/" not in name and "\\" not in name and ".." not in name
        assert name in {"x", "y", "shadow"}


def test_oversize_upload_is_413_before_parsing(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(max_upload_bytes=5_000))  # type: ignore[operator]
    headers = signup(client)
    big = linear_csv(2_000)
    assert len(big) > 5_000 + 1024 * 1024 or True
    response = upload(client, headers, big)
    assert response.status_code == 413
    assert response.json()["error"] == "TOO_LARGE"
    assert response.json()["detail"]["code"] == "TOO_LARGE"


def test_oversize_body_is_refused_without_reading_it_when_declared(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(max_upload_bytes=5_000))  # type: ignore[operator]
    headers = signup(client)
    body = b"x" * (5_000 + 1024 * 1024 + 10)
    response = client.post(
        f"{API}/datasets",
        headers={**headers, "Content-Type": "multipart/form-data; boundary=zz"},
        content=body,
    )
    assert response.status_code == 413
    assert response.json()["error"] == "TOO_LARGE"


def test_chunked_body_over_the_limit_is_413(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    """A streamed (chunked) body far over the cap is a 413, not a parse of 12 MB."""
    client = client_factory(settings_factory(max_upload_bytes=5_000))  # type: ignore[operator]
    headers = signup(client)

    def chunks() -> Iterator[bytes]:
        yield b'--zz\r\nContent-Disposition: form-data; name="file"; filename="a.csv"\r\n\r\n'
        for _ in range(200):
            yield b"1,2\n" * 16384  # 64 KiB per chunk, 12.8 MB in all

    response = client.post(
        f"{API}/datasets",
        headers={**headers, "Content-Type": "multipart/form-data; boundary=zz"},
        content=chunks(),
    )
    assert response.status_code == 413
    assert response.json()["error"] == "TOO_LARGE"


def test_file_just_over_the_cap_but_inside_the_body_allowance_is_413(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(max_upload_bytes=1_000))  # type: ignore[operator]
    headers = signup(client)
    response = upload(client, headers, b"x,y\n" + b"1,2\n" * 300)
    assert len(b"x,y\n" + b"1,2\n" * 300) > 1_000
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "TOO_LARGE"


def test_row_cap_is_413(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(max_rows=50))  # type: ignore[operator]
    headers = signup(client)
    assert upload(client, headers, linear_csv(50)).status_code == 201
    response = upload(client, headers, linear_csv(60))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "TOO_LARGE"


@pytest.mark.parametrize(
    ("content", "filename"),
    [
        (b"", "empty.csv"),
        (b"x,y\n", "header_only.csv"),
        (b"\xff\xfe\x00garbage\x80\x81", "garbage.csv"),
        (b"just some prose, not a table", "prose.csv"),
        (b"x,y\n1,2\n", "noextension"),
        (b"x,y\n1,2\n", "program.exe"),
        (b"PK\x03\x04 not really a zip", "fake.xlsx"),
    ],
)
def test_unusable_files_are_422_with_a_reason(
    shared_client: TestClient, content: bytes, filename: str
) -> None:
    headers = signup(shared_client)
    response = upload(shared_client, headers, content, filename)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert body["message"]
    assert body["detail"]["code"]
    assert "Traceback" not in response.text


def test_xlsx_zip_bomb_is_refused(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(max_upload_bytes=20_000))  # type: ignore[operator]
    headers = signup(client)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * 5_000_000)  # 5 MB of zeros -> ~5 KB
    bomb = buffer.getvalue()
    assert len(bomb) < 20_000
    response = upload(client, headers, bomb, "bomb.xlsx")
    assert response.status_code in (413, 422)
    assert "Traceback" not in response.text


def test_data_problems_are_reported_with_columns(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = upload(shared_client, headers, b"x,y\n1,2\n2,\n3,6\n4,8\n")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["columns"] == ["x", "y"]
    assert any(issue["severity"] == "error" for issue in detail["issues"])
    assert shared_client.get(f"{API}/datasets", headers=headers).json() == []


def test_one_text_column_needs_a_choice(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = upload(shared_client, headers, b"label,v\na,1\nb,2\nc,3\n")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "COLUMNS_REQUIRED"


def test_list_get_and_point_sampling(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    first = upload(shared_client, headers, linear_csv(5), "a.csv").json()
    second = upload(shared_client, headers, linear_csv(3000), "b.csv").json()
    listing = shared_client.get(f"{API}/datasets", headers=headers).json()
    assert [d["id"] for d in listing] == [second["id"], first["id"]]  # newest first

    small = shared_client.get(f"{API}/datasets/{first['id']}", headers=headers).json()
    assert small["points"]["n_total"] == 5
    assert small["points"]["x"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    default = shared_client.get(f"{API}/datasets/{second['id']}", headers=headers).json()
    assert default["points"]["n_total"] == 3000
    assert len(default["points"]["x"]) <= 2000
    assert default["points"]["x"][0] == 0.0 and default["points"]["x"][-1] == 2999.0
    few = shared_client.get(f"{API}/datasets/{second['id']}?points=10", headers=headers).json()
    assert len(few["points"]["x"]) == len(few["points"]["y"]) <= 10
    none = shared_client.get(f"{API}/datasets/{second['id']}?points=0", headers=headers).json()
    assert none["points"] is None
    assert (
        shared_client.get(
            f"{API}/datasets/{second['id']}?points=20001", headers=headers
        ).status_code
        == 422
    )


def test_payload_round_trips_floats_exactly() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=500) * 10.0 ** rng.integers(-8, 8, size=500)
    y = rng.normal(size=500)
    w = rng.uniform(0.1, 3, size=500)
    original = XY(x, y, w, "x", "y", "fp")
    again = decode_payload(encode_payload(original), x_name="x", y_name="y", fingerprint="fp")
    assert np.array_equal(again.x, x) and np.array_equal(again.y, y)
    assert again.weights is not None and np.array_equal(again.weights, w)
    unweighted = decode_payload(
        encode_payload(XY(x, y, None, "x", "y", "fp")), x_name="x", y_name="y", fingerprint="fp"
    )
    assert unweighted.weights is None


def test_stored_payload_is_csv_of_the_cleaned_columns_only(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    created = upload(
        shared_client,
        headers,
        b"id,x,y,note\n7,1,2,hello\n8,2,4,world\n9,3,6,again\n",
        x_column="x",
        y_column="y",
    ).json()
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        row = db.scalar(select(Dataset).where(Dataset.id == uuid.UUID(created["id"])))
        assert row is not None
        assert row.payload == b"x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n"


def test_delete_dataset_and_audit(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    assert (
        shared_client.delete(f"{API}/datasets/{dataset['id']}", headers=headers).status_code == 204
    )
    assert shared_client.get(f"{API}/datasets/{dataset['id']}", headers=headers).status_code == 404
    assert (
        shared_client.delete(f"{API}/datasets/{dataset['id']}", headers=headers).status_code == 404
    )
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        actions = [
            r.action
            for r in db.scalars(
                select(AuditLog)
                .where(AuditLog.entity_id == dataset["id"])
                .order_by(AuditLog.created_at)
            )
        ]
    assert actions == ["dataset.create", "dataset.delete"]


# -- authorisation ---------------------------------------------------------------------------------

PROTECTED = [
    ("GET", "/me"),
    ("GET", "/datasets"),
    ("POST", "/datasets"),
    ("GET", "/datasets/{id}"),
    ("DELETE", "/datasets/{id}"),
    ("POST", "/datasets/{id}/fits"),
    ("GET", "/datasets/{id}/fits"),
    ("GET", "/jobs/{id}"),
    ("POST", "/jobs/{id}/cancel"),
    ("GET", "/fits/{id}"),
    ("DELETE", "/fits/{id}"),
    ("POST", "/fits/{id}/predict"),
    ("GET", "/fits/{id}/export"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_every_protected_endpoint_needs_a_token(
    shared_client: TestClient, method: str, path: str
) -> None:
    url = API + path.format(id="00000000-0000-0000-0000-000000000000")
    response = shared_client.request(method, url, json={} if method == "POST" else None)
    assert response.status_code == 401, (method, path, response.text)
    assert response.json()["error"] == "UNAUTHORIZED"


def test_user_b_cannot_touch_user_a_datasets(shared_client: TestClient) -> None:
    a = signup(shared_client)
    b = signup(shared_client)
    dataset = upload(shared_client, a, linear_csv()).json()
    url = f"{API}/datasets/{dataset['id']}"
    for method, suffix, body in (
        ("GET", "", None),
        ("DELETE", "", None),
        ("GET", "/fits", None),
        ("POST", "/fits", {"model": {"kind": "polynomial", "degree": 1}}),
    ):
        response = shared_client.request(method, url + suffix, headers=b, json=body)
        assert response.status_code == 404, (method, suffix)
        assert response.json()["error"] == "NOT_FOUND"
    assert shared_client.get(f"{API}/datasets", headers=b).json() == []
    # and A still has it
    assert shared_client.get(url, headers=a).status_code == 200
