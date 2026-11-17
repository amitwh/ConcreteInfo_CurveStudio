"""Security hardening: odd input, key and password policy, headers, migration 0002, erasure."""

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from conftest import PASSWORD, linear_csv, make_settings, new_email, signup, upload
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from curvestudio_api.app import create_app
from curvestudio_api.migrations import _config
from curvestudio_api.orm import AuditLog, Dataset, Fit, Job, RefreshToken, User
from curvestudio_api.settings import Settings, load_jwt_keys

API = "/api/v1"


# -- input -----------------------------------------------------------------------------------------


@pytest.fixture
def spa_client(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    tmp_path: Path,
) -> TestClient:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    return client_factory(settings_factory(web_dist=dist))


@pytest.mark.parametrize("path", ["/%00", "/a%00b", "/assets/%00.js"])
def test_nul_bytes_in_paths_are_404_not_500(spa_client: TestClient, path: str) -> None:
    response = spa_client.get(path)
    assert response.status_code == 404, path
    assert response.json()["error"] == "NOT_FOUND"


@pytest.mark.parametrize("route", ["register", "login"])
def test_a_password_with_a_lone_surrogate_is_422(shared_client: TestClient, route: str) -> None:
    body = '{"email": "sur@example.org", "password": "\\ud800 plus enough characters"}'
    response = shared_client.post(
        f"{API}/auth/{route}", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "VALIDATION_ERROR"
    assert "enough characters" not in response.text


def test_a_name_with_a_lone_surrogate_is_422(shared_client: TestClient) -> None:
    body = '{"email": "sur2@example.org", "password": "' + PASSWORD + '", "name": "\\udfff"}'
    response = shared_client.post(
        f"{API}/auth/register", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_validation_errors_are_capped(shared_client: TestClient) -> None:
    body = {"email": "a@b.co", "password": PASSWORD, **{f"unknown_{i}": 1 for i in range(60)}}
    response = shared_client.post(f"{API}/auth/register", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail["errors"]) == 20
    assert detail["more"] == 40
    long_key = shared_client.post(
        f"{API}/auth/register", json={"email": "a@b.co", "password": PASSWORD, "k" * 5000: 1}
    )
    assert len(long_key.text) < 2000


@pytest.mark.parametrize(
    "email", ["a@b.c", "a<b>@example.org", "x@y<z>.org", "a b@example.org", "@x.org"]
)
def test_registration_checks_the_email_shape(shared_client: TestClient, email: str) -> None:
    response = shared_client.post(
        f"{API}/auth/register", json={"email": email, "password": PASSWORD}
    )
    assert response.status_code == 422, email


# -- password policy -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("longlocalpart@example.org", "longlocalpart"),
        ("LongLocalPart@example.org", "LONGLOCALPART"),
        ("someone@example.org", "password123"),
        ("someone@example.org", "1234567890"),
        ("someone@example.org", "Qwertyuiop"),
        ("someone@example.org", "iloveyou2024"),
    ],
)
def test_weak_passwords_are_refused(shared_client: TestClient, email: str, password: str) -> None:
    response = shared_client.post(
        f"{API}/auth/register", json={"email": email, "password": password}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "VALIDATION_ERROR"
    assert password not in response.text


def test_length_limits_still_apply_and_a_good_password_passes(shared_client: TestClient) -> None:
    assert (
        shared_client.post(
            f"{API}/auth/register", json={"email": new_email(), "password": "short"}
        ).status_code
        == 422
    )
    assert (
        shared_client.post(
            f"{API}/auth/register", json={"email": new_email(), "password": "x" * 129}
        ).status_code
        == 422
    )
    assert (
        shared_client.post(
            f"{API}/auth/register",
            json={"email": new_email(), "password": "tr1cky-but-long-enough"},
        ).status_code
        == 201
    )


def test_the_common_password_list_is_bundled() -> None:
    from curvestudio_api.passwords import common_passwords

    words = common_passwords()
    assert len(words) >= 1000
    assert all(10 <= len(w) <= 128 and w == w.lower() for w in words)


# -- headers ---------------------------------------------------------------------------------------


def test_token_responses_are_not_cacheable(shared_client: TestClient) -> None:
    email = new_email()
    shared_client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD})
    login = shared_client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD})
    assert login.headers["cache-control"] == "no-store"
    refreshed = shared_client.post(
        f"{API}/auth/refresh", json={"refresh_token": login.json()["refresh_token"]}
    )
    assert refreshed.status_code == 200
    assert refreshed.headers["cache-control"] == "no-store"


# -- keys ------------------------------------------------------------------------------------------


def pem_pair(bits: int) -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


def test_empty_key_variables_mean_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CURVESTUDIO_JWT_PRIVATE_KEY_PEM", "")
    monkeypatch.setenv("CURVESTUDIO_JWT_PUBLIC_KEY_PEM", "  ")
    monkeypatch.setenv("CURVESTUDIO_DATA_DIR", str(tmp_path))
    settings = Settings()
    assert settings.jwt_private_key_pem is None and settings.jwt_public_key_pem is None
    assert load_jwt_keys(settings)[0].startswith("-----BEGIN PRIVATE KEY")  # generated (dev)
    monkeypatch.setenv("CURVESTUDIO_ENV", "prod")
    with pytest.raises(ValidationError, match="required"):
        Settings()  # in production empty is as good as missing


def test_rsa_keys_below_2048_bits_are_refused(tmp_path: Path) -> None:
    private, public = pem_pair(1024)
    settings = Settings(
        env="prod",
        data_dir=tmp_path,
        jwt_private_key_pem=private,
        jwt_public_key_pem=public,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="2048"):
        create_app(settings)
    good_private, good_public = pem_pair(2048)
    create_app(
        Settings(
            env="prod",
            data_dir=tmp_path,
            jwt_private_key_pem=good_private,
            jwt_public_key_pem=good_public,  # type: ignore[arg-type]
        )
    )


# -- migration 0002 --------------------------------------------------------------------------------


def test_emails_are_unique_ignoring_case_in_the_database(shared_client: TestClient) -> None:
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(User(id=uuid.uuid4(), email="Case@Example.org", password_hash="x"))
        db.commit()
        db.add(User(id=uuid.uuid4(), email="case@example.org", password_hash="x"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_upgrading_from_0001_keeps_data_and_applies_the_new_rules(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    engine = app.state.ctx.engine
    command.upgrade(_config(engine), "0001")
    with engine.begin() as connection:
        limits = connection.execute(text("SELECT limits FROM plans")).scalar_one()
        assert "unlimited" in str(limits)
        connection.execute(
            text(
                "INSERT INTO users (id, email, password_hash, name, is_active, created_at, "
                "updated_at) VALUES ('00000000000000000000000000000001', 'old@example.org', "
                "'x', '', 1, '2026-01-01', '2026-01-01')"
            )
        )
    command.upgrade(_config(engine), "head")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM users")).scalar_one() == 1
        limits = connection.execute(text("SELECT limits FROM plans")).scalar_one()
        assert "max_datasets" in str(limits) and "unlimited" not in str(limits)
    command.downgrade(_config(engine), "0001")
    command.upgrade(_config(engine), "head")


# -- audit trail and erasure -----------------------------------------------------------------------


def test_only_erasing_the_user_id_is_allowed_on_audit_rows(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    me = shared_client.get(f"{API}/me", headers=headers).json()
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        for statement in (
            "UPDATE audit_logs SET action = 'tampered'",
            "UPDATE audit_logs SET detail = '{}'",
            "UPDATE audit_logs SET created_at = '2000-01-01'",
            "UPDATE audit_logs SET user_id = NULL, action = 'tampered'",
            "DELETE FROM audit_logs",
        ):
            with pytest.raises(SQLAlchemyError, match="append-only"):
                db.execute(text(statement))
            db.rollback()
        rows = db.scalars(select(AuditLog).where(AuditLog.user_id == uuid.UUID(me["id"]))).all()
        assert rows
        db.execute(
            text("UPDATE audit_logs SET user_id = NULL WHERE user_id = :u"),
            {"u": uuid.UUID(me["id"]).hex},
        )
        db.commit()
        assert (
            db.scalars(select(AuditLog).where(AuditLog.user_id == uuid.UUID(me["id"]))).all() == []
        )
        with pytest.raises(
            SQLAlchemyError, match="append-only"
        ):  # a second erasure is a no-op change
            db.execute(text("UPDATE audit_logs SET user_id = '00000000000000000000000000000001'"))
        db.rollback()


def test_delete_me_erases_the_account_and_anonymises_the_audit_trail(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    victim, bystander = signup(client), signup(client)
    victim_id = uuid.UUID(client.get(f"{API}/me", headers=victim).json()["id"])
    bystander_id = uuid.UUID(client.get(f"{API}/me", headers=bystander).json()["id"])
    mine = upload(client, victim, linear_csv()).json()
    theirs = upload(client, bystander, linear_csv()).json()
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(
            Fit(
                id=uuid.uuid4(),
                user_id=victim_id,
                dataset_id=uuid.UUID(mine["id"]),
                spec={},
                bundle={},
                status="converged",
            )
        )
        db.add(
            Job(
                id=uuid.uuid4(),
                user_id=victim_id,
                dataset_id=uuid.UUID(mine["id"]),
                kind="fit",
                status="succeeded",
                spec={},
            )
        )
        db.commit()
        before = len(db.scalars(select(AuditLog).where(AuditLog.user_id == victim_id)).all())
    assert before >= 3  # register, login, dataset.create

    assert client.delete(f"{API}/me", headers=victim).status_code == 204

    # the account and everything in it are gone
    assert client.get(f"{API}/me", headers=victim).status_code == 401
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.get(User, victim_id) is None
        for model in (Dataset, Fit, Job, RefreshToken):
            assert db.scalars(select(model).where(model.user_id == victim_id)).all() == []  # type: ignore[attr-defined]
        # audit rows survive without the user id
        assert db.scalars(select(AuditLog).where(AuditLog.user_id == victim_id)).all() == []
        orphaned = db.scalars(
            select(AuditLog).where(AuditLog.user_id.is_(None), AuditLog.action == "user.register")
        ).all()
        assert any(a.entity_id == str(victim_id) for a in orphaned)
        assert (
            db.scalars(select(AuditLog).where(AuditLog.action == "user.delete")).one().user_id
            is None
        )
        # the bystander is untouched
        assert db.get(User, bystander_id) is not None
        assert db.get(Dataset, uuid.UUID(theirs["id"])) is not None
        assert db.scalars(select(AuditLog).where(AuditLog.user_id == bystander_id)).all()
    # the deleted user cannot log back in, and the address is free again
    login = client.post(f"{API}/auth/login", json={"email": "gone", "password": PASSWORD})
    assert login.status_code in (401, 422)
    assert client.delete(f"{API}/me").status_code == 401


def test_delete_me_stops_running_jobs(
    client_factory: Callable[..., TestClient],
) -> None:
    import time

    import api_job_workers

    client = client_factory(job_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset: dict[str, Any] = upload(client, headers, linear_csv()).json()
    job = client.post(
        f"{API}/datasets/{dataset['id']}/fits",
        headers=headers,
        json={"model": {"kind": "polynomial", "degree": 1}},
    ).json()["job_id"]
    deadline = time.monotonic() + 30
    while client.get(f"{API}/jobs/{job}", headers=headers).json()["status"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert client.delete(f"{API}/me", headers=headers).status_code == 204
    import multiprocessing

    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert multiprocessing.active_children() == []
