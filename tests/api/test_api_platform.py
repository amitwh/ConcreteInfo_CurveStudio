"""Health, settings, envelope, headers, request ids, logging, migrations, OpenAPI, SPA."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import api_job_workers
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from conftest import linear_csv, make_settings, new_email, signup, upload
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import Message

from curvestudio_api.app import create_app
from curvestudio_api.db import make_session_factory
from curvestudio_api.errors import ApiError
from curvestudio_api.export_openapi import DEFAULT_PATH, generate, render
from curvestudio_api.logging_setup import JsonFormatter
from curvestudio_api.middleware import BodyLimitMiddleware
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.orm import AuditLog, Base, Plan
from curvestudio_api.settings import Settings, load_jwt_keys
from curvestudio_api.timeutil import IST, to_ist_iso

API = "/api/v1"
REPO = Path(__file__).resolve().parents[2]


# -- health ----------------------------------------------------------------------------------------


def test_health_and_ready(shared_client: TestClient) -> None:
    health = shared_client.get(f"{API}/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok" and health.json()["database"] == "ok"
    ready = shared_client.get(f"{API}/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_health_is_503_when_the_database_is_down(
    client_factory: Callable[..., TestClient], tmp_path: Path
) -> None:
    client = client_factory()
    assert client.get(f"{API}/health").status_code == 200
    broken = create_engine(f"sqlite:///{tmp_path}/missing-dir/nope.db")
    client.app.state.ctx.session_factory = make_session_factory(broken)  # type: ignore[attr-defined]
    response = client.get(f"{API}/health")
    assert response.status_code == 503
    assert response.json()["error"] == "SERVICE_UNAVAILABLE"
    assert client.get(f"{API}/ready").status_code == 503


def test_the_event_loop_is_not_blocked_by_a_running_job(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(job_worker=api_job_workers.sleeping_worker)
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    spec = {"model": {"kind": "polynomial", "degree": 1}}
    job_id = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=spec).json()[
        "job_id"
    ]
    deadline = time.monotonic() + 30
    while client.get(f"{API}/jobs/{job_id}", headers=headers).json()["status"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    started = time.perf_counter()
    for _ in range(10):
        assert client.get(f"{API}/health").status_code == 200
    assert time.perf_counter() - started < 3


# -- settings --------------------------------------------------------------------------------------


def key_pair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


def test_production_without_keys_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="JWT_PRIVATE_KEY_PEM"):
        Settings(env="prod", database_url=f"sqlite:///{tmp_path}/x.db", data_dir=tmp_path)
    private, _public = key_pair()
    with pytest.raises(ValidationError, match="required"):
        Settings(env="prod", jwt_private_key_pem=private, data_dir=tmp_path)  # type: ignore[arg-type]


def test_production_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURVESTUDIO_ENV", "prod")
    monkeypatch.setenv("CURVESTUDIO_DATA_DIR", str(tmp_path))
    with pytest.raises(ValidationError):
        Settings()
    private, public = key_pair()
    monkeypatch.setenv("CURVESTUDIO_JWT_PRIVATE_KEY_PEM", private.replace("\n", "\\n"))
    monkeypatch.setenv("CURVESTUDIO_JWT_PUBLIC_KEY_PEM", public.replace("\n", "\\n"))
    monkeypatch.setenv("CURVESTUDIO_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("CURVESTUDIO_JOB_WORKERS", "3")
    monkeypatch.setenv("CURVESTUDIO_DATABASE_URL", "postgresql://u:p@db/curvestudio")
    settings = Settings()
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
    assert settings.job_workers == 3
    assert settings.database_url == "postgresql+psycopg://u:p@db/curvestudio"
    assert load_jwt_keys(settings)[0].startswith("-----BEGIN PRIVATE KEY-----")
    assert not list(tmp_path.glob("*.pem"))  # production never writes keys to disk


@pytest.mark.parametrize("which", ["garbage", "mismatch"])
def test_production_with_bad_keys_does_not_start(tmp_path: Path, which: str) -> None:
    private, public = key_pair()
    if which == "garbage":
        settings = Settings(
            env="prod",
            data_dir=tmp_path,
            jwt_private_key_pem="not a key",
            jwt_public_key_pem="nope",  # type: ignore[arg-type]
        )
    else:
        settings = Settings(
            env="prod",
            data_dir=tmp_path,
            jwt_private_key_pem=private,
            jwt_public_key_pem=key_pair()[1],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="JWT"):
        create_app(settings)
    assert public


def test_cors_origins_are_explicit() -> None:
    with pytest.raises(ValidationError, match="wildcards"):
        Settings(cors_origins=["*"])
    with pytest.raises(ValidationError, match="wildcards"):
        Settings(cors_origins=["https://*.example.org"])


def test_dev_keys_are_generated_once_and_reused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = load_jwt_keys(settings)
    again = load_jwt_keys(settings)
    assert first == again
    assert (tmp_path / "jwt_private.pem").stat().st_mode & 0o077 == 0


def test_env_example_lists_every_setting() -> None:
    example = (REPO / "apps" / "api" / ".env.example").read_text()
    for name in Settings.model_fields:
        assert f"CURVESTUDIO_{name.upper()}=" in example, name
    assert "BEGIN" not in example  # no real key material


# -- envelope, headers, request ids ----------------------------------------------------------------


def test_error_envelope_shape(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    cases = [
        (shared_client.get(f"{API}/nope"), 404, "NOT_FOUND"),
        (shared_client.delete(f"{API}/health"), 405, "METHOD_NOT_ALLOWED"),
        (shared_client.get(f"{API}/me"), 401, "UNAUTHORIZED"),
        (shared_client.get(f"{API}/datasets/not-a-uuid", headers=headers), 422, "VALIDATION_ERROR"),
        (
            shared_client.get(
                f"{API}/datasets/{'0' * 8}-0000-0000-0000-{'0' * 12}", headers=headers
            ),
            404,
            "NOT_FOUND",
        ),
    ]
    for response, status, code in cases:
        assert response.status_code == status, response.text
        body = response.json()
        assert set(body) == {"error", "message", "detail"}
        assert body["error"] == code
        assert isinstance(body["message"], str) and body["message"]
        assert isinstance(body["detail"], dict)
    assert shared_client.get(f"{API}/me").headers["www-authenticate"] == "Bearer"


def test_unhandled_errors_are_500_envelopes_without_a_traceback(
    client_factory: Callable[..., TestClient], caplog: pytest.LogCaptureFixture
) -> None:
    client = client_factory()

    @client.app.get("/boom")  # type: ignore[attr-defined]
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    with caplog.at_level(logging.ERROR, logger="curvestudio_api"):
        response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "INTERNAL_ERROR"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    assert response.headers["x-request-id"]
    assert any("unhandled error" in r.getMessage() and r.exc_info for r in caplog.records)


def test_request_id_is_echoed_or_generated(shared_client: TestClient) -> None:
    echoed = shared_client.get(f"{API}/health", headers={"X-Request-ID": "trace-123.abc"})
    assert echoed.headers["x-request-id"] == "trace-123.abc"
    generated = shared_client.get(f"{API}/health")
    assert len(generated.headers["x-request-id"]) == 32
    hostile = shared_client.get(
        f"{API}/health", headers={"X-Request-ID": "bad id\twith space" + "x" * 100}
    )
    assert hostile.headers["x-request-id"] != "bad id\twith space"
    assert len(hostile.headers["x-request-id"]) == 32
    other = shared_client.get(f"{API}/health")
    assert other.headers["x-request-id"] != generated.headers["x-request-id"]


def test_security_headers_on_every_response(shared_client: TestClient) -> None:
    for response in (
        shared_client.get(f"{API}/health"),
        shared_client.get(f"{API}/me"),
        shared_client.get(f"{API}/nope"),
    ):
        h = response.headers
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["referrer-policy"]
        assert h["content-security-policy"] == (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'"
        )
        assert "strict-transport-security" not in h  # only in production


def test_cors_is_an_explicit_allowlist(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory(cors_origins=["https://app.example.org"]))
    allowed = client.options(
        f"{API}/health",
        headers={"Origin": "https://app.example.org", "Access-Control-Request-Method": "GET"},
    )
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.org"
    denied = client.get(f"{API}/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers
    same_origin = client_factory().get(
        f"{API}/health", headers={"Origin": "https://app.example.org"}
    )
    assert "access-control-allow-origin" not in same_origin.headers


def test_hsts_in_production(tmp_path: Path) -> None:
    private, public = key_pair()
    app = create_app(
        make_settings(tmp_path, env="prod", jwt_private_key_pem=private, jwt_public_key_pem=public)
    )
    with TestClient(app) as client:
        assert "max-age" in client.get(f"{API}/health").headers["strict-transport-security"]


# -- body limit (streamed) -------------------------------------------------------------------------


def test_streamed_bodies_stop_at_the_limit() -> None:
    delivered: list[int] = []
    outcome: list[Any] = []

    async def reader(scope: Any, receive: Any, send: Any) -> None:
        total = 0
        while True:
            message = await receive()
            total += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        outcome.append(total)

    app = BodyLimitMiddleware(reader, default_limit=1_000, upload_limit=1_000, upload_path="/up")
    chunks = [b"x" * 400 for _ in range(100)]

    async def receive() -> Message:
        delivered.append(1)
        index = len(delivered) - 1
        return {"type": "http.request", "body": chunks[index], "more_body": index < len(chunks) - 1}

    async def send(_message: Message) -> None:
        return None

    scope = {"type": "http", "method": "POST", "path": "/up", "headers": []}
    with pytest.raises(ApiError) as caught:
        asyncio.run(app(scope, receive, send))  # type: ignore[arg-type]
    assert caught.value.status_code == 413
    assert len(delivered) == 3  # 1200 bytes > 1000: nothing after the third chunk was read
    assert outcome == []


# -- logging ---------------------------------------------------------------------------------------


def test_logs_are_json_with_request_ids_and_never_hold_secrets(
    shared_client: TestClient, caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    secret = "Tr0ub4dor&3-distinct-secret"
    email = new_email()
    with caplog.at_level(logging.DEBUG):
        shared_client.post(f"{API}/auth/register", json={"email": email, "password": secret})
        shared_client.post(
            f"{API}/auth/login", json={"email": email, "password": "wrong-" + secret}
        )
        tokens = shared_client.post(
            f"{API}/auth/login", json={"email": email, "password": secret}
        ).json()
        shared_client.post(f"{API}/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        shared_client.get(
            f"{API}/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        shared_client.post(f"{API}/auth/register", json={"email": email, "password": "x"})
    formatter = JsonFormatter()
    lines = [formatter.format(r) for r in caplog.records]
    rendered = "\n".join(lines) + capfd.readouterr().err + caplog.text
    for forbidden in (secret, tokens["access_token"], tokens["refresh_token"], "wrong-" + secret):
        assert forbidden not in rendered
    requests = [json.loads(line) for line in lines if '"message": "request"' in line]
    assert len(requests) >= 6
    assert all(r["request_id"] and r["status"] and r["path"].startswith(API) for r in requests)
    assert all(r["ts"].endswith("+05:30") for r in requests)


# -- database --------------------------------------------------------------------------------------


def test_migration_matches_the_models_and_seeds_the_plan(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    engine = app.state.ctx.engine
    upgrade_to_head(engine)
    upgrade_to_head(engine)  # idempotent
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []
    tables = set(inspect(engine).get_table_names())
    assert {
        "users",
        "refresh_tokens",
        "datasets",
        "fits",
        "jobs",
        "audit_logs",
        "plans",
    } <= tables
    with app.state.ctx.session_factory() as db:
        plans = list(db.scalars(select(Plan)))
    assert [(p.name, p.limits) for p in plans] == [("self-hosted", {"unlimited": True})]


def test_migration_can_be_rolled_back(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    engine = app.state.ctx.engine
    upgrade_to_head(engine)
    command.downgrade(_config(engine), "base")
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}


def test_audit_logs_are_append_only(shared_client: TestClient) -> None:
    signup(shared_client)
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.scalar(select(AuditLog)) is not None
        with pytest.raises(SQLAlchemyError, match="append-only"):
            db.execute(text("UPDATE audit_logs SET action = 'tampered'"))
        db.rollback()
        with pytest.raises(SQLAlchemyError, match="append-only"):
            db.execute(text("DELETE FROM audit_logs"))
        db.rollback()


def test_timestamps_are_stored_in_utc_and_served_in_ist(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    created = shared_client.get(f"{API}/me", headers=headers).json()["created_at"]
    assert created.endswith("+05:30")
    dataset = upload(shared_client, headers, linear_csv()).json()
    assert dataset["created_at"].endswith("+05:30")
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        stored = db.scalar(select(AuditLog).order_by(AuditLog.created_at.desc()))
    assert stored is not None and stored.created_at.utcoffset().total_seconds() == 0
    assert to_ist_iso(stored.created_at).endswith("+05:30")
    assert IST.utcoffset(None).total_seconds() == 5.5 * 3600


# -- OpenAPI ---------------------------------------------------------------------------------------


def test_committed_openapi_matches_the_application() -> None:
    assert DEFAULT_PATH.read_text(encoding="utf-8") == render(generate()), (
        "apps/api/openapi.json is stale: run `uv run python -m curvestudio_api.export_openapi`"
    )


def test_openapi_covers_the_documented_endpoints() -> None:
    paths = generate()["paths"]
    expected = {
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/auth/logout",
        "/api/v1/me",
        "/api/v1/models",
        "/api/v1/models/{model_id}",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/datasets/{dataset_id}/fits",
        "/api/v1/jobs/{job_id}",
        "/api/v1/jobs/{job_id}/cancel",
        "/api/v1/fits/{fit_id}",
        "/api/v1/fits/{fit_id}/predict",
        "/api/v1/fits/{fit_id}/export",
    }
    assert expected <= set(paths)


# -- single-page app -------------------------------------------------------------------------------


def test_the_built_web_app_is_served_with_spa_fallback(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    client = client_factory(settings_factory(web_dist=dist))

    assert "<title>app</title>" in client.get("/").text
    deep = client.get("/datasets/123/fit")
    assert deep.status_code == 200 and "<title>app</title>" in deep.text
    assert deep.headers["cache-control"] == "no-cache"
    asset = client.get("/assets/app.js")
    assert asset.text == "console.log(1)"
    assert "immutable" in asset.headers["cache-control"]
    assert client.get("/favicon.svg").text == "<svg/>"
    for attempt in ("/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/%2e%2e/%2e%2e/secret.txt"):
        assert "top secret" not in client.get(attempt).text
    # the API is untouched and unknown API paths stay JSON
    assert client.get(f"{API}/health").json()["status"] == "ok"
    missing = client.get(f"{API}/nope")
    assert missing.status_code == 404 and missing.json()["error"] == "NOT_FOUND"
    assert client.get("/api").status_code == 404
    assert client.get("/").headers["content-security-policy"].startswith("default-src 'self'")


def test_web_dist_without_index_html_fails_at_startup(
    settings_factory: Callable[..., Settings], tmp_path: Path
) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=r"index\.html"):
        create_app(settings_factory(web_dist=tmp_path / "empty"))
