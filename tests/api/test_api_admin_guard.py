"""The admin dependency (``AdminUser``), promotion from ``CURVESTUDIO_ADMIN_EMAILS``,
and the 0007 migration's round trip."""

from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from alembic import command
from api_test_helpers import PASSWORD, make_settings, new_email
from fastapi.testclient import TestClient
from sqlalchemy import Engine, inspect, select, text

from curvestudio_api import oauth
from curvestudio_api.app import create_app
from curvestudio_api.deps import AdminUser
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.orm import AuditLog, User
from curvestudio_api.settings import Settings

API = "/api/v1"


def register(client: TestClient, email: str) -> object:
    return client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD})


def login(client: TestClient, email: str) -> object:
    return client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD})


def mount_admin_scratch_route(client: TestClient) -> str:
    """A throwaway route guarded by ``AdminUser`` — the dependency has no route of its own
    until the admin console (P5) ships."""
    path = f"{API}/admin/scratch-whoami"

    @client.app.get(path)  # type: ignore[attr-defined]
    def scratch_whoami(user: AdminUser) -> dict[str, str]:
        return {"id": str(user.id)}

    return path


def role_change_rows(client: TestClient, user_id: object) -> list[AuditLog]:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return list(
            db.scalars(
                select(AuditLog).where(
                    AuditLog.user_id == user_id, AuditLog.action == "user.role_change"
                )
            )
        )


def test_admin_guard_admits_an_admin_and_refuses_a_member(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    owner, member = new_email(), new_email()
    client = client_factory(settings_factory(admin_emails=[owner]))
    assert register(client, owner).status_code == 201  # type: ignore[attr-defined]
    assert register(client, member).status_code == 201  # type: ignore[attr-defined]
    path = mount_admin_scratch_route(client)

    admin_headers = {"Authorization": f"Bearer {login(client, owner).json()['access_token']}"}
    allowed = client.get(path, headers=admin_headers)
    assert allowed.status_code == 200
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == owner))
        assert user is not None
        assert allowed.json() == {"id": str(user.id)}

    member_headers = {"Authorization": f"Bearer {login(client, member).json()['access_token']}"}
    refused = client.get(path, headers=member_headers)
    assert refused.status_code == 403
    body = refused.json()
    assert body["error"] == "FORBIDDEN"
    assert set(body) == {"error", "message", "detail"}  # no data rides along
    assert client.get(path).status_code == 401


def test_a_demoted_administrator_loses_access_at_once(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    owner = new_email()
    client = client_factory(settings_factory(admin_emails=[owner]))
    assert register(client, owner).status_code == 201  # type: ignore[attr-defined]
    path = mount_admin_scratch_route(client)
    headers = {"Authorization": f"Bearer {login(client, owner).json()['access_token']}"}
    assert client.get(path, headers=headers).status_code == 200

    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == owner))
        assert user is not None
        user.role = "member"
        db.commit()

    refused = client.get(path, headers=headers)
    assert refused.status_code == 403
    assert refused.json()["error"] == "FORBIDDEN"


def test_admin_emails_are_comma_separated_and_case_folded() -> None:
    """The env var is a comma-separated list; entries are addresses, so lower-cased."""
    settings = Settings(env="test", admin_emails=" Owner@Example.org , second@Example.org,")
    assert settings.admin_emails == ["owner@example.org", "second@example.org"]
    assert Settings(env="test").admin_emails == []
    # a list built in code is folded the same way as the environment string
    assert Settings(env="test", admin_emails=[" Owner@Example.org ", ""]).admin_emails == [
        "owner@example.org"
    ]


def test_a_listed_address_is_promoted_on_login_once_and_audited_once(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    email = new_email()
    client = client_factory(settings_factory(admin_emails=[email]))
    assert register(client, email).status_code == 201  # type: ignore[attr-defined]

    assert login(client, email).status_code == 200  # type: ignore[attr-defined]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        user_id = user.id
        assert user.role == "admin"
    rows = role_change_rows(client, user_id)
    assert len(rows) == 1
    assert rows[0].detail == {}  # a role change never carries the address

    assert login(client, email).status_code == 200  # type: ignore[attr-defined]
    assert len(role_change_rows(client, user_id)) == 1  # the list never re-audits


def test_an_unverified_listed_account_is_not_promoted(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    email = new_email()
    client = client_factory(settings_factory(admin_emails=[email]))
    assert register(client, email).status_code == 201  # type: ignore[attr-defined]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        user.email_verified_at = None
        db.commit()
        user_id = user.id

    # verification is not enforced in the test settings, so the sign-in succeeds
    assert login(client, email).status_code == 200  # type: ignore[attr-defined]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.role == "member"
    assert role_change_rows(client, user_id) == []


def test_a_listed_address_is_promoted_on_the_oauth_exchange(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    email = new_email()

    def fake_http_json(
        url: str, data: dict[str, str] | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, object]:
        del data, headers
        if url.lower().endswith("token"):
            return {"access_token": "provider-access-token"}
        return {"sub": "google-sub", "email": email, "email_verified": True, "name": "Owner"}

    monkeypatch.setattr(oauth, "_http_json", fake_http_json)
    client = client_factory(
        settings_factory(
            admin_emails=[email],
            oauth_google_client_id="google-client",
            oauth_google_client_secret="google-secret",
        )
    )

    def sign_in_with_google() -> object:
        started = client.get(f"{API}/auth/oauth/google/start", follow_redirects=False)
        state = parse_qs(urlparse(started.headers["location"]).query)["state"][0]
        redirect = client.get(
            f"{API}/auth/oauth/google/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )
        code = parse_qs(urlparse(redirect.headers["location"]).query)["code"][0]
        return client.post(f"{API}/auth/oauth/exchange", json={"code": code})

    def exchanges_for(user_id: object) -> int:
        with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
            return [
                row.action
                for row in db.scalars(select(AuditLog).where(AuditLog.user_id == user_id))
            ].count("auth.oauth_exchange")

    first = sign_in_with_google()
    assert first.status_code == 200  # type: ignore[attr-defined]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        user_id = user.id
        assert user.role == "admin"
    assert exchanges_for(user_id) == 1
    assert len(role_change_rows(client, user_id)) == 1

    assert sign_in_with_google().status_code == 200  # type: ignore[attr-defined]
    assert exchanges_for(user_id) == 2  # the second exchange really happened
    assert len(role_change_rows(client, user_id)) == 1  # already an administrator: no new row


def _users_indexes(engine: Engine) -> dict[str, str]:
    """The raw index definitions on ``users`` (SQLAlchemy skips expression indexes)."""
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'users'")
        )
        return {str(name): str(sql) for name, sql in rows}


def test_migration_0007_round_trip_keeps_the_functional_email_index(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    engine = app.state.ctx.engine  # type: ignore[attr-defined]
    upgrade_to_head(engine)
    config = _config(engine)
    expected_index = "CREATE UNIQUE INDEX uq_users_email_lower ON users (lower(email))"

    command.downgrade(config, "0006")
    tables = inspect(engine)
    assert "user_daily_activity" not in set(tables.get_table_names())
    columns = {column["name"] for column in tables.get_columns("users")}
    assert not {"role", "last_login_at", "last_active_on", "preferences"} & columns
    assert _users_indexes(engine).get("uq_users_email_lower") == expected_index

    upgrade_to_head(engine)
    tables = inspect(engine)
    assert "user_daily_activity" in set(tables.get_table_names())
    columns = {column["name"] for column in tables.get_columns("users")}
    assert {"role", "last_login_at", "last_active_on", "preferences"} <= columns
    assert _users_indexes(engine).get("uq_users_email_lower") == expected_index
