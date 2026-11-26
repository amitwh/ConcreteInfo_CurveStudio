"""Registration, login, refresh rotation with reuse detection, logout, rate limits."""

import uuid
from collections.abc import Callable
from datetime import date, timedelta

import jwt
import pytest
from api_test_helpers import PASSWORD, new_email, signup
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from curvestudio_api.orm import AuditLog, RefreshToken, User, UserDailyActivity
from curvestudio_api.timeutil import IST, utcnow

API = "/api/v1"


def register(client: TestClient, email: str, password: str = PASSWORD, **extra: str) -> object:
    return client.post(f"{API}/auth/register", json={"email": email, "password": password, **extra})


def login(client: TestClient, email: str, password: str = PASSWORD) -> object:
    return client.post(f"{API}/auth/login", json={"email": email, "password": password})


def test_register_login_me(shared_client: TestClient) -> None:
    email = new_email().upper()
    created = register(shared_client, email, name="  Asha  ")
    assert created.status_code == 201  # type: ignore[attr-defined]
    body = created.json()  # type: ignore[attr-defined]
    assert body["email"] == email.lower()
    assert body["name"] == "Asha"
    assert body["created_at"].endswith("+05:30")
    assert "password" not in str(body)

    tokens = login(shared_client, email.lower()).json()  # type: ignore[attr-defined]
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == 15 * 60
    me = shared_client.get(
        f"{API}/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["email"] == email.lower()


def test_access_token_is_rs256_with_required_claims(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    token = login(shared_client, email).json()["access_token"]  # type: ignore[attr-defined]
    header = jwt.get_unverified_header(token)
    claims = jwt.decode(token, options={"verify_signature": False})
    assert header["alg"] == "RS256"
    assert claims["typ"] == "access"
    assert {"sub", "exp", "iat", "jti"} <= set(claims)
    assert claims["exp"] - claims["iat"] == 15 * 60


def test_duplicate_email_is_409_and_case_insensitive(shared_client: TestClient) -> None:
    email = new_email()
    assert register(shared_client, email).status_code == 201  # type: ignore[attr-defined]
    again = register(shared_client, email.upper())
    assert again.status_code == 409  # type: ignore[attr-defined]
    body = again.json()  # type: ignore[attr-defined]
    assert body["error"] == "CONFLICT"
    assert email not in body["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": "a@b.co", "password": "short"},
        {"email": "a@b.co", "password": "x" * 129},
        {"email": "a@b.co"},
        {"email": "a@b.co", "password": PASSWORD, "is_admin": True},
    ],
)
def test_register_validation_never_echoes_the_password(
    shared_client: TestClient, payload: dict[str, object]
) -> None:
    response = shared_client.post(f"{API}/auth/register", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert isinstance(body["detail"]["errors"], list)
    assert '"input"' not in response.text  # pydantic's echo of the submitted value
    assert PASSWORD not in response.text
    assert "x" * 129 not in response.text


def test_login_failures_look_the_same(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    wrong = login(shared_client, email, "wrong password!!")
    unknown = login(shared_client, new_email())
    assert wrong.status_code == unknown.status_code == 401  # type: ignore[attr-defined]
    assert wrong.json() == unknown.json()  # type: ignore[attr-defined]
    assert wrong.json()["error"] == "UNAUTHORIZED"  # type: ignore[attr-defined]


def test_registration_can_be_disabled(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(registration_enabled=False))  # type: ignore[operator]
    response = register(client, new_email())
    assert response.status_code == 403  # type: ignore[attr-defined]
    assert response.json()["detail"]["code"] == "REGISTRATION_DISABLED"  # type: ignore[attr-defined]


def test_refresh_rotates_and_reuse_revokes_the_chain(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    first = login(shared_client, email).json()  # type: ignore[attr-defined]
    second = shared_client.post(
        f"{API}/auth/refresh", json={"refresh_token": first["refresh_token"]}
    )
    assert second.status_code == 200
    second_tokens = second.json()
    assert second_tokens["refresh_token"] != first["refresh_token"]
    third = shared_client.post(
        f"{API}/auth/refresh", json={"refresh_token": second_tokens["refresh_token"]}
    ).json()

    # Presenting the first (already rotated) token again is reuse: the whole chain dies.
    replay = shared_client.post(
        f"{API}/auth/refresh", json={"refresh_token": first["refresh_token"]}
    )
    assert replay.status_code == 401
    assert replay.json()["error"] == "UNAUTHORIZED"
    for token in (second_tokens["refresh_token"], third["refresh_token"]):
        dead = shared_client.post(f"{API}/auth/refresh", json={"refresh_token": token})
        assert dead.status_code == 401

    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        rows = db.scalars(select(RefreshToken).where(RefreshToken.user_id == user.id)).all()
        assert len(rows) == 3
        assert all(row.revoked_at is not None for row in rows)
        actions = [
            a.action for a in db.scalars(select(AuditLog).where(AuditLog.user_id == user.id))
        ]
        assert "auth.refresh_reuse" in actions


def test_other_sessions_survive_reuse_detection(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    laptop = login(shared_client, email).json()  # type: ignore[attr-defined]
    phone = login(shared_client, email).json()  # type: ignore[attr-defined]
    shared_client.post(f"{API}/auth/refresh", json={"refresh_token": laptop["refresh_token"]})
    shared_client.post(f"{API}/auth/refresh", json={"refresh_token": laptop["refresh_token"]})
    still = shared_client.post(
        f"{API}/auth/refresh", json={"refresh_token": phone["refresh_token"]}
    )
    assert still.status_code == 200


def test_unknown_and_expired_refresh_tokens_are_401(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    assert (
        client_factory().post(f"{API}/auth/refresh", json={"refresh_token": "nope"}).status_code
        == 401
    )
    client = client_factory(settings_factory(refresh_ttl_days=1))  # type: ignore[operator]
    email = new_email()
    register(client, email)
    token = login(client, email).json()["refresh_token"]  # type: ignore[attr-defined]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        from datetime import timedelta

        from curvestudio_api.timeutil import utcnow

        for row in db.scalars(select(RefreshToken)):
            row.expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
    assert client.post(f"{API}/auth/refresh", json={"refresh_token": token}).status_code == 401


def test_logout_revokes_the_refresh_token(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    tokens = login(shared_client, email).json()  # type: ignore[attr-defined]
    out = shared_client.post(f"{API}/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert out.status_code == 204
    again = shared_client.post(
        f"{API}/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert again.status_code == 401
    # logging out with an unknown token reveals nothing
    unknown = shared_client.post(f"{API}/auth/logout", json={"refresh_token": "who-knows"})
    assert unknown.status_code == 204


def test_bad_access_tokens_are_401(
    shared_client: TestClient, client_factory: Callable[..., TestClient]
) -> None:
    headers = signup(shared_client)
    assert shared_client.get(f"{API}/me", headers=headers).status_code == 200
    token = headers["Authorization"].split()[1]
    other = client_factory()  # a different key pair
    assert other.get(f"{API}/me", headers=headers).status_code == 401
    for value in ("Bearer", "Bearer abc.def.ghi", f"Basic {token}", token):
        response = shared_client.get(f"{API}/me", headers={"Authorization": value})
        assert response.status_code == 401, value
        assert response.json()["error"] == "UNAUTHORIZED"
    forged = jwt.encode({"sub": str(uuid.uuid4()), "typ": "access"}, "secret", algorithm="HS256")
    assert (
        shared_client.get(f"{API}/me", headers={"Authorization": f"Bearer {forged}"}).status_code
        == 401
    )


def test_a_refresh_or_wrong_typed_token_is_not_an_access_token(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    tokens = login(shared_client, email).json()  # type: ignore[attr-defined]
    as_bearer = {"Authorization": f"Bearer {tokens['refresh_token']}"}
    assert shared_client.get(f"{API}/me", headers=as_bearer).status_code == 401
    keys = shared_client.app.state.ctx.keys  # type: ignore[attr-defined]
    refresh_typed = jwt.encode(
        {
            **jwt.decode(tokens["access_token"], options={"verify_signature": False}),
            "typ": "refresh",
        },
        keys.private_pem,
        algorithm="RS256",
    )
    assert (
        shared_client.get(
            f"{API}/me", headers={"Authorization": f"Bearer {refresh_typed}"}
        ).status_code
        == 401
    )


def test_expired_access_token_is_401(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory())  # type: ignore[operator]
    headers = signup(client)
    keys = client.app.state.ctx.keys  # type: ignore[attr-defined]
    claims = jwt.decode(headers["Authorization"].split()[1], options={"verify_signature": False})
    claims["exp"] = claims["iat"] - 10
    expired = jwt.encode(claims, keys.private_pem, algorithm="RS256")
    assert (
        client.get(f"{API}/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    )


def test_login_is_rate_limited_per_ip_and_email(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    email = new_email()
    register(client, email)
    statuses = [login(client, email, "wrong password!!").status_code for _ in range(12)]  # type: ignore[attr-defined]
    assert statuses[:9].count(401) == 9
    assert 429 in statuses
    limited = login(client, email, "wrong password!!")
    assert limited.status_code == 429  # type: ignore[attr-defined]
    assert limited.json()["error"] == "RATE_LIMITED"  # type: ignore[attr-defined]
    assert int(limited.headers["Retry-After"]) >= 1  # type: ignore[attr-defined]
    # a different email from the same address has its own allowance
    other = new_email()
    register(client, other)
    assert login(client, other).status_code == 200  # type: ignore[attr-defined]


def test_audit_rows_for_auth_events(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    login(shared_client, email, "wrong password!!")
    tokens = login(shared_client, email).json()  # type: ignore[attr-defined]
    shared_client.post(f"{API}/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        rows = list(db.scalars(select(AuditLog).where(AuditLog.user_id == user.id)))
    assert [r.action for r in rows].count("user.register") == 1
    assert [r.action for r in rows].count("auth.login_failed") == 1
    assert [r.action for r in rows].count("auth.login") == 1
    assert [r.action for r in rows].count("auth.logout") == 1
    assert all(email not in str(r.detail) for r in rows)
    assert all(PASSWORD not in str(r.detail) for r in rows)


def test_new_accounts_are_members_with_the_default_preferences(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.role == "member"
        assert user.preferences == {"error_reports": True, "report_byline": False}


def test_login_stamps_last_login_and_activity_lands_once_per_ist_day(
    shared_client: TestClient,
) -> None:
    email = new_email()
    register(shared_client, email)
    tokens = login(shared_client, email).json()  # type: ignore[attr-defined]
    today = utcnow().astimezone(IST).date()
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.last_login_at is not None
        assert user.last_active_on is None  # a sign-in is not an authenticated request
        assert _daily_rows(db, user.id) == []

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert shared_client.get(f"{API}/me", headers=headers).status_code == 200
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.last_active_on == today
        assert _daily_rows(db, user.id) == [today]

    # a second request on the same day adds nothing
    assert shared_client.get(f"{API}/me", headers=headers).status_code == 200
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert _daily_rows(db, user.id) == [today]

    # on the next IST day the stamp moves on: rewind it and the request lands a new day
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        user.last_active_on = today - timedelta(days=1)
        db.execute(delete(UserDailyActivity).where(UserDailyActivity.user_id == user.id))
        db.commit()
    assert shared_client.get(f"{API}/me", headers=headers).status_code == 200
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.last_active_on == today
        assert _daily_rows(db, user.id) == [today]

    # a row another (concurrent) request already inserted never fails the request: the
    # stamp's commit loses the race, rolls back, and the other request's row stands
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        user.last_active_on = today - timedelta(days=1)
        db.commit()
    assert shared_client.get(f"{API}/me", headers=headers).status_code == 200
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert _daily_rows(db, user.id) == [today]


def _daily_rows(db: Session, user_id: uuid.UUID) -> list[date]:
    """The activity days recorded for an account, oldest first."""
    return sorted(
        db.scalars(select(UserDailyActivity.day).where(UserDailyActivity.user_id == user_id))
    )


def test_user_daily_activity_rows_cascade_with_the_account(shared_client: TestClient) -> None:
    email = new_email()
    headers = signup(shared_client, email)
    assert shared_client.get(f"{API}/me", headers=headers).status_code == 200
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert (
            db.scalar(
                select(func.count(UserDailyActivity.user_id)).where(
                    UserDailyActivity.user_id == user.id
                )
            )
            == 1
        )
        # the append-only audit trail is the one reference that does not cascade
        db.execute(update(AuditLog).where(AuditLog.user_id == user.id).values(user_id=None))
        db.delete(user)
        db.commit()
        assert (
            db.scalar(
                select(func.count(UserDailyActivity.user_id)).where(
                    UserDailyActivity.user_id == user.id
                )
            )
            == 0
        )
