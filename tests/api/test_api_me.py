"""The profile endpoints: name, preferences, password change, signing out other sessions,
and the typed confirmation before the account is deleted."""

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import PASSWORD, new_email, signup
from curvestudio_api.orm import AuditLog, User
from curvestudio_api.routers.auth import FAILED_LOGIN_ROWS_PER_DAY
from curvestudio_api.security import verify_password

API = "/api/v1"
ME = f"{API}/me"
NEW_PASSWORD = "orchid lantern marble fountain"
DEFAULT_PREFERENCES = {"error_reports": True, "report_byline": False}


def register(client: TestClient, email: str) -> None:
    response = client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201, response.text


def login(client: TestClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    """One sign-in: the access/refresh pair of one session."""
    response = client.post(f"{API}/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()


def bearer(session: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {session['access_token']}"}


def refresh(client: TestClient, session: dict[str, str]) -> int:
    """Whether the session's refresh token is still accepted."""
    response = client.post(f"{API}/auth/refresh", json={"refresh_token": session["refresh_token"]})
    return response.status_code


def change_password(
    client: TestClient, headers: dict[str, str], refresh_token: str, **passwords: str
) -> Response:
    """``POST /me/password`` presenting ``refresh_token`` as the session to keep."""
    body: dict[str, str] = {
        "current_password": PASSWORD,
        "new_password": NEW_PASSWORD,
        "refresh_token": refresh_token,
    }
    body.update(passwords)
    return client.post(f"{ME}/password", headers=headers, json=body)


def delete_me(client: TestClient, headers: dict[str, str], typed: str | None = None) -> Response:
    """``DELETE /me`` with the typed phrase, or with no body at all when ``typed`` is None
    (the test client's own ``delete`` cannot carry a body; ``request`` can)."""
    if typed is None:
        return client.delete(ME, headers=headers)
    return client.request("DELETE", ME, headers=headers, json={"confirm": typed})


def field_problem(response: Response) -> str:
    """The message of the 422's first field error."""
    return response.json()["detail"]["errors"][0]["msg"]


def audit_rows(db: Session, user_id: uuid.UUID, action: str) -> list[AuditLog]:
    return list(
        db.scalars(select(AuditLog).where(AuditLog.user_id == user_id, AuditLog.action == action))
    )


def test_name_update_trims_and_returns_me(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = shared_client.patch(ME, headers=headers, json={"name": "  Asha Verma  "})
    assert response.status_code == 200, response.text
    user_id = uuid.UUID(response.json()["id"])
    assert response.json()["name"] == "Asha Verma"
    assert shared_client.get(ME, headers=headers).json()["name"] == "Asha Verma"
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        rows = audit_rows(db, user_id, "user.name_change")
    assert len(rows) == 1
    assert rows[0].detail == {}

    # the name it already has, however it is padded, is neither written nor audited
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        written_at = db.get(User, user_id).updated_at
    same = shared_client.patch(ME, headers=headers, json={"name": " Asha Verma "})
    assert same.status_code == 200, same.text
    assert same.json()["name"] == "Asha Verma"
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.get(User, user_id).updated_at == written_at
        rows = audit_rows(db, user_id, "user.name_change")
    assert len(rows) == 1


def test_name_update_rejects_blank_and_overlong(shared_client: TestClient) -> None:
    email = new_email()
    headers = signup(shared_client, email)
    assert shared_client.patch(ME, headers=headers, json={"name": "Asha"}).status_code == 200
    for bad in ("", "   ", "x" * 101):
        response = shared_client.patch(ME, headers=headers, json={"name": bad})
        assert response.status_code == 422, repr(bad)
        assert response.json()["error"] == "VALIDATION_ERROR"
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.name == "Asha"


def test_preferences_partial_update_and_audit(shared_client: TestClient) -> None:
    email = new_email()
    headers = signup(shared_client, email)
    response = shared_client.put(f"{ME}/preferences", headers=headers, json={"report_byline": True})
    assert response.status_code == 200, response.text
    byline_on = {"error_reports": True, "report_byline": True}
    assert response.json()["preferences"] == byline_on
    # a new request reads the row: the change was stored, not just echoed
    assert shared_client.get(ME, headers=headers).json()["preferences"] == byline_on
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert user.preferences == byline_on
        user_id = user.id
        rows = audit_rows(db, user_id, "user.preferences_change")
    assert len(rows) == 1
    assert rows[0].detail == {"report_byline": True}

    # the same value again changes nothing and writes no second row
    again = shared_client.put(f"{ME}/preferences", headers=headers, json={"report_byline": True})
    assert again.status_code == 200
    assert again.json()["preferences"] == byline_on
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert len(audit_rows(db, user_id, "user.preferences_change")) == 1

    # the other key, on its own, is recorded alone
    other = shared_client.put(f"{ME}/preferences", headers=headers, json={"error_reports": False})
    assert other.json()["preferences"] == {"error_reports": False, "report_byline": True}
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        rows = audit_rows(db, user_id, "user.preferences_change")
    assert [row.detail for row in rows] == [{"report_byline": True}, {"error_reports": False}]


def test_preferences_reject_unknown_key_and_empty_body(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    for payload in ({"newsletter": True}, {}):
        response = shared_client.put(f"{ME}/preferences", headers=headers, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"] == "VALIDATION_ERROR"
    assert shared_client.get(ME, headers=headers).json()["preferences"] == DEFAULT_PREFERENCES


def test_me_reports_role_preferences_and_last_login(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = shared_client.get(ME, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "member"
    assert body["preferences"] == DEFAULT_PREFERENCES
    last_login = datetime.fromisoformat(body["last_login_at"])
    assert last_login.tzinfo is not None
    assert last_login.utcoffset() == timedelta(hours=5, minutes=30)
    assert body["last_login_at"].endswith("+05:30")


def test_password_change_requires_the_current_password(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    session_a, session_b = login(shared_client, email), login(shared_client, email)
    response = change_password(
        shared_client,
        bearer(session_a),
        session_a["refresh_token"],
        current_password="wrong password!!",
    )
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "FORBIDDEN"
    assert body["message"] == "the current password is not correct"
    assert set(body) == {"error", "message", "detail"}
    login(shared_client, email)  # the old password still signs in
    assert refresh(shared_client, session_b) == 200
    assert refresh(shared_client, session_a) == 200

    # the refusal leaves a trail: the address only as a digest, and no password text
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        rows = audit_rows(db, user.id, "auth.password_check_failed")
    assert len(rows) == 1
    assert set(rows[0].detail) == {"email_digest", "ip_digest"}
    assert email not in str(rows[0].detail)
    assert PASSWORD not in str(rows[0].detail)
    assert "wrong password!!" not in str(rows[0].detail)


def test_password_check_failures_share_the_sign_in_failure_cap(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    session = login(shared_client, email)
    headers = bearer(session)

    def wrong_sign_in() -> int:
        response = shared_client.post(
            f"{API}/auth/login", json={"email": email, "password": "wrong password!!"}
        )
        return response.status_code

    def wrong_password_check() -> int:
        return change_password(
            shared_client, headers, session["refresh_token"], current_password="wrong password!!"
        ).status_code

    def failed() -> list[str]:
        """The account's audit rows for refused sign-ins and password checks, oldest first."""
        with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
            user = db.scalar(select(User).where(User.email == email))
            assert user is not None
            rows = db.scalars(
                select(AuditLog)
                .where(AuditLog.user_id == user.id, AuditLog.action.like("%failed%"))
                .order_by(AuditLog.created_at)
            )
            return [row.action for row in rows]

    # the same daily budget, whichever endpoint the failures come through
    for _ in range(FAILED_LOGIN_ROWS_PER_DAY // 2):
        assert wrong_sign_in() == 401
        assert wrong_password_check() == 403
    assert len(failed()) == FAILED_LOGIN_ROWS_PER_DAY

    # the next failure is the single marker, whichever endpoint it comes through; after it
    # neither endpoint adds a row
    assert wrong_password_check() == 403
    assert wrong_sign_in() == 401
    assert wrong_password_check() == 403
    actions = failed()
    assert actions.count("auth.login_failed") == FAILED_LOGIN_ROWS_PER_DAY // 2
    assert actions.count("auth.password_check_failed") == FAILED_LOGIN_ROWS_PER_DAY // 2
    assert actions.count("auth.password_check_failed_capped") == 1
    assert "auth.login_failed_capped" not in actions
    assert actions[-1] == "auth.password_check_failed_capped"
    assert len(actions) == FAILED_LOGIN_ROWS_PER_DAY + 1


def test_password_change_is_rate_limited(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory()
    email = new_email()
    register(client, email)
    session = login(client, email)
    wrong = {"current_password": "wrong password!!"}
    statuses = [
        change_password(client, bearer(session), session["refresh_token"], **wrong).status_code
        for _ in range(12)
    ]
    first_limited = statuses.index(429)
    assert first_limited > 0
    assert statuses[:first_limited] == [403] * first_limited
    assert statuses[first_limited:] == [429] * (len(statuses) - first_limited)

    # the limit is checked before anything else: the right password is refused too, and
    # the password is left as it was
    limited = change_password(client, bearer(session), session["refresh_token"])
    assert limited.status_code == 429
    assert limited.json()["error"] == "RATE_LIMITED"
    assert int(limited.headers["Retry-After"]) >= 1
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        assert verify_password(user.password_hash, PASSWORD)


def test_password_change_rejects_same_and_weak_password(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    session = login(shared_client, email)
    same = change_password(
        shared_client, bearer(session), session["refresh_token"], new_password=PASSWORD
    )
    assert same.status_code == 422
    assert same.json()["error"] == "VALIDATION_ERROR"

    weak = change_password(
        shared_client, bearer(session), session["refresh_token"], new_password="1234567890"
    )
    assert weak.status_code == 422
    assert weak.json()["error"] == "VALIDATION_ERROR"
    assert weak.json()["detail"]["errors"][0]["loc"] == ["body", "new_password"]
    assert field_problem(weak) == "this password is too common; choose another"

    login(shared_client, email)  # a refused change leaves the password alone


def test_password_change_revokes_other_sessions_and_keeps_the_current_one(
    shared_client: TestClient,
) -> None:
    email = new_email()
    register(shared_client, email)
    session_a, session_b = login(shared_client, email), login(shared_client, email)
    response = change_password(shared_client, bearer(session_a), session_a["refresh_token"])
    assert response.status_code == 200, response.text
    assert refresh(shared_client, session_b) == 401
    assert refresh(shared_client, session_a) == 200
    login(shared_client, email, NEW_PASSWORD)
    old = shared_client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD})
    assert old.status_code == 401

    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        rows = list(db.scalars(select(AuditLog).where(AuditLog.user_id == user.id)))
    changes = [row for row in rows if row.action == "user.password_change"]
    assert len(changes) == 1
    assert changes[0].detail == {}
    assert all(PASSWORD not in str(row.detail) for row in rows)
    assert all(NEW_PASSWORD not in str(row.detail) for row in rows)


def test_password_change_with_an_unknown_session_token_revokes_every_session(
    shared_client: TestClient,
) -> None:
    email = new_email()
    register(shared_client, email)
    session_a, session_b, session_c = (login(shared_client, email) for _ in range(3))
    response = change_password(shared_client, bearer(session_a), "not-a-live-token")
    assert response.status_code == 200, response.text
    for session in (session_a, session_b, session_c):
        assert refresh(shared_client, session) == 401
    login(shared_client, email, NEW_PASSWORD)
    user_id = uuid.UUID(response.json()["id"])
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        rows = audit_rows(db, user_id, "user.password_change")
    assert len(rows) == 1
    assert rows[0].detail == {}


def test_revoke_all_keeps_the_presented_session(shared_client: TestClient) -> None:
    email = new_email()
    register(shared_client, email)
    session_a, session_b = login(shared_client, email), login(shared_client, email)
    response = shared_client.post(
        f"{ME}/sessions/revoke-all",
        headers=bearer(session_a),
        json={"refresh_token": session_a["refresh_token"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["email"] == email
    assert refresh(shared_client, session_b) == 401
    assert refresh(shared_client, session_a) == 200
    user_id = uuid.UUID(response.json()["id"])
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        rows = audit_rows(db, user_id, "user.sessions_revoked")
    assert len(rows) == 1
    assert rows[0].detail == {"revoked": 1}

    # a token that is not one of the account's live ones keeps nothing. A's refresh above
    # rotated its token, so the live ones are now its successor, C and D.
    session_c, session_d = login(shared_client, email), login(shared_client, email)
    response = shared_client.post(
        f"{ME}/sessions/revoke-all",
        headers=bearer(session_a),
        json={"refresh_token": "not-a-live-token"},
    )
    assert response.status_code == 200, response.text
    assert refresh(shared_client, session_c) == 401
    assert refresh(shared_client, session_d) == 401
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        rows = audit_rows(db, user_id, "user.sessions_revoked")
    assert [row.detail for row in rows] == [{"revoked": 1}, {"revoked": 3}]


def test_delete_me_requires_the_typed_confirmation(
    client_factory: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = client_factory()
    email = new_email()
    signup(client, email)
    session = login(client, email)
    headers = bearer(session)
    phrase = email.split("@", 1)[0]
    user_id = uuid.UUID(client.get(ME, headers=headers).json()["id"])
    cancelled: list[uuid.UUID] = []
    jobs = client.app.state.ctx.jobs  # type: ignore[attr-defined]
    monkeypatch.setattr(jobs, "cancel_for_user", cancelled.append)

    # no body, or the wrong words: nothing is deleted, revoked or cancelled
    assert delete_me(client, headers).status_code == 422
    wrong = delete_me(client, headers, typed="delete my account")
    assert wrong.status_code == 422
    assert wrong.json()["error"] == "VALIDATION_ERROR"
    assert "delete my account" not in wrong.text
    assert client.get(ME, headers=headers).status_code == 200
    assert refresh(client, session) == 200
    assert cancelled == []

    # the part before the @, in capitals and padded: the account goes
    typed = f"  {phrase.upper()}  "
    assert delete_me(client, headers, typed=typed).status_code == 204
    assert cancelled == [user_id]
    assert client.get(ME, headers=headers).status_code == 401
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.scalar(select(User).where(User.email == email)) is None
