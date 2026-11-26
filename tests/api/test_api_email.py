"""E-mail verification and password reset: tokens, links, revocation and the password policy."""

import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from api_test_helpers import PASSWORD, new_email
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from curvestudio_api import mailer
from curvestudio_api.orm import EmailToken, User
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

API = "/api/v1"
TOKEN_RE = re.compile(r"[?&]token=([A-Za-z0-9_\-]+)")
VERIFIED_MESSAGE = "if an unverified account exists, a new verification email has been sent"
FORGOT_MESSAGE = "if an account exists, a password reset email has been sent"


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """Capture every e-mail the app would send as ``(to, subject, body)``."""
    captured: list[tuple[str, str, str]] = []

    def fake(to: str, subject: str, body: str) -> bool:
        captured.append((to, subject, body))
        return True

    monkeypatch.setattr(mailer, "send_email", fake)
    return captured


@pytest.fixture
def client(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    sent: list[tuple[str, str, str]],
) -> TestClient:
    """An app that requires verification, with the sender captured by ``sent``."""
    return client_factory(settings_factory(require_email_verification=True))


def register(client: TestClient, email: str | None = None, password: str = PASSWORD) -> Any:
    email = email or new_email()
    return client.post(f"{API}/auth/register", json={"email": email, "password": password})


def login(client: TestClient, email: str, password: str = PASSWORD) -> Any:
    return client.post(f"{API}/auth/login", json={"email": email, "password": password})


def token_from(body: str) -> str:
    match = TOKEN_RE.search(body)
    assert match, f"no token in the e-mail body:\n{body}"
    return match.group(1)


def verify(client: TestClient, token: str) -> Any:
    return client.post(f"{API}/auth/verify-email", json={"token": token})


def forgot(client: TestClient, email: str) -> Any:
    return client.post(f"{API}/auth/password/forgot", json={"email": email})


def reset(client: TestClient, token: str, new_password: str) -> Any:
    return client.post(
        f"{API}/auth/password/reset", json={"token": token, "new_password": new_password}
    )


def test_register_creates_an_unverified_account_and_queues_one_email(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    email = new_email()
    created = register(client, email)
    assert created.status_code == 201
    assert created.json()["email_verified"] is False

    assert len(sent) == 1
    to, subject, body = sent[0]
    assert to == email
    assert "verify" in subject.lower()
    assert "/verify-email?token=" in body

    refused = login(client, email)
    assert refused.status_code == 403
    error = refused.json()
    assert error["error"] == "FORBIDDEN"
    assert error["detail"]["code"] == "EMAIL_NOT_VERIFIED"
    assert "access_token" not in error
    assert "refresh_token" not in error


def test_verify_email_allows_login_and_the_token_is_single_use(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    email = new_email()
    register(client, email)
    token = token_from(sent[0][2])

    verified = verify(client, token)
    assert verified.status_code == 200
    assert verified.json() == {"email": email, "verified": True}
    assert login(client, email).status_code == 200

    again = verify(client, token)
    assert again.status_code == 400
    assert again.json()["error"] == "INVALID_TOKEN"


def test_expired_verification_token_is_rejected(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    email = new_email()
    register(client, email)
    token = token_from(sent[0][2])
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        assert user is not None
        db.execute(
            update(EmailToken)
            .where(EmailToken.user_id == user.id)
            .values(expires_at=utcnow() - timedelta(minutes=1))
        )
        db.commit()

    rejected = verify(client, token)
    assert rejected.status_code == 400
    assert rejected.json()["error"] == "INVALID_TOKEN"


def test_resend_is_generic_and_invalidates_the_previous_token(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    unknown = client.post(f"{API}/auth/verify-email/resend", json={"email": new_email()})
    assert unknown.status_code == 202
    assert unknown.json() == {"message": VERIFIED_MESSAGE}

    email = new_email()
    register(client, email)
    assert len(sent) == 1
    old_token = token_from(sent[0][2])

    resent = client.post(f"{API}/auth/verify-email/resend", json={"email": email})
    assert resent.status_code == 202
    assert resent.json() == {"message": VERIFIED_MESSAGE}
    assert len(sent) == 2

    assert verify(client, old_token).status_code == 400
    assert verify(client, token_from(sent[-1][2])).status_code == 200


def test_forgot_and_reset_change_the_password(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    unknown = forgot(client, new_email())
    assert unknown.status_code == 202
    assert unknown.json() == {"message": FORGOT_MESSAGE}

    email = new_email()
    register(client, email)
    assert len(sent) == 1
    assert forgot(client, email).status_code == 202
    assert len(sent) == 2
    assert "reset" in sent[-1][1].lower()
    token = token_from(sent[-1][2])

    new_password = "a completely new passphrase"
    changed = reset(client, token, new_password)
    assert changed.status_code == 200
    assert changed.json() == {"message": "your password has been reset; sign in with the new one"}

    assert login(client, email, PASSWORD).status_code == 401
    assert login(client, email, new_password).status_code == 200
    assert reset(client, token, new_password).status_code == 400


def test_reset_revokes_every_refresh_token(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    email = new_email()
    register(client, email)
    assert verify(client, token_from(sent[0][2])).status_code == 200
    tokens = login(client, email).json()

    assert forgot(client, email).status_code == 202
    new_password = "yet another passphrase"
    assert reset(client, token_from(sent[-1][2]), new_password).status_code == 200

    reused = client.post(f"{API}/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert reused.status_code == 401


def test_reset_marks_an_unverified_account_verified(
    client: TestClient, sent: list[tuple[str, str, str]]
) -> None:
    email = new_email()
    register(client, email)  # no verification step
    assert forgot(client, email).status_code == 202
    new_password = "possession proves control"
    assert reset(client, token_from(sent[-1][2]), new_password).status_code == 200
    assert login(client, email, new_password).status_code == 200


@pytest.mark.parametrize("new_password", ["short", "password123"])
def test_reset_enforces_the_password_policy(
    client: TestClient, sent: list[tuple[str, str, str]], new_password: str
) -> None:
    email = new_email()
    register(client, email)
    assert forgot(client, email).status_code == 202
    token = token_from(sent[-1][2])
    assert reset(client, token, new_password).status_code == 422


def test_verification_can_be_disabled(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    sent: list[tuple[str, str, str]],
) -> None:
    client = client_factory(settings_factory(require_email_verification=False))
    email = new_email()
    created = register(client, email)
    assert created.status_code == 201
    assert created.json()["email_verified"] is True
    assert sent == []
    assert login(client, email).status_code == 200
