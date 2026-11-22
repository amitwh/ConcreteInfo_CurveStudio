"""Social sign-in: providers, PKCE start, callback, single-use exchange, error redirects."""

import base64
import hashlib
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from conftest import PASSWORD, new_email
from curvestudio_api import oauth
from curvestudio_api.orm import OAuthCode, User
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

API = "/api/v1"
COOKIE = "oauth_flow"
APP_URL = "http://localhost:5173"  # Settings.app_url default
GOOGLE_AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"


def oauth_settings(settings_factory: Callable[..., Settings], **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "oauth_google_client_id": "google-client",
        "oauth_google_client_secret": "google-secret",
        "oauth_microsoft_client_id": "microsoft-client",
        "oauth_microsoft_client_secret": "microsoft-secret",
        "oauth_microsoft_tenant": "common",
        "oauth_linkedin_client_id": "linkedin-client",
        "oauth_linkedin_client_secret": "linkedin-secret",
    }
    values.update(overrides)
    return settings_factory(**values)


@pytest.fixture
def provider_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Replace the network with token and userinfo payloads the test controls."""
    payloads: dict[str, dict[str, Any]] = {
        "token": {"access_token": "provider-access-token"},
        "identity": {},
    }

    def fake_http_json(
        url: str, data: dict[str, str] | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        del data, headers
        if url.lower().endswith("token"):  # every provider token endpoint ends in "token"
            return dict(payloads["token"])
        return dict(payloads["identity"])

    monkeypatch.setattr(oauth, "_http_json", fake_http_json)
    return payloads


def start(client: TestClient, provider: str = "google") -> tuple[str, str]:
    """Start a flow; returns the provider location and its state."""
    response = client.get(f"{API}/auth/oauth/{provider}/start", follow_redirects=False)
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    return location, parse_qs(urlparse(location).query)["state"][0]


def callback(
    client: TestClient,
    provider: str,
    *,
    state: str,
    code: str | None = "auth-code",
    **extra: str,
) -> Any:
    params: dict[str, str] = {"state": state, **extra}
    if code is not None:
        params["code"] = code
    return client.get(
        f"{API}/auth/oauth/{provider}/callback", params=params, follow_redirects=False
    )


def exchange_code_from(location: str) -> str:
    return parse_qs(urlparse(location).query)["code"][0]


def test_providers_lists_only_configured_providers(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
) -> None:
    empty = client_factory(settings_factory())
    assert empty.get(f"{API}/auth/oauth/providers").json() == {"providers": []}

    google_only = client_factory(
        oauth_settings(
            settings_factory,
            oauth_microsoft_client_id="",
            oauth_microsoft_client_secret="",
            oauth_linkedin_client_id="",
            oauth_linkedin_client_secret="",
        )
    )
    listed = google_only.get(f"{API}/auth/oauth/providers")
    assert listed.status_code == 200
    assert listed.json() == {"providers": ["google"]}


def test_start_redirects_with_state_and_pkce_and_sets_the_flow_cookie(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    response = client.get(f"{API}/auth/oauth/google/start", follow_redirects=False)

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == GOOGLE_AUTHORIZE
    query = parse_qs(parsed.query)
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["google-client"]
    assert query["redirect_uri"] == [f"{APP_URL}/api/v1/auth/oauth/google/callback"]
    assert query["scope"] == ["openid email profile"]
    assert query["code_challenge_method"] == ["S256"]

    set_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert f"{COOKIE}=" in set_cookie

    value = response.cookies.get(COOKIE)
    assert value is not None
    provider, state, verifier = value.split(":")
    assert provider == "google"
    assert state == query["state"][0]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    assert query["code_challenge"] == [challenge]


def test_google_callback_exchanges_a_single_use_code_for_tokens(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    email = new_email()
    provider_calls["identity"] = {
        "sub": "google-sub",
        "email": email,
        "email_verified": True,
        "name": "Grace Hopper",
    }
    _, state = start(client)
    redirect = callback(client, "google", state=state)

    assert redirect.status_code == 302
    location = redirect.headers["location"]
    assert location.startswith(f"{APP_URL}/oauth/callback?code=")
    raw = exchange_code_from(location)
    assert client.cookies.get(COOKIE) is None  # the flow cookie is cleared

    exchanged = client.post(f"{API}/auth/oauth/exchange", json={"code": raw})
    assert exchanged.status_code == 200
    tokens = exchanged.json()
    assert tokens["token_type"] == "bearer"
    me = client.get(f"{API}/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == email
    assert me.json()["email_verified"] is True
    assert me.json()["name"] == "Grace Hopper"

    reused = client.post(f"{API}/auth/oauth/exchange", json={"code": raw})
    assert reused.status_code == 400
    assert reused.json()["error"] == "INVALID_TOKEN"
    assert reused.json()["message"] == "this sign-in link is invalid or has expired"

    # The random, unknowable stored hash means password sign-in cannot be used.
    refused = client.post(f"{API}/auth/login", json={"email": email, "password": PASSWORD})
    assert refused.status_code == 401


def test_callback_verifies_an_existing_unverified_user_without_duplicating_it(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    email = new_email()
    assert (
        client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD}).status_code
        == 201
    )
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(func.lower(User.email) == email))
        assert user is not None
        user.email_verified_at = None
        db.commit()
        original_id = user.id

    provider_calls["identity"] = {
        "sub": "google-sub",
        "email": email.upper(),
        "email_verified": True,
        "name": "",
    }
    _, state = start(client)
    redirect = callback(client, "google", state=state)
    raw = exchange_code_from(redirect.headers["location"])
    assert client.post(f"{API}/auth/oauth/exchange", json={"code": raw}).status_code == 200

    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        users = list(db.scalars(select(User).where(func.lower(User.email) == email)))
        assert len(users) == 1
        assert users[0].id == original_id
        assert users[0].email_verified_at is not None


def test_state_mismatch_redirects_to_login_and_clears_the_cookie(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    _, state = start(client)

    mismatch = callback(client, "google", state="not-the-state")
    assert mismatch.status_code == 302
    assert mismatch.headers["location"] == f"{APP_URL}/login?oauth_error=state"
    assert client.cookies.get(COOKIE) is None

    _, state = start(client)
    cross_provider = callback(client, "microsoft", state=state)
    assert cross_provider.headers["location"] == f"{APP_URL}/login?oauth_error=state"


def test_provider_error_redirects_to_login(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    _, state = start(client)

    denied = callback(client, "google", state=state, code=None, error="access_denied")
    assert denied.status_code == 302
    assert denied.headers["location"] == f"{APP_URL}/login?oauth_error=provider"


def test_unverified_google_identity_is_rejected(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    provider_calls["identity"] = {
        "sub": "google-sub",
        "email": new_email(),
        "email_verified": False,
        "name": "Unverified",
    }
    _, state = start(client)

    rejected = callback(client, "google", state=state)
    assert rejected.status_code == 302
    assert rejected.headers["location"] == f"{APP_URL}/login?oauth_error=provider"


def test_microsoft_identity_without_email_is_rejected(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    provider_calls["identity"] = {"sub": "microsoft-sub", "name": "No Email"}
    _, state = start(client, "microsoft")

    rejected = callback(client, "microsoft", state=state)
    assert rejected.status_code == 302
    assert rejected.headers["location"] == f"{APP_URL}/login?oauth_error=provider"


def test_microsoft_identity_is_accepted_when_it_returns_an_email(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    email = new_email()
    # No email_verified claim: Microsoft vouches for the address by policy.
    provider_calls["identity"] = {"sub": "microsoft-sub", "email": email, "name": "Satya Nadella"}
    _, state = start(client, "microsoft")

    accepted = callback(client, "microsoft", state=state)
    assert accepted.status_code == 302
    assert accepted.headers["location"].startswith(f"{APP_URL}/oauth/callback?code=")


def test_microsoft_identity_falls_back_to_preferred_username(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    """Personal Microsoft accounts may carry the address in ``preferred_username``."""
    client = client_factory(oauth_settings(settings_factory))
    email = new_email()
    provider_calls["identity"] = {
        "sub": "microsoft-personal-sub",
        "preferred_username": email,
        "name": "Personal Account",
    }
    _, state = start(client, "microsoft")

    accepted = callback(client, "microsoft", state=state)
    assert accepted.status_code == 302
    assert accepted.headers["location"].startswith(f"{APP_URL}/oauth/callback?code=")


def test_start_for_a_disabled_or_unknown_provider_is_404(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
) -> None:
    client = client_factory(
        oauth_settings(
            settings_factory,
            oauth_linkedin_client_id="",
            oauth_linkedin_client_secret="",
        )
    )
    for provider in ("linkedin", "github"):
        response = client.get(f"{API}/auth/oauth/{provider}/start", follow_redirects=False)
        assert response.status_code == 404, provider


def test_exchange_rejects_an_expired_code(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    provider_calls: dict[str, dict[str, Any]],
) -> None:
    client = client_factory(oauth_settings(settings_factory))
    provider_calls["identity"] = {
        "sub": "google-sub",
        "email": new_email(),
        "email_verified": True,
        "name": "Late",
    }
    _, state = start(client)
    redirect = callback(client, "google", state=state)
    raw = exchange_code_from(redirect.headers["location"])
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.execute(update(OAuthCode).values(expires_at=utcnow() - timedelta(seconds=1)))
        db.commit()

    expired = client.post(f"{API}/auth/oauth/exchange", json={"code": raw})
    assert expired.status_code == 400
    assert expired.json()["error"] == "INVALID_TOKEN"
    assert expired.json()["message"] == "this sign-in link is invalid or has expired"


def test_http_error_keeps_the_provider_reason(monkeypatch) -> None:
    """The token error body names the cause (AADSTS…); it must reach the callback log."""
    import io
    import json
    import urllib.error

    from curvestudio_api import oauth as oauth_module

    body = json.dumps(
        {"error": "invalid_client", "error_description": "AADSTS7000215: Invalid client secret"}
    ).encode()

    def failing(*args: object, **kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://example.test", 401, "Unauthorized", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(oauth_module.urllib.request, "urlopen", failing)
    with pytest.raises(oauth_module.OAuthError) as caught:
        oauth_module._http_json("https://example.test", data={"grant_type": "authorization_code"})
    assert "HTTP 401" in str(caught.value)
    assert "invalid_client" in str(caught.value)
    assert "AADSTS7000215" in str(caught.value)
