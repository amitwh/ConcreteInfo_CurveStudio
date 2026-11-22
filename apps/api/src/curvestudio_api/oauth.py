"""OAuth 2.0 authorization-code + PKCE sign-in for Google, Microsoft and LinkedIn.

Standard library only (``urllib.request``, ``secrets``, ``hashlib``): no runtime dependency is
added. A provider is enabled only when both its client id and secret are configured; nothing
about it is exposed otherwise.

Register the *exact* callback URI at every provider you enable::

    <CURVESTUDIO_APP_URL>/api/v1/auth/oauth/<provider>/callback

Providers compare it byte for byte, so a trailing slash or a different host is rejected. The web
app and the API must therefore be reached on the same origin (``CURVESTUDIO_APP_URL``), which is
how a single-process deployment serves them.

The browser flow is: ``GET .../start`` (state + PKCE, ``oauth_flow`` cookie) -> the provider ->
``GET .../callback`` (exchange the code, read the profile, issue a single-use exchange code) ->
``POST /auth/oauth/exchange`` from the web app. The provider access token never reaches the
browser.
"""

import base64
import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from curvestudio_api.settings import Settings

HTTP_TIMEOUT_S = 10.0
CALLBACK_PATH = "/api/v1/auth/oauth/{provider}/callback"


class OAuthError(Exception):
    """A provider failure. The callback turns these into a redirect, never a 500."""


class OAuthExchangeError(OAuthError):
    """The token endpoint refused, or returned no access token for, the authorization code."""


class OAuthIdentityError(OAuthError):
    """The provider profile did not identify a verified e-mail address."""


@dataclass(frozen=True)
class _Provider:
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: str


# Microsoft URLs carry the tenant; ``_endpoint`` substitutes it from settings.
_PROVIDERS: dict[str, _Provider] = {
    "google": _Provider(
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        scopes="openid email profile",
    ),
    "microsoft": _Provider(
        authorize_url="https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        userinfo_url="https://graph.microsoft.com/oidc/userinfo",
        scopes="openid email profile User.Read",
    ),
    "linkedin": _Provider(
        authorize_url="https://www.linkedin.com/oauth/v2/authorization",
        token_url="https://www.linkedin.com/oauth/v2/accessToken",
        userinfo_url="https://api.linkedin.com/v2/userinfo",
        scopes="openid profile email",
    ),
}


def _provider(name: str) -> _Provider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise OAuthError(f"unknown OAuth provider: {name}") from None


def _client_id(settings: Settings, provider: str) -> str:
    return {
        "google": settings.oauth_google_client_id,
        "microsoft": settings.oauth_microsoft_client_id,
        "linkedin": settings.oauth_linkedin_client_id,
    }[provider]


def _client_secret(settings: Settings, provider: str) -> str:
    secret = {
        "google": settings.oauth_google_client_secret,
        "microsoft": settings.oauth_microsoft_client_secret,
        "linkedin": settings.oauth_linkedin_client_secret,
    }[provider]
    return "" if secret is None else secret.get_secret_value()


def _endpoint(provider: str, template: str, settings: Settings) -> str:
    if provider == "microsoft":
        return template.format(tenant=settings.oauth_microsoft_tenant)
    return template


def configured_providers(settings: Settings) -> list[str]:
    """The provider names that have both a client id and a client secret, in a stable order."""
    return [
        name for name in _PROVIDERS if _client_id(settings, name) and _client_secret(settings, name)
    ]


def callback_url(settings: Settings, provider: str) -> str:
    """The redirect URI to register at ``provider``; derived from ``CURVESTUDIO_APP_URL``."""
    return settings.app_url.rstrip("/") + CALLBACK_PATH.format(provider=provider)


def new_state() -> str:
    """A fresh CSRF ``state`` value."""
    return secrets.token_urlsafe(32)


def new_pkce_pair() -> tuple[str, str]:
    """A fresh PKCE ``(verifier, S256 challenge)`` pair."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def build_authorize_url(
    provider: str,
    settings: Settings,
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    """The provider's authorization URL with the authorization-code + PKCE parameters."""
    config = _provider(provider)
    params = {
        "response_type": "code",
        "client_id": _client_id(settings, provider),
        "redirect_uri": redirect_uri,
        "scope": config.scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    base = _endpoint(provider, config.authorize_url, settings)
    return f"{base}?{urllib.parse.urlencode(params)}"


def exchange_code(
    provider: str,
    settings: Settings,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Redeem an authorization code for the provider's token response.

    Raises:
        OAuthExchangeError: on a non-2xx response, a network failure, or a response without
            ``access_token``.
    """
    config = _provider(provider)
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": _client_id(settings, provider),
        "client_secret": _client_secret(settings, provider),
        "code_verifier": code_verifier,
    }
    url = _endpoint(provider, config.token_url, settings)
    try:
        payload = _http_json(url, data=form)
    except OAuthError as exc:
        raise OAuthExchangeError("the provider rejected the token exchange") from exc
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise OAuthExchangeError("the provider returned no access token")
    return payload


def fetch_identity(provider: str, access_token: str) -> dict[str, Any]:
    """``{subject, email, email_verified, name}`` from the provider's userinfo endpoint.

    Google and LinkedIn must report ``email_verified: true``. Microsoft accepts an address it
    returns without that claim (it verifies addresses by policy), but still rejects an explicit
    ``false``.

    Raises:
        OAuthIdentityError: when no verified address can be read.
    """
    config = _provider(provider)
    try:
        data = _http_json(config.userinfo_url, headers={"Authorization": f"Bearer {access_token}"})
    except OAuthError as exc:
        raise OAuthIdentityError("could not read the provider profile") from exc
    return _identity(provider, data)


def _identity(provider: str, data: dict[str, Any]) -> dict[str, Any]:
    verified = data.get("email_verified")
    if provider == "microsoft":
        if verified is False:
            raise OAuthIdentityError("the provider has not verified this e-mail address")
    elif verified is not True:
        raise OAuthIdentityError("the provider has not verified this e-mail address")
    email = data.get("email")
    if provider == "microsoft" and (not isinstance(email, str) or not email.strip()):
        # Personal Microsoft accounts often omit ``email`` from the OIDC userinfo and carry
        # the address in ``preferred_username`` (or ``upn`` for work accounts).
        for claim in ("preferred_username", "upn"):
            candidate = data.get(claim)
            if isinstance(candidate, str) and "@" in candidate:
                email = candidate
                break
    if not isinstance(email, str) or not email.strip():
        raise OAuthIdentityError("the provider returned no e-mail address")
    subject = data.get("sub")
    return {
        "subject": subject if isinstance(subject, str) else "",
        "email": email.strip().lower(),
        "email_verified": True,
        "name": _name(data),
    }


def _name(data: dict[str, Any]) -> str:
    name = data.get("name")
    return name.strip()[:100] if isinstance(name, str) else ""


def _http_json(
    url: str,
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST ``data`` as a form (or GET when ``data`` is ``None``) and parse a JSON object.

    Tests monkeypatch this function; nothing else talks to the network.

    Raises:
        OAuthError: on a non-2xx response, a timeout or network failure, or a body that is not
            a JSON object.
    """
    body = None if data is None else urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(url, data=body)
    if body is not None:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    request.add_header("Accept", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # The provider names the reason in its JSON body (Microsoft: AADSTS codes). It never
        # contains a client secret; keep it short and attach it so the callback log is useful.
        detail = ""
        try:
            payload = json.loads(exc.read() or b"{}")
        except json.JSONDecodeError, OSError:
            payload = {}
        if isinstance(payload, dict):
            code = payload.get("error")
            description = payload.get("error_description")
            if isinstance(code, str) or isinstance(description, str):
                detail = f" ({str(code)[:40]}: {str(description)[:200]})"
        raise OAuthError(f"the provider returned HTTP {exc.code}{detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OAuthError("could not reach the provider") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OAuthError("the provider returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise OAuthError("the provider returned an unexpected JSON value")
    return payload
