"""Social sign-in: provider discovery, the PKCE redirect, the callback and the token exchange.

The browser never sees a provider token. ``GET .../{provider}/callback`` trades the provider
code for the profile and redirects with a single-use code that the web app claims once at
``POST /auth/oauth/exchange``.
"""

import logging
import secrets
from datetime import timedelta

from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from curvestudio_api import audit, oauth
from curvestudio_api.deps import Ctx, Db, client_ip
from curvestudio_api.errors import ApiError, not_found
from curvestudio_api.orm import OAuthCode, User
from curvestudio_api.routers.auth import _issue_tokens
from curvestudio_api.schemas import OAuthExchangeIn, OAuthProvidersOut, TokenOut
from curvestudio_api.security import hash_password, hash_token
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

router = APIRouter(prefix="/auth/oauth", tags=["auth"])
logger = logging.getLogger(__name__)

COOKIE_NAME = "oauth_flow"
COOKIE_PATH = "/api/v1/auth/oauth"
COOKIE_MAX_AGE_S = 600
EXCHANGE_TTL_S = 120
_BAD_CODE = "this sign-in link is invalid or has expired"


def _redirect(settings: Settings, path_and_query: str, *, clear_cookie: bool) -> RedirectResponse:
    """A 302 into the web app (never into the provider); clears the flow cookie when asked."""
    response = RedirectResponse(f"{settings.app_url.rstrip('/')}{path_and_query}", status_code=302)
    if clear_cookie:
        response.delete_cookie(COOKIE_NAME, path=COOKIE_PATH)
    return response


def _frontend_error(settings: Settings, reason: str) -> RedirectResponse:
    return _redirect(settings, f"/login?oauth_error={reason}", clear_cookie=True)


def _invalid_code() -> ApiError:
    return ApiError(400, "INVALID_TOKEN", _BAD_CODE)


@router.get("/providers", response_model=OAuthProvidersOut)
def providers(request: Request, ctx: Ctx) -> OAuthProvidersOut:
    """The providers a client may offer; configured ones only (public)."""
    ctx.ip_limiter.check(client_ip(request))
    return OAuthProvidersOut(providers=oauth.configured_providers(ctx.settings))


@router.get("/{provider}/start")
def start(provider: str, request: Request, ctx: Ctx) -> RedirectResponse:
    """Send the browser to the provider with a fresh state, PKCE verifier and flow cookie."""
    ctx.ip_limiter.check(client_ip(request))
    settings = ctx.settings
    if provider not in oauth.configured_providers(settings):
        raise not_found("oauth provider")
    state = oauth.new_state()
    verifier, challenge = oauth.new_pkce_pair()
    url = oauth.build_authorize_url(
        provider, settings, oauth.callback_url(settings, provider), state, challenge
    )
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        COOKIE_NAME,
        f"{provider}:{state}:{verifier}",
        max_age=COOKIE_MAX_AGE_S,
        httponly=True,
        secure=settings.app_url.startswith("https://"),
        samesite="lax",
        path=COOKIE_PATH,
    )
    return response


@router.get("/{provider}/callback")
def callback(
    provider: str,
    request: Request,
    ctx: Ctx,
    db: Db,
    code: str = "",
    state: str = "",
    error: str = "",
) -> RedirectResponse:
    """Validate the flow, sign the user in, and redirect with a single-use exchange code."""
    ctx.ip_limiter.check(client_ip(request))
    settings = ctx.settings
    if provider not in oauth.configured_providers(settings):
        return _frontend_error(settings, "provider")
    if error:
        logger.warning("oauth %s callback: provider error %r", provider, error[:120])
        return _frontend_error(settings, "provider")
    flow = request.cookies.get(COOKIE_NAME)
    if flow is None:
        logger.warning("oauth %s callback: no flow cookie (state=%r)", provider, state[:16])
        return _frontend_error(settings, "state")
    parts = flow.split(":")
    if len(parts) != 3:
        logger.warning("oauth %s callback: malformed flow cookie", provider)
        return _frontend_error(settings, "state")
    cookie_provider, cookie_state, verifier = parts
    if cookie_provider != provider or not code or not secrets.compare_digest(cookie_state, state):
        logger.warning(
            "oauth %s callback: state mismatch (cookie provider %r, code %s)",
            provider,
            cookie_provider,
            "present" if code else "missing",
        )
        return _frontend_error(settings, "state")
    try:
        token_payload = oauth.exchange_code(
            provider, settings, code, oauth.callback_url(settings, provider), verifier
        )
        identity = oauth.fetch_identity(provider, token_payload["access_token"])
    except oauth.OAuthError as exc:
        logger.warning("oauth %s callback failed: %s", provider, exc)
        return _frontend_error(settings, "provider")

    email = str(identity["email"]).strip().lower()
    now = utcnow()
    user = db.scalar(select(User).where(func.lower(User.email) == email))
    if user is None:
        name = str(identity.get("name") or "").strip()[:100] or email.split("@", 1)[0]
        new_user = User(
            email=email,
            password_hash=hash_password(secrets.token_urlsafe(32)),
            name=name,
            email_verified_at=now,
        )
        db.add(new_user)
        try:
            db.flush()
        except IntegrityError:
            # Another request created the same address first (the unique lower(email) index).
            db.rollback()
            user = db.scalar(select(User).where(func.lower(User.email) == email))
            if user is None:
                return _frontend_error(settings, "provider")
        else:
            user = new_user
    if user is None:
        return _frontend_error(settings, "provider")
    if not user.is_active:
        return _frontend_error(settings, "provider")
    if user.email_verified_at is None:
        # The provider vouched for the address (verification was checked in ``fetch_identity``).
        user.email_verified_at = now
    audit.record(
        db,
        "auth.oauth_login",
        "user",
        user_id=user.id,
        entity_id=user.id,
        detail={"provider": provider},
    )
    raw_code = secrets.token_urlsafe(32)
    db.add(
        OAuthCode(
            user_id=user.id,
            code_hash=hash_token(raw_code),
            expires_at=now + timedelta(seconds=EXCHANGE_TTL_S),
        )
    )
    db.commit()
    return _redirect(settings, f"/oauth/callback?code={raw_code}", clear_cookie=True)


@router.post("/exchange", response_model=TokenOut)
def exchange(
    body: OAuthExchangeIn, request: Request, response: Response, ctx: Ctx, db: Db
) -> TokenOut:
    """Claim a single-use exchange code and return an access/refresh pair."""
    ctx.ip_limiter.check(client_ip(request))
    response.headers["Cache-Control"] = "no-store"
    now = utcnow()
    code = db.scalar(select(OAuthCode).where(OAuthCode.code_hash == hash_token(body.code)))
    if code is None or code.used_at is not None or code.expires_at <= now:
        raise _invalid_code()
    claimed = db.execute(
        update(OAuthCode)
        .where(
            OAuthCode.id == code.id,
            OAuthCode.used_at.is_(None),
            OAuthCode.expires_at > now,
        )
        .values(used_at=now)
    ).rowcount  # type: ignore[attr-defined]
    if not claimed:
        raise _invalid_code()
    user = db.get(User, code.user_id)
    if user is None or not user.is_active:
        raise _invalid_code()
    tokens, _ = _issue_tokens(ctx, db, user)
    audit.record(db, "auth.oauth_exchange", "user", user_id=user.id, entity_id=user.id)
    db.commit()
    return tokens


__all__ = ["COOKIE_NAME", "router"]
