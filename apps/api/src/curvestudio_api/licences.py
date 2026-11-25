"""The AppLicense connector: activation, release and the daily re-check of entitlements.

The customer buys a licence in the AppLicense self-service portal and pastes the key into the
CurveStudio profile. The server calls the portal's verify API for this app's slug, using
``web:<account id>`` as the device identifier, verifies the returned JWT with the portal's
public key, stores an ``entitlements`` row and moves the account to the mapped plan through
``quotas.set_plan_override`` — the shipped plan resolver stays the only source of limits.

The portal's token carries no plan name: the plan is the licence plan's feature flags
(``features``, documented as e.g. ``["pro", "export", "api_access"]``), so the plan code is the
first feature that ``CURVESTUDIO_LICENCE_PLAN_MAP`` maps to a plan name here.

Nothing here blocks sign-in, and nothing deletes data. Every public function refuses with
``BILLING_OFF`` while the connector is unconfigured. Outbound calls time out after five seconds.
The licence key and the token are never written to an audit row or a log line.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from curvestudio_api import audit
from curvestudio_api.logging_setup import get_logger, log_fields
from curvestudio_api.orm import Entitlement, Plan, User
from curvestudio_api.quotas import set_plan_override
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

HTTP_TIMEOUT_S = 5.0
ALGORITHM = "RS256"
GRACE = timedelta(days=7)
"""How long an entitlement survives without a successful re-check, past the token's expiry."""
RECHECK_AFTER = timedelta(hours=24)
ACTIVATE_PATH = "/api/verify/activate"
CHECK_PATH = "/api/verify/check"
DEACTIVATE_PATH = "/api/verify/deactivate"
PUBLIC_KEY_PATH = "/api/verify/public-key"

ACTIVE = "active"
EXPIRED = "expired"
RELEASED = "released"

_log = get_logger("licences")


class LicenceError(Exception):
    """A refusal the routes turn into an error envelope; ``code`` is the machine-readable one."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PortalUnreachable(LicenceError):
    """The licence service could not be reached, or answered something that is not its API."""

    def __init__(self, message: str = "the licence service could not be reached") -> None:
        super().__init__("LICENCE_UNREACHABLE", message)


@dataclass(frozen=True)
class Claims:
    """The licence facts read from a verified token."""

    licence_id: str
    app_slug: str
    device_fingerprint: str
    features: tuple[str, ...]
    expires_at: datetime | None
    """The licence's own expiry; ``None`` for a lifetime licence."""
    expires_token_at: datetime
    """When the token itself expires (an hour after issue); the service renews it on a check."""
    grace_until: datetime
    """Seven days past the token's expiry (``exp + GRACE``)."""


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """POST ``payload`` as JSON (GET when it is ``None``) and parse a JSON object.

    Tests monkeypatch this function; nothing else talks to the network.

    Raises:
        PortalUnreachable: on a non-2xx response, a timeout or network failure, or a body that
            is not a JSON object.
    """
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise PortalUnreachable(f"the licence service returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PortalUnreachable("could not reach the licence service") from exc
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PortalUnreachable("the licence service returned invalid JSON") from exc
    if not isinstance(answer, dict):
        raise PortalUnreachable("the licence service returned an unexpected JSON value")
    return answer


_public_keys: dict[str, str] = {}
"""The portal's public key, fetched once per service URL and kept for the process's lifetime."""


def _normalise_pem(pem: str) -> str:
    return pem.replace("\\n", "\n").strip() + "\n"


def _public_key(settings: Settings) -> str:
    """The portal's RS256 public key: the setting, or ``GET /api/verify/public-key`` once."""
    if settings.licence_public_key_pem is not None:
        return _normalise_pem(settings.licence_public_key_pem.get_secret_value())
    url = settings.licence_url.rstrip("/")
    if url in _public_keys:
        return _public_keys[url]
    answer = _http_json(url + PUBLIC_KEY_PATH)
    pem = answer.get("public_key")
    if not isinstance(pem, str) or not pem.strip():
        raise PortalUnreachable("the licence service returned no public key")
    _public_keys[url] = _normalise_pem(pem)
    return _public_keys[url]


def _as_datetime(value: object, what: str) -> datetime | None:
    """An ISO 8601 claim as an aware UTC datetime; ``None`` when the claim is absent."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise LicenceError("LICENCE_INVALID", f"the licence token's {what} is not a date")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LicenceError("LICENCE_INVALID", f"the licence token's {what} is not a date") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def verify_token(token: str, settings: Settings, *, device: str) -> Claims:
    """Verify the portal's license JWT and read the licence facts from it.

    The signature is checked with the portal's public key (RS256). The token must name this
    app's slug and the device that activated it, and neither it nor the licence it describes may
    have expired.

    Raises:
        LicenceError: ``LICENCE_EXPIRED`` for an expired token or licence, ``LICENCE_INVALID``
            for a bad signature or another application's token, ``LICENCE_DEVICE`` for a token
            activated for another device, ``LICENCE_UNREACHABLE`` when the public key must be
            fetched and cannot be.
    """
    try:
        claims = jwt.decode(
            token,
            _public_key(settings),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "license_id", "app_slug"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise LicenceError("LICENCE_EXPIRED", "the licence token has expired") from exc
    except (jwt.PyJWTError, ValueError) as exc:
        raise LicenceError("LICENCE_INVALID", "the licence token is not valid") from exc
    if claims.get("app_slug") != settings.licence_app_id:
        raise LicenceError("LICENCE_INVALID", "the licence belongs to another application")
    if claims.get("device_fingerprint") != device:
        raise LicenceError("LICENCE_DEVICE", "the licence is activated for another device")
    licence_id = claims.get("license_id")
    if not isinstance(licence_id, str) or not licence_id:
        raise LicenceError("LICENCE_INVALID", "the licence token names no licence")
    expires_token_at = datetime.fromtimestamp(int(claims["exp"]), UTC)
    expires_at = _as_datetime(claims.get("expires_at"), "licence expiry")
    if expires_at is not None and expires_at <= utcnow():
        raise LicenceError("LICENCE_EXPIRED", "the licence has expired")
    features = claims.get("features")
    return Claims(
        licence_id=licence_id,
        app_slug=str(claims["app_slug"]),
        device_fingerprint=str(claims["device_fingerprint"]),
        features=tuple(feature for feature in features if isinstance(feature, str))
        if isinstance(features, list)
        else (),
        expires_at=expires_at,
        expires_token_at=expires_token_at,
        grace_until=expires_token_at + GRACE,
    )


def _require_billing(settings: Settings) -> None:
    if not (
        settings.billing_enabled
        and settings.licence_url.strip()
        and settings.licence_app_id.strip()
    ):
        raise LicenceError(
            "BILLING_OFF", "the licence connector is not configured on this installation"
        )


def _device(user: User) -> str:
    """How one account appears to the portal: one web account is one device."""
    return f"web:{user.id}"


def _base_url(settings: Settings) -> str:
    return settings.licence_url.rstrip("/")


def _mapped_plan(claims: Claims, settings: Settings) -> tuple[str, str]:
    """The portal plan code and this installation's plan name for it.

    Raises:
        LicenceError: ``LICENCE_UNKNOWN_PLAN`` when no feature of the licence is mapped.
    """
    mapping = settings.licence_plans
    for feature in claims.features:
        if feature in mapping:
            return feature, mapping[feature]
    codes = ", ".join(claims.features) or "none"
    raise LicenceError(
        "LICENCE_UNKNOWN_PLAN",
        f"the licence's plan ({codes}) has no CurveStudio plan in CURVESTUDIO_LICENCE_PLAN_MAP",
    )


def _row(db: Session, user: User) -> Entitlement | None:
    return db.scalar(select(Entitlement).where(Entitlement.user_id == user.id))


def activate(db: Session, settings: Settings, user: User, key: str) -> Entitlement:
    """Activate a licence key for ``user``: store the entitlement and move the account's plan.

    Raises:
        LicenceError: ``BILLING_OFF``, ``LICENCE_UNREACHABLE``, ``LICENCE_INVALID`` (the service
            refused the key or returned no usable token), ``LICENCE_EXPIRED``, ``LICENCE_DEVICE``
            or ``LICENCE_UNKNOWN_PLAN``.
    """
    _require_billing(settings)
    device = _device(user)
    answer = _http_json(
        _base_url(settings) + ACTIVATE_PATH,
        {
            "key": key,
            "device_fingerprint": device,
            "app_slug": settings.licence_app_id,
        },
    )
    if answer.get("valid") is not True:
        message = answer.get("message")
        raise LicenceError(
            "LICENCE_INVALID",
            message[:300]
            if isinstance(message, str) and message.strip()
            else "the licence service refused the key",
        )
    token = answer.get("license_jwt")
    if not isinstance(token, str) or not token:
        raise LicenceError("LICENCE_INVALID", "the licence service returned no licence token")
    claims = verify_token(token, settings, device=device)
    plan_code, plan_name = _mapped_plan(claims, settings)
    if db.scalar(select(Plan).where(Plan.name == plan_name)) is None:
        raise LicenceError(
            "LICENCE_UNKNOWN_PLAN",
            f"the portal plan code {plan_code!r} is mapped to the unknown plan {plan_name!r}",
        )
    row = _row(db, user)
    if row is None:
        row = Entitlement(user_id=user.id)
        db.add(row)
    row.licence_id = claims.licence_id
    row.plan_code = plan_code
    row.status = ACTIVE
    row.expires_at = claims.expires_at
    row.grace_until = claims.grace_until
    row.last_checked_at = utcnow()
    row.token = token
    set_plan_override(db, user, plan_name, actor=user)
    audit.record(
        db,
        "user.licence_activate",
        "user",
        user_id=user.id,
        entity_id=user.id,
        detail={
            "licence_id": claims.licence_id,
            "plan_code": plan_code,
            "plan": plan_name,
        },
    )
    db.commit()
    return row


def release(db: Session, settings: Settings, user: User) -> Entitlement | None:
    """Release ``user``'s licence: tell the portal (best effort), then return the account to free.

    An unreachable portal does not block the local release; the seat may stay counted at the
    portal until it is released there.

    Raises:
        LicenceError: ``BILLING_OFF``.
    """
    _require_billing(settings)
    row = _row(db, user)
    if row is None:
        return None
    if row.token:
        try:
            _http_json(
                _base_url(settings) + DEACTIVATE_PATH,
                {"device_fingerprint": _device(user), "app_slug": settings.licence_app_id},
                token=row.token,
            )
        except LicenceError as exc:
            _log.warning(
                "the licence service did not confirm the release",
                extra=log_fields(code=exc.code, licence_id=row.licence_id),
            )
    row.status = RELEASED
    set_plan_override(db, user, None, actor=user)
    audit.record(
        db,
        "user.licence_release",
        "user",
        user_id=user.id,
        entity_id=user.id,
        detail={"licence_id": row.licence_id, "plan_code": row.plan_code},
    )
    db.commit()
    return row


def status(db: Session, user: User, settings: Settings) -> Entitlement | None:
    """The account's entitlement row, or ``None`` when it has never activated a licence.

    Raises:
        LicenceError: ``BILLING_OFF``.
    """
    _require_billing(settings)
    return _row(db, user)


def _lapsed(row: Entitlement, now: datetime) -> bool:
    deadline = row.grace_until or row.expires_at
    return deadline is not None and now > deadline


def _expire(db: Session, row: Entitlement, user: User) -> None:
    """Lapse the entitlement and return the account to free, both audited."""
    row.status = EXPIRED
    set_plan_override(db, user, None, actor=None)
    audit.record(
        db,
        "user.licence_expired",
        "user",
        user_id=None,
        entity_id=user.id,
        detail={"licence_id": row.licence_id, "plan_code": row.plan_code},
    )


def recheck(db: Session, settings: Settings, now: datetime | None = None) -> int:
    """Re-check every active entitlement last checked more than a day ago.

    A service that says the licence is still valid renews the stored dates and token; a real
    refusal or an unreachable service before the grace deadline keeps the entitlement as it is,
    and after the deadline lapses it (the account returns to ``free``; nothing is deleted).

    Returns:
        How many entitlements were checked.

    Raises:
        LicenceError: ``BILLING_OFF``. A network failure is never raised.
    """
    _require_billing(settings)
    at = now or utcnow()
    due = at - RECHECK_AFTER
    rows = db.scalars(
        select(Entitlement).where(
            Entitlement.status == ACTIVE,
            or_(Entitlement.last_checked_at.is_(None), Entitlement.last_checked_at < due),
        )
    ).all()
    checked = 0
    for row in rows:
        user = db.get(User, row.user_id)
        if user is None:  # the cascade takes the row with the account; nothing to audit
            continue
        checked += 1
        try:
            answer = _http_json(
                _base_url(settings) + CHECK_PATH,
                {"app_slug": settings.licence_app_id},
                token=row.token,
            )
        except LicenceError as exc:
            if _lapsed(row, at):
                _expire(db, row, user)
                row.last_checked_at = at
            _log.warning(
                "the licence service could not re-check an entitlement",
                extra=log_fields(code=exc.code, licence_id=row.licence_id),
            )
            continue
        if answer.get("valid") is not True:
            if _lapsed(row, at):
                _expire(db, row, user)
            row.last_checked_at = at
            continue
        renewed = answer.get("license_jwt")
        if isinstance(renewed, str) and renewed:
            try:
                claims = verify_token(renewed, settings, device=_device(user))
            except LicenceError as exc:
                # The service said valid but its token does not verify; keep the old dates.
                _log.warning(
                    "the licence service returned a token that does not verify",
                    extra=log_fields(code=exc.code, licence_id=row.licence_id),
                )
                row.last_checked_at = at
                continue
            row.token = renewed
            row.expires_at = claims.expires_at
            row.grace_until = claims.grace_until
        row.last_checked_at = at
    db.commit()
    return checked
