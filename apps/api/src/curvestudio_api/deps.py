"""Shared request dependencies: settings, database session and the authenticated user."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from curvestudio_api.db import SessionFactory
from curvestudio_api.errors import ApiError, not_found
from curvestudio_api.importer import ImportRunner
from curvestudio_api.jobs import JobManager
from curvestudio_api.logging_setup import get_logger
from curvestudio_api.orm import Autofit, Dataset, Fit, Job, User, UserDailyActivity
from curvestudio_api.proxy import client_address
from curvestudio_api.ratelimit import DailyCap, SlidingWindowLimiter
from curvestudio_api.security import JwtKeys, unauthorized, verify_access_token
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, utcnow

AUTH_ATTEMPTS_PER_MINUTE = 10
AUTH_PER_IP_PER_MINUTE = 60

_log = get_logger("deps")


@dataclass
class Context:
    """Everything the handlers share, built once by :func:`curvestudio_api.app.create_app`."""

    settings: Settings
    engine: Engine
    session_factory: SessionFactory
    keys: JwtKeys
    jobs: JobManager
    auth_limiter: SlidingWindowLimiter
    ip_limiter: SlidingWindowLimiter
    imports: ImportRunner
    failed_logins: DailyCap


def get_context(request: Request) -> Context:
    ctx: Context = request.app.state.ctx
    return ctx


Ctx = Annotated[Context, Depends(get_context)]
_bearer = HTTPBearer(auto_error=False, description="RS256 access token from /auth/login")


def get_db(ctx: Ctx) -> Iterator[Session]:
    with ctx.session_factory() as session:
        yield session


Db = Annotated[Session, Depends(get_db)]


def current_user(
    ctx: Ctx,
    db: Db,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    """The signed-in user, from a valid RS256 access token. Every protected handler uses it."""
    if credentials is None:
        raise unauthorized()
    user_id = verify_access_token(ctx.keys, credentials.credentials)
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise unauthorized("invalid or expired access token")
    _stamp_daily_activity(db, user)
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require_admin(user: CurrentUser) -> User:
    """The current user, when an administrator.

    The role comes from the row ``current_user`` just loaded from the database, so a demotion
    takes effect on the very next request.
    """
    if user.role != "admin":
        raise ApiError(403, "FORBIDDEN", "administrator access is required")
    return user


AdminUser = Annotated[User, Depends(require_admin)]


def _stamp_daily_activity(db: Session, user: User) -> None:
    """Advance the IST-day activity markers on the first authenticated request of a new day.

    Commits on its own: it runs before any handler writes (read-only handlers never commit,
    so the stamp would otherwise be rolled back when their session closes).
    """
    day = utcnow().astimezone(IST).date()
    if user.last_active_on == day:
        return
    user.last_active_on = day
    db.add(UserDailyActivity(user_id=user.id, day=day))
    try:
        db.commit()
    except IntegrityError:
        # A concurrent request of the same user on the same day inserted the row first; its
        # commit also advanced the marker, so nothing is lost.
        db.rollback()
        _log.debug("daily activity row already recorded for user %s", user.id)


def client_ip(request: Request) -> str:
    """The client address, from ``X-Forwarded-For`` as far as the trusted proxies vouch for it."""
    ctx: Context = request.app.state.ctx
    peer = request.client.host if request.client else None
    return client_address(
        request.headers.getlist("x-forwarded-for"), peer, ctx.settings.trusted_proxy_hops
    )


def own_dataset(db: Session, user: User, dataset_id: uuid.UUID) -> Dataset:
    """The user's dataset, or 404 (also for somebody else's)."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.user_id != user.id:
        raise not_found("dataset")
    return dataset


def own_fit(db: Session, user: User, fit_id: uuid.UUID) -> Fit:
    fit = db.get(Fit, fit_id)
    if fit is None or fit.user_id != user.id:
        raise not_found("fit")
    return fit


def own_job(db: Session, user: User, job_id: uuid.UUID) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise not_found("job")
    return job


def own_autofit(db: Session, user: User, autofit_id: uuid.UUID) -> Autofit:
    autofit = db.get(Autofit, autofit_id)
    if autofit is None or autofit.user_id != user.id:
        raise not_found("autofit")
    return autofit
