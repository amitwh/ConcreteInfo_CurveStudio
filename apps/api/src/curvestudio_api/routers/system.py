"""Liveness and readiness probes, the signed-in user's profile and the model catalogue."""

from dataclasses import asdict
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Query, Request, Response
from sqlalchemy import delete, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from curvestudio.workflow import ModelInfo, WorkflowError, get_model, list_models
from curvestudio_api import __version__, audit, quotas
from curvestudio_api.deps import Ctx, CurrentUser, Db
from curvestudio_api.errors import ApiError
from curvestudio_api.logging_setup import get_logger
from curvestudio_api.migrations import current_revision_is_head
from curvestudio_api.orm import DEFAULT_PREFERENCES, AuditLog, Dataset, Fit, Job, RefreshToken, User
from curvestudio_api.routers.auth import _limit, audit_failed_password_check
from curvestudio_api.schemas import (
    DeleteMeIn,
    HealthOut,
    MeOut,
    MeUpdateIn,
    PasswordChangeIn,
    PlanOut,
    PreferencesOut,
    PreferencesUpdateIn,
    ReadyOut,
    RefreshIn,
    UsageOut,
    password_problem,
)
from curvestudio_api.security import hash_password, hash_token, verify_password
from curvestudio_api.timeutil import utcnow

router = APIRouter()
_log = get_logger("health")


def _database_up(ctx: Ctx) -> bool:
    try:
        with ctx.session_factory() as db:
            db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        _log.exception("database check failed")
        return False
    return True


@router.get("/health", response_model=HealthOut, tags=["system"])
def health(ctx: Ctx) -> HealthOut:
    """Liveness plus a real ``SELECT 1``; 503 when the database cannot be reached."""
    if not _database_up(ctx):
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the database is not reachable")
    return HealthOut(status="ok", database="ok", version=__version__)


@router.get("/ready", response_model=ReadyOut, tags=["system"])
def ready(ctx: Ctx) -> ReadyOut:
    """Ready to serve: database reachable, schema at the newest migration, jobs running."""
    if not _database_up(ctx):
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the database is not reachable")
    if not current_revision_is_head(ctx.engine):
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the database schema is not up to date")
    if not ctx.jobs.is_running():
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the job scheduler is not running")
    return ReadyOut(status="ready")


def _preferences(user: User) -> dict[str, Any]:
    """The stored preferences over the defaults: a NULL column, or a key the row lacks, falls
    back to the default for that key."""
    return {**DEFAULT_PREFERENCES, **(user.preferences or {})}


def _me_out(db: Session, ctx: Ctx, user: User) -> MeOut:
    """The signed-in user, the resolved plan's limits and what the user has used so far."""
    plan = quotas.resolve_plan(db, ctx.settings, user.plan_override)
    limits = quotas.effective_limits(db, ctx.settings, user.plan_override)
    used = quotas.usage(db, user.id)
    preferences = _preferences(user)
    return MeOut(
        id=user.id,
        email=user.email,
        name=user.name,
        created_at=user.created_at,
        email_verified=user.email_verified_at is not None,
        role=cast(Literal["member", "admin"], user.role),
        preferences=PreferencesOut(
            error_reports=preferences["error_reports"],
            report_byline=preferences["report_byline"],
        ),
        last_login_at=user.last_login_at,
        plan=PlanOut(
            name=plan.name,
            **asdict(limits),
            max_autofits_per_day=quotas.autofits_per_day_limit(
                db, ctx.settings, user.plan_override
            ),
        ),
        usage=UsageOut(**asdict(used), autofits_today=quotas.autofit_runs_today(db, user.id)),
    )


def _body_error(field: str, message: str) -> ApiError:
    """A 422 for a rule the body schema cannot check on its own because it needs the account."""
    return ApiError(
        422,
        "VALIDATION_ERROR",
        "the request is not valid",
        {"errors": [{"loc": ["body", field], "msg": message, "type": "value_error"}]},
    )


def _revoke_other_sessions(db: Session, user: User, keep_refresh_token: str) -> int:
    """Sign out every session of the account except the one holding ``keep_refresh_token``;
    returns how many refresh tokens were revoked. A token that is not one of the account's
    live ones keeps nothing. Access tokens already issued run until they expire."""
    revoked = db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.user_id == user.id,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.token_hash != hash_token(keep_refresh_token),
        )
        .values(revoked_at=utcnow())
    )
    return int(revoked.rowcount)  # type: ignore[attr-defined]


@router.get("/me", response_model=MeOut, tags=["auth"])
def me(user: CurrentUser, ctx: Ctx, db: Db) -> MeOut:
    """The signed-in user: profile, preferences, the resolved plan's limits and usage so far."""
    return _me_out(db, ctx, user)


@router.patch("/me", response_model=MeOut, tags=["auth"])
def update_me(body: MeUpdateIn, user: CurrentUser, ctx: Ctx, db: Db) -> MeOut:
    """Change the displayed name. The name the account already has writes nothing."""
    if body.name != user.name:
        user.name = body.name
        # A name is personal data: the trail records that it changed, never what it became.
        audit.record(db, "user.name_change", "user", user_id=user.id, entity_id=user.id)
        db.commit()
    return _me_out(db, ctx, user)


@router.put("/me/preferences", response_model=MeOut, tags=["auth"])
def update_preferences(body: PreferencesUpdateIn, user: CurrentUser, ctx: Ctx, db: Db) -> MeOut:
    """Change one or both preferences; the ones not named keep their value."""
    stored = _preferences(user)
    changed = {
        key: value
        for key, value in body.model_dump(exclude_none=True).items()
        if stored.get(key) != value
    }
    if changed:
        # A new object, never an in-place edit: a change inside a JSON column is not tracked.
        user.preferences = {**stored, **changed}
        audit.record(
            db,
            "user.preferences_change",
            "user",
            user_id=user.id,
            entity_id=user.id,
            detail=changed,
        )
        db.commit()
    return _me_out(db, ctx, user)


@router.post("/me/password", response_model=MeOut, tags=["auth"])
def change_password(
    body: PasswordChangeIn, request: Request, user: CurrentUser, ctx: Ctx, db: Db
) -> MeOut:
    """Set a new password and sign out every other session.

    The current password must be given. The session whose refresh token is presented is the
    one that stays signed in. A wrong current password is refused and audited under the same
    daily cap as failed sign-ins.
    """
    ip = _limit(ctx, request, user.email)
    current = body.current_password.get_secret_value()
    new = body.new_password.get_secret_value()
    if not verify_password(user.password_hash, current):
        audit_failed_password_check(ctx, db, user, ip)
        # Committed on its own so the row survives the error response; nothing else is pending.
        db.commit()
        raise ApiError(403, "FORBIDDEN", "the current password is not correct")
    if new == current:
        raise _body_error("new_password", "the new password must be different from the current one")
    problem = password_problem(body.new_password, user.email)
    if problem is not None:
        raise _body_error("new_password", problem)
    user.password_hash = hash_password(new)
    _revoke_other_sessions(db, user, body.refresh_token)
    audit.record(db, "user.password_change", "user", user_id=user.id, entity_id=user.id)
    db.commit()
    return _me_out(db, ctx, user)


@router.post("/me/sessions/revoke-all", response_model=MeOut, tags=["auth"])
def revoke_all_sessions(body: RefreshIn, user: CurrentUser, ctx: Ctx, db: Db) -> MeOut:
    """Sign out every other session; the one holding ``refresh_token`` stays signed in."""
    revoked = _revoke_other_sessions(db, user, body.refresh_token)
    audit.record(
        db,
        "user.sessions_revoked",
        "user",
        user_id=user.id,
        entity_id=user.id,
        detail={"revoked": revoked},
    )
    db.commit()
    return _me_out(db, ctx, user)


@router.delete("/me", status_code=204, tags=["auth"])
def delete_me(body: DeleteMeIn, user: CurrentUser, ctx: Ctx, db: Db) -> Response:
    """Delete the account and everything in it.

    The body must carry the part of the e-mail address before the ``@``; without it, or with
    the wrong words, nothing is deleted, revoked or cancelled. Datasets, fits, jobs and refresh
    tokens are removed (running jobs are stopped first). The audit trail is kept but
    anonymised: ``audit_logs.user_id`` is set to NULL, the one change the append-only trigger
    allows. A final ``user.delete`` row records that it happened.
    """
    expected = user.email.partition("@")[0].lower()
    if body.confirm.strip().lower() != expected:
        raise _body_error(
            "confirm", "this does not match the part of your e-mail address before the @"
        )
    ctx.jobs.cancel_for_user(user.id)
    user_id = user.id
    for table in (Job, Fit, Dataset, RefreshToken):
        db.execute(delete(table).where(table.user_id == user_id))
    db.execute(update(AuditLog).where(AuditLog.user_id == user_id).values(user_id=None))
    db.delete(user)
    db.flush()
    audit.record(db, "user.delete", "user", user_id=None, entity_id=user_id)
    db.commit()
    return Response(status_code=204)


@router.get("/models", response_model=list[ModelInfo], tags=["models"])
def models(
    category: Annotated[str | None, Query(max_length=100)] = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
) -> list[ModelInfo]:
    """The model catalogue, optionally filtered by category and a search text."""
    found = list_models()
    if category:
        wanted = category.strip().lower()
        found = [m for m in found if m.category.lower() == wanted]
    if q and q.strip():
        needle = q.strip().lower()
        found = [
            m for m in found if needle in " ".join((m.id, m.name, m.category, m.equation)).lower()
        ]
    return found


@router.get("/models/{model_id}", response_model=ModelInfo, tags=["models"])
def model(model_id: str) -> ModelInfo:
    try:
        return get_model(model_id)
    except WorkflowError as exc:
        raise ApiError(404, "NOT_FOUND", exc.message, {"code": exc.code}) from exc
