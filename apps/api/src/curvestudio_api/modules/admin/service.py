"""The rules behind the admin API: what an administrator may list and change, and what every
change leaves in the audit trail. Nothing here knows about HTTP beyond the errors it raises.

Error events stay anonymous: no function here attaches an account to one. The audit row of a
status change names the administrator who made it, never anybody who met the error.
"""

import json
import re
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

from sqlalchemy import Engine, and_, case, delete, false, func, or_, select, text, true, update
from sqlalchemy.orm import Session

from curvestudio_api import __version__, audit, licences, quotas
from curvestudio_api.deps import Context
from curvestudio_api.errors import ApiError, not_found
from curvestudio_api.migrations import current_revision
from curvestudio_api.modules.admin.models import RequestStatDaily
from curvestudio_api.modules.admin.requeststats import BUCKET_COUNT, capped_percentiles
from curvestudio_api.modules.admin.schemas import (
    DEFAULT_PAGE_SIZE,
    AccountCountsOut,
    ActiveUsersOut,
    AdminLicenceOut,
    AdminUserDetailOut,
    AdminUserOut,
    AdminUserUpdateIn,
    AuditOut,
    CountOut,
    DatabaseOut,
    ErrorEventDetailOut,
    ErrorEventOut,
    ErrorStatus,
    JobsOut,
    OverviewOut,
    PageOut,
    PlanAdminOut,
    PlanLimitsIn,
    PlanUpdateIn,
    Role,
    RouteLatencyOut,
    SystemOut,
)
from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.modules.errors.service import STATUS_NEW
from curvestudio_api.modules.installs import service as installs
from curvestudio_api.modules.installs.schemas import InstallSummaryOut
from curvestudio_api.orm import (
    AuditLog,
    Dataset,
    Fit,
    Job,
    Plan,
    RefreshToken,
    User,
    UserDailyActivity,
)
from curvestudio_api.quotas import FREE_PLAN_NAME, PLAN_NAME
from curvestudio_api.schemas import LicenceStatus, UsageOut
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, to_ist_iso, utcnow

AUDIT_ROWS_PER_ACCOUNT = 50
SERIES_DAYS = 90
ACTIVE_WINDOWS = {"d1": 1, "d7": 7, "d30": 30}
"""The active-user windows, in days; each one ends with today."""
NEW_ACCOUNT_DAYS = 30
AUTOFIT_RUNS_HOURS = 24
ONE_DAY = timedelta(days=1)
SELF_CHANGE_REFUSED = "SELF_CHANGE_REFUSED"
_BACKTICKS = re.compile(r"`+")
"""A run of backticks, to size a code fence in the error bundle."""


def _query_error(field: str, message: str) -> ApiError:
    """A 422 for a query parameter rule its type cannot check on its own."""
    return ApiError(
        422,
        "VALIDATION_ERROR",
        "the request is not valid",
        {"errors": [{"loc": ["query", field], "msg": message, "type": "value_error"}]},
    )


def _total(db: Session, model: Any, *where: Any) -> int:
    """How many rows of ``model`` satisfy ``where``."""
    return int(db.scalar(select(func.count()).select_from(model).where(*where)) or 0)


def _ist_day_start(day: date) -> datetime:
    """The UTC instant at which the IST calendar day ``day`` begins."""
    return datetime.combine(day, time.min, tzinfo=IST).astimezone(UTC)


# -- accounts --------------------------------------------------------------------------------------


def erase_account(
    ctx: Context, db: Session, user_id: uuid.UUID, *, actor_id: uuid.UUID | None
) -> None:
    """Delete the account and everything in it. The audit trail stays, anonymised.

    Running jobs are stopped first. Datasets, fits, jobs and refresh tokens go; the account's
    ``audit_logs`` rows lose their ``user_id``, the one change the append-only trigger allows.
    A final ``user.delete`` row records it under ``actor_id`` (an administrator) or under
    nobody (the account itself); an administrator erasing their own account through the admin
    route is the second case, since the row cannot name a user who is being deleted.

    Flushes; the caller commits. Raises 404 when there is no such account.
    """
    user = db.get(User, user_id)
    if user is None:
        raise not_found("user")
    ctx.jobs.cancel_for_user(user_id)
    for table in (Job, Fit, Dataset, RefreshToken):
        db.execute(delete(table).where(table.user_id == user_id))
    db.execute(update(AuditLog).where(AuditLog.user_id == user_id).values(user_id=None))
    db.delete(user)
    db.flush()
    audit.record(
        db,
        "user.delete",
        "user",
        user_id=None if actor_id == user_id else actor_id,
        entity_id=user_id,
    )


def delete_account(ctx: Context, db: Session, actor: User, user_id: uuid.UUID) -> None:
    """Erase the account ``user_id`` on ``actor``'s request. Flushes; the caller commits."""
    erase_account(ctx, db, user_id, actor_id=actor.id)


def get_user(db: Session, user_id: uuid.UUID) -> User:
    """The account, or 404."""
    user = db.get(User, user_id)
    if user is None:
        raise not_found("user")
    return user


def _usage_out(db: Session, user_id: uuid.UUID) -> UsageOut:
    return UsageOut(
        **asdict(quotas.usage(db, user_id)),
        autofits_today=quotas.autofit_runs_today(db, user_id),
    )


def _licence_out(db: Session, settings: Settings, user: User) -> AdminLicenceOut | None:
    """The account's licence entitlement for the console, when it holds one; an
    unconfigured connector is no entitlement and no error."""
    try:
        row = licences.status(db, user, settings)
    except licences.LicenceError:
        return None
    if row is None:
        return None
    return AdminLicenceOut(
        status=cast(LicenceStatus, row.status),
        plan_code=row.plan_code,
        expires_at=row.expires_at,
        last_checked_at=row.last_checked_at,
    )


def admin_user_out(db: Session, settings: Settings, user: User) -> AdminUserOut:
    """The account as the console lists it, with its resolved plan, usage and licence."""
    return AdminUserOut(
        id=user.id,
        email=user.email,
        name=user.name,
        role=cast(Role, user.role),
        plan=quotas.resolve_plan(db, settings, user.plan_override).name,
        plan_override=user.plan_override,
        is_active=user.is_active,
        email_verified=user.email_verified_at is not None,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        last_active_on=user.last_active_on,
        usage=_usage_out(db, user.id),
        licence=_licence_out(db, settings, user),
    )


def list_users(
    db: Session,
    settings: Settings,
    *,
    q: str | None = None,
    plan: str | None = None,
    role: Role | None = None,
    active: bool | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> PageOut[AdminUserOut]:
    """The accounts matching the filters, newest first, one page of them.

    ``q`` is a case-insensitive substring of the name or the address. ``plan`` matches the
    plan each account resolves to (and must name a plan row); ``role`` and ``active`` match
    the columns.
    """
    conditions: list[Any] = []
    if q and q.strip():
        needle = q.strip().lower()
        conditions.append(
            or_(
                func.lower(User.email).contains(needle, autoescape=True),
                func.lower(User.name).contains(needle, autoescape=True),
            )
        )
    if plan is not None:
        if db.scalar(select(Plan.id).where(Plan.name == plan)) is None:
            raise _query_error("plan", f"there is no plan named {plan!r}")
        conditions.append(_resolved_plan_is(settings, plan))
    if role is not None:
        conditions.append(User.role == role)
    if active is not None:
        conditions.append(User.is_active == active)
    users = db.scalars(
        select(User)
        .where(*conditions)
        .order_by(User.created_at.desc(), User.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return PageOut[AdminUserOut](
        items=[admin_user_out(db, settings, user) for user in users],
        total=_total(db, User, *conditions),
        page=page,
        page_size=page_size,
    )


def _resolved_plan_is(settings: Settings, plan: str) -> Any:
    """The condition for accounts whose resolved plan is ``plan`` (``quotas.resolve_plan``)."""
    if not settings.billing_enabled:
        # Every account is self-hosted, whatever override it carries.
        return true() if plan == PLAN_NAME else false()
    if plan != FREE_PLAN_NAME:
        return User.plan_override == plan
    # ``free`` is what an account with no override, or with one naming no row, resolves to.
    return or_(
        User.plan_override.is_(None),
        User.plan_override == plan,
        User.plan_override.not_in(select(Plan.name)),
    )


def user_detail(db: Session, settings: Settings, user: User) -> AdminUserDetailOut:
    """The account and the last fifty rows of its audit trail, newest first."""
    rows = db.scalars(
        select(AuditLog)
        .where(AuditLog.user_id == user.id)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(AUDIT_ROWS_PER_ACCOUNT)
    )
    return AdminUserDetailOut(
        user=admin_user_out(db, settings, user), audit=[_audit_out(row) for row in rows]
    )


def _audit_out(row: AuditLog) -> AuditOut:
    return AuditOut(
        id=row.id,
        user_id=row.user_id,
        action=row.action,
        entity=row.entity,
        entity_id=row.entity_id,
        detail=dict(row.detail or {}),
        request_id=row.request_id,
        created_at=row.created_at,
    )


def update_user(
    db: Session,
    settings: Settings,
    actor: User,
    user: User,
    body: AdminUserUpdateIn,
) -> AdminUserOut:
    """Apply ``body`` to ``user`` on ``actor``'s request; each change is audited under the
    administrator. Nothing at all is written when an administrator names their own ``role``
    or ``is_active``: that would let the last administrator lock everybody out.

    Flushes; the caller commits.
    """
    if user.id == actor.id and {"role", "is_active"} & body.model_fields_set:
        raise ApiError(
            422,
            "VALIDATION_ERROR",
            "an administrator cannot change their own role or active flag",
            {"code": SELF_CHANGE_REFUSED},
        )
    if body.role is not None:
        set_role(db, actor, user, body.role)
    if "plan_override" in body.model_fields_set:
        set_plan(db, actor, user, body.plan_override)
    if body.is_active is not None:
        set_active(db, actor, user, body.is_active)
    return admin_user_out(db, settings, user)


def set_role(db: Session, actor: User, user: User, role: Role) -> None:
    """Make ``user`` a ``role``; the role it already has writes nothing."""
    if user.role == role:
        return
    audit.record(
        db,
        "user.role_change",
        "user",
        user_id=actor.id,
        entity_id=user.id,
        detail={"from": user.role, "to": role},
    )
    user.role = role


def set_active(db: Session, actor: User, user: User, is_active: bool) -> None:
    """Activate or deactivate ``user``; the state it already has writes nothing. A deactivated
    account is refused on its next request, tokens or not."""
    if user.is_active == is_active:
        return
    audit.record(
        db,
        "user.activate" if is_active else "user.deactivate",
        "user",
        user_id=actor.id,
        entity_id=user.id,
    )
    user.is_active = is_active


def set_plan(db: Session, actor: User, user: User, override: str | None) -> None:
    """Set (``override``) or clear (``None``) the account's plan override. An unknown plan
    name is a 422 from ``quotas.set_plan_override``, which writes the ``user.plan_change``
    row; the override the account already has writes nothing."""
    if user.plan_override == override:
        return
    quotas.set_plan_override(db, user, override, actor)


# -- error events ----------------------------------------------------------------------------------


def get_event(db: Session, event_id: uuid.UUID) -> ErrorEvent:
    """The error event, or 404."""
    event = db.get(ErrorEvent, event_id)
    if event is None:
        raise not_found("error event")
    return event


def _event_out(event: ErrorEvent) -> ErrorEventOut:
    return ErrorEventOut(
        id=event.id,
        source=event.source,
        category=event.category,
        severity=event.severity,
        error_type=event.error_type,
        message=event.message,
        fingerprint=event.fingerprint,
        route=event.route,
        method=event.method,
        status_code=event.status_code,
        request_id=event.request_id,
        app_version=event.app_version,
        status=cast(ErrorStatus, event.status),
        occurrence_count=event.occurrence_count,
        first_seen_at=event.first_seen_at,
        last_seen_at=event.last_seen_at,
    )


def list_errors(
    db: Session,
    *,
    status: ErrorStatus | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> PageOut[ErrorEventOut]:
    """The error events matching the filters, newest first (by ``last_seen_at``), one page."""
    conditions: list[Any] = []
    if status is not None:
        conditions.append(ErrorEvent.status == status)
    if category is not None:
        conditions.append(ErrorEvent.category == category)
    events = db.scalars(
        select(ErrorEvent)
        .where(*conditions)
        .order_by(ErrorEvent.last_seen_at.desc(), ErrorEvent.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return PageOut[ErrorEventOut](
        items=[_event_out(event) for event in events],
        total=_total(db, ErrorEvent, *conditions),
        page=page,
        page_size=page_size,
    )


def event_detail(db: Session, event_id: uuid.UUID) -> ErrorEventDetailOut:
    """One error event with everything stored about it."""
    event = get_event(db, event_id)
    return ErrorEventDetailOut(
        **_event_out(event).model_dump(),
        stack=event.stack,
        context_json=None if event.context_json is None else dict(event.context_json),
        console_log=event.console_log,
        user_agent=event.user_agent,
    )


def set_error_status(
    db: Session, actor: User, event: ErrorEvent, status: ErrorStatus
) -> ErrorEventOut:
    """Give ``event`` its new ``status`` and return it as the console lists it; the status it
    already has writes nothing. A repeat of a ``resolved`` or ``ignored`` event reopens it
    (``modules.errors.service``). Flushes; the caller commits."""
    if event.status != status:
        audit.record(
            db,
            "error.status_change",
            "error_event",
            user_id=actor.id,
            entity_id=event.id,
            detail={"from": event.status, "to": status},
        )
        event.status = status
    return _event_out(event)


def _fenced(text: str | None, language: str) -> str:
    """``text`` in a code block whose fence is longer than any run of backticks inside it."""
    if text is None:
        return "(none)"
    longest = max((len(run) for run in _BACKTICKS.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _cell(value: Any) -> str:
    """A table cell: ``(none)`` for a missing value; pipes and line breaks escaped."""
    if value is None:
        return "(none)"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def error_bundle_markdown(db: Session, event_id: uuid.UUID) -> str:
    """The error event as a Markdown document: every field in a table, then the stack in a code
    block, the context as JSON and the console lines in a code block. 404 for an unknown event."""
    event = get_event(db, event_id)
    fields = [
        ("Id", str(event.id)),
        ("Source", event.source),
        ("Category", event.category),
        ("Severity", event.severity),
        ("Type", event.error_type),
        ("Status", event.status),
        ("Occurrences", event.occurrence_count),
        ("First seen", to_ist_iso(event.first_seen_at)),
        ("Last seen", to_ist_iso(event.last_seen_at)),
        ("Route", event.route),
        ("Method", event.method),
        ("Status code", event.status_code),
        ("Request ID", event.request_id),
        ("App version", event.app_version),
        ("User agent", event.user_agent),
        ("Fingerprint", event.fingerprint),
        ("Message", event.message),
    ]
    context = (
        None
        if event.context_json is None
        else json.dumps(event.context_json, indent=2, ensure_ascii=False, sort_keys=True)
    )
    lines = ["# Error event", "", "| Field | Value |", "| --- | --- |"]
    lines += [f"| {name} | {_cell(value)} |" for name, value in fields]
    for title, block in (
        ("Stack", _fenced(event.stack, "text")),
        ("Context", _fenced(context, "json")),
        ("Console log", _fenced(event.console_log, "text")),
    ):
        lines += ["", f"## {title}", "", block]
    lines.append("")
    return "\n".join(lines)


# -- the audit trail -------------------------------------------------------------------------------


def list_audit(
    db: Session,
    *,
    user_id: uuid.UUID | None = None,
    action: str | None = None,
    from_day: date | None = None,
    to_day: date | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> PageOut[AuditOut]:
    """The audit rows matching the filters, newest first, one page of them.

    ``from_day`` and ``to_day`` are IST calendar days, both inclusive; each may come alone.
    """
    if from_day is not None and to_day is not None and from_day > to_day:
        raise _query_error("to", '"to" must not be before "from"')
    conditions: list[Any] = []
    if user_id is not None:
        conditions.append(AuditLog.user_id == user_id)
    if action is not None:
        conditions.append(AuditLog.action == action)
    if from_day is not None:
        conditions.append(AuditLog.created_at >= _ist_day_start(from_day))
    if to_day is not None:
        conditions.append(AuditLog.created_at < _ist_day_start(to_day) + ONE_DAY)
    rows = db.scalars(
        select(AuditLog)
        .where(*conditions)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return PageOut[AuditOut](
        items=[_audit_out(row) for row in rows],
        total=_total(db, AuditLog, *conditions),
        page=page,
        page_size=page_size,
    )


# -- the overview ----------------------------------------------------------------------------------


def overview(db: Session, settings: Settings, *, now: datetime | None = None) -> OverviewOut:
    """The figures of the console's front page as of ``now``.

    Active users are the accounts whose last authenticated request fell within the window,
    by IST calendar day, today counting as the first day. The series covers the last ninety
    days ending today: sign-ups by the IST day of ``created_at``, and accounts with an
    activity marker on each day. The install figures come from ``modules.installs``.
    """
    at = now or utcnow()
    today = at.astimezone(IST).date()
    return OverviewOut(
        accounts=AccountCountsOut(
            total=_total(db, User),
            verified=_total(db, User, User.email_verified_at.is_not(None)),
            new_30d=_total(db, User, User.created_at >= at - timedelta(days=NEW_ACCOUNT_DAYS)),
        ),
        active_users=ActiveUsersOut(
            **{
                name: _total(db, User, User.last_active_on >= today - timedelta(days=window - 1))
                for name, window in ACTIVE_WINDOWS.items()
            }
        ),
        installs=install_figures(db, settings, now=at),
        open_errors=_total(db, ErrorEvent, ErrorEvent.status == STATUS_NEW),
        autofit_runs_24h=_total(
            db,
            AuditLog,
            AuditLog.action == "autofit.submit",
            AuditLog.created_at >= at - timedelta(hours=AUTOFIT_RUNS_HOURS),
        ),
        series=_series(db, today),
    )


def _series(db: Session, today: date) -> list[CountOut]:
    """One row per day of the last :data:`SERIES_DAYS` ending ``today``, oldest first; a day
    with nothing is a zero.

    IST keeps one offset all year, so each of its days is the same 24 hours in UTC: the bounds
    are worked out here and the database does the counting.
    """
    days = [today - timedelta(days=back) for back in range(SERIES_DAYS - 1, -1, -1)]
    created = User.created_at
    day_of = case(
        *[
            (and_(created >= _ist_day_start(day), created < _ist_day_start(day + ONE_DAY)), index)
            for index, day in enumerate(days)
        ],
        else_=None,
    )
    signups = [0] * SERIES_DAYS
    counted = db.execute(
        select(day_of, func.count()).where(created >= _ist_day_start(days[0])).group_by(day_of)
    )
    for index, count in counted:
        if index is not None:  # None: created after the last day (a clock that moved)
            signups[index] = count
    active = [0] * SERIES_DAYS
    index_of = {day: index for index, day in enumerate(days)}
    marked = db.execute(
        select(UserDailyActivity.day, func.count())
        .where(UserDailyActivity.day >= days[0])
        .group_by(UserDailyActivity.day)
    )
    for day, count in marked:
        if day in index_of:
            active[index_of[day]] = count
    return [
        CountOut(day=day, signups=signups[index], active=active[index])
        for index, day in enumerate(days)
    ]


def install_figures(
    db: Session, settings: Settings, *, now: datetime | None = None
) -> InstallSummaryOut:
    """The install figures of ``modules.installs``, for the release this deployment counts as
    the latest (the API's own version when none is set)."""
    return installs.install_summary(db, current_release=settings.current_release, now=now)


# -- the system report -----------------------------------------------------------------------------


def system(ctx: Context) -> SystemOut:
    """The API's version, release, uptime, job scheduler and database."""
    at = utcnow()
    jobs = ctx.jobs
    revision = current_revision(ctx.engine)
    return SystemOut(
        version=__version__,
        current_release=ctx.settings.current_release or __version__,
        started_at=ctx.started_at,
        uptime_s=(at - ctx.started_at).total_seconds(),
        jobs=JobsOut(
            running=jobs.is_running(),
            workers=jobs.worker_count(),
            queue_depth=jobs.queue_size(),
            queue_max=jobs.queue_max(),
        ),
        database=DatabaseOut(size_bytes=_database_size(ctx.engine), migration_head=revision or ""),
    )


def route_latency(db: Session, *, days: int, now: datetime | None = None) -> list[RouteLatencyOut]:
    """Each route template over the last ``days`` IST days, today included, most requests first.

    The days' histograms are summed and the percentiles are read off that sum: an average of
    daily percentiles is not a percentile (one slow day and one fast day do not make a medium
    one). The slowest request is the largest the days recorded, and neither percentile is
    allowed above it. A route with no request in the window is left out.
    """
    first = (now or utcnow()).astimezone(IST).date() - timedelta(days=days - 1)
    rows = db.execute(
        select(
            RequestStatDaily.route_template,
            RequestStatDaily.count,
            RequestStatDaily.errors,
            RequestStatDaily.histogram,
            RequestStatDaily.max_ms,
        ).where(RequestStatDaily.day >= first)
    )
    histograms: dict[str, list[int]] = {}
    totals: dict[str, list[int]] = {}
    for route, count, errors, histogram, slowest in rows:
        summed = histograms.setdefault(route, [0] * BUCKET_COUNT)
        for bucket, seen in enumerate(histogram):
            summed[bucket] += seen
        running = totals.setdefault(route, [0, 0, 0])
        running[0] += count
        running[1] += errors
        running[2] = max(running[2], slowest)

    def _out(route: str) -> RouteLatencyOut:
        count, errors, slowest = totals[route]
        p50_ms, p95_ms = capped_percentiles(histograms[route], slowest)
        return RouteLatencyOut(
            route_template=route,
            count=count,
            errors=errors,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            max_ms=slowest,
        )

    return [_out(route) for route in sorted(totals, key=lambda route: (-totals[route][0], route))]


def _database_size(engine: Engine) -> int | None:
    """The size of the database in bytes on PostgreSQL; ``None`` anywhere else."""
    if engine.dialect.name != "postgresql":
        return None
    with engine.connect() as connection:
        return int(connection.scalar(text("SELECT pg_database_size(current_database())")) or 0)


# -- plans -----------------------------------------------------------------------------------------


def _limits_of(plan: Plan) -> PlanLimitsIn:
    """The plan's four limits. A stored value that is missing or unusable is the built-in
    default, which is what ``quotas`` applies."""
    stored = dict(plan.limits or {})

    def value(key: str, default: int) -> int:
        found = stored.get(key)
        return found if isinstance(found, int) and found > 0 else default

    return PlanLimitsIn(
        max_datasets=value("max_datasets", quotas.DEFAULT_LIMITS["max_datasets"]),
        max_dataset_bytes=value("max_dataset_bytes", quotas.DEFAULT_LIMITS["max_dataset_bytes"]),
        max_fits=value("max_fits", quotas.DEFAULT_LIMITS["max_fits"]),
        max_autofits_per_day=value("max_autofits_per_day", quotas.DEFAULT_MAX_AUTOFITS_PER_DAY),
    )


def plans(db: Session) -> list[PlanAdminOut]:
    """Every plan row, by name, with the limits that apply to it."""
    return [
        PlanAdminOut(name=plan.name, limits=_limits_of(plan))
        for plan in db.scalars(select(Plan).order_by(Plan.name))
    ]


def get_plan(db: Session, name: str) -> Plan:
    """The plan row, or 404."""
    plan = db.scalar(select(Plan).where(Plan.name == name))
    if plan is None:
        raise not_found("plan")
    return plan


def update_plan(db: Session, actor: User, plan: Plan, body: PlanUpdateIn) -> PlanAdminOut:
    """Set the plan's four limits; a stored limit the body does not name is kept. Audited as
    ``plan.update``; limits that leave the row as it is write nothing. Flushes; the caller
    commits."""
    limits = body.limits.model_dump()
    # A new object, never an in-place edit: a change inside a JSON column is not tracked.
    merged = {**(plan.limits or {}), **limits}
    if merged != plan.limits:
        audit.record(
            db,
            "plan.update",
            "plan",
            user_id=actor.id,
            entity_id=plan.id,
            detail={"name": plan.name, "limits": limits},
        )
        plan.limits = merged
    return PlanAdminOut(name=plan.name, limits=PlanLimitsIn(**limits))
