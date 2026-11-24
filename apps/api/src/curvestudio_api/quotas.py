"""Per-user quotas from the account's resolved ``plans`` row, overridable by settings."""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from curvestudio_api import audit
from curvestudio_api.errors import ApiError
from curvestudio_api.orm import AuditLog, Dataset, Fit, Job, Plan, User
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, utcnow

PLAN_NAME = "self-hosted"
FREE_PLAN_NAME = "free"
DEFAULT_LIMITS = {"max_datasets": 200, "max_dataset_bytes": 500_000_000, "max_fits": 2000}
DEFAULT_MAX_AUTOFITS_PER_DAY = 100
Quota = Literal["datasets", "dataset_bytes", "fits"]


@dataclass(frozen=True)
class Limits:
    max_datasets: int
    max_dataset_bytes: int
    max_fits: int


@dataclass(frozen=True)
class Usage:
    datasets: int
    dataset_bytes: int
    fits: int


def resolve_plan(db: Session, settings: Settings, override: str | None) -> Plan:
    """The one way to resolve an account's plan row.

    Billing off: ``self-hosted`` — today's behaviour for every install. Billing on: the
    override when it names a known row, else ``free``.

    Raises:
        ApiError: 503 when the plans table holds no rows at all (both seeds are removed only
            by hand).
    """
    wanted = PLAN_NAME if not settings.billing_enabled else (override or FREE_PLAN_NAME)
    plan = db.scalar(select(Plan).where(Plan.name == wanted))
    if plan is None:
        # An override naming an unknown row (or, unexpectedly, a missing seeded row) falls
        # back to the other seeded row rather than turning every request into an error.
        wanted = FREE_PLAN_NAME if wanted != FREE_PLAN_NAME else PLAN_NAME
        plan = db.scalar(select(Plan).where(Plan.name == wanted))
    if plan is None:
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the plan rows are not seeded")
    return plan


def _plan_override(db: Session, user_id: uuid.UUID) -> str | None:
    return db.scalar(select(User.plan_override).where(User.id == user_id))


def effective_limits(db: Session, settings: Settings, override: str | None = None) -> Limits:
    """Settings override the resolved plan row, which overrides the built-in defaults."""
    stored = dict(resolve_plan(db, settings, override).limits)

    def pick(setting: int | None, key: str) -> int:
        if setting is not None:
            return setting
        value = stored.get(key)
        return value if isinstance(value, int) and value > 0 else DEFAULT_LIMITS[key]

    return Limits(
        max_datasets=pick(settings.quota_max_datasets, "max_datasets"),
        max_dataset_bytes=pick(settings.quota_max_dataset_bytes, "max_dataset_bytes"),
        max_fits=pick(settings.quota_max_fits, "max_fits"),
    )


def usage(db: Session, user_id: uuid.UUID) -> Usage:
    # A dataset's bytes are its two stored payloads. A dataset from before has no table
    # payload (NULL), which SUM leaves out; the sum of an account with none at all is NULL
    # too, so each part is brought to 0 on its own.
    stored = func.coalesce(func.sum(func.length(Dataset.payload)), 0) + func.coalesce(
        func.sum(func.length(Dataset.table)), 0
    )
    datasets, nbytes = db.execute(
        select(func.count(Dataset.id), stored).where(Dataset.user_id == user_id)
    ).one()
    fits = db.scalar(select(func.count(Fit.id)).where(Fit.user_id == user_id)) or 0
    return Usage(datasets=int(datasets), dataset_bytes=int(nbytes), fits=int(fits))


def active_jobs(db: Session, user_id: uuid.UUID) -> int:
    """Queued or running jobs that will store a fit (an Auto-fit run stores none, and a
    cross-validation stores its result on a fit that is already counted)."""
    return (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.user_id == user_id,
                Job.kind.not_in(("autofit", "audit")),
                Job.status.in_(("queued", "running")),
            )
        )
        or 0
    )


def check(
    db: Session, settings: Settings, user_id: uuid.UUID, quota: Quota, *, adding_bytes: int = 0
) -> None:
    """Refuse the creation of one more dataset or fit (or ``adding_bytes`` more data).

    Raises:
        ApiError: 409 ``QUOTA_EXCEEDED`` for counts, 413 ``QUOTA_EXCEEDED`` for bytes; the
            detail names the quota, the limits and the current usage.
    """
    limits = effective_limits(db, settings, _plan_override(db, user_id))
    used = usage(db, user_id)
    if quota == "datasets":
        exceeded = used.datasets + 1 > limits.max_datasets
        limit, status = limits.max_datasets, 409
    elif quota == "dataset_bytes":
        exceeded = used.dataset_bytes + adding_bytes > limits.max_dataset_bytes
        limit, status = limits.max_dataset_bytes, 413
    else:
        # Jobs in flight will each store a fit, so they count against the quota too.
        exceeded = used.fits + active_jobs(db, user_id) + 1 > limits.max_fits
        limit, status = limits.max_fits, 409
    if exceeded:
        raise ApiError(
            status,
            "QUOTA_EXCEEDED",
            f"the {quota.replace('_', ' ')} quota of this plan ({limit:,}) is used up; "
            "delete something first",
            {
                "code": "QUOTA_EXCEEDED",
                "quota": quota,
                "plan": resolve_plan(db, settings, _plan_override(db, user_id)).name,
                "limit": limit,
                "limits": asdict(limits),
                "usage": asdict(used),
            },
        )


def autofits_per_day_limit(db: Session, settings: Settings, override: str | None = None) -> int:
    """Auto-fit runs one user may start per day: the setting, else the plan row, else 100."""
    if settings.quota_max_autofits_per_day is not None:
        return settings.quota_max_autofits_per_day
    value = resolve_plan(db, settings, override).limits.get("max_autofits_per_day")
    return value if isinstance(value, int) and value > 0 else DEFAULT_MAX_AUTOFITS_PER_DAY


def autofit_runs_today(db: Session, user_id: uuid.UUID, now: datetime | None = None) -> int:
    """Auto-fit runs the user has started in the IST calendar day.

    This is the number ``check_autofit_runs`` compares with the allowance, so a page that shows
    it beside the limit never disagrees with the refusal. Every run counts, including
    cancelled and failed ones.
    """
    current = (now or utcnow()).astimezone(IST)
    day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    # Audit rows, not Job rows: deleting a dataset cascades its jobs away, which would
    # otherwise reset the day's allowance.
    return int(
        db.scalar(
            select(func.count(AuditLog.id)).where(
                AuditLog.user_id == user_id,
                AuditLog.action == "autofit.submit",
                AuditLog.created_at >= day_start,
            )
        )
        or 0
    )


def check_autofit_runs(
    db: Session, settings: Settings, user_id: uuid.UUID, now: datetime | None = None
) -> None:
    """Refuse one more Auto-fit run once the user has started the day's allowance.

    The day is the IST calendar day. Every run counts, including cancelled and failed ones.

    Raises:
        ApiError: 429 ``QUOTA_EXCEEDED`` with ``Retry-After`` set to the seconds until the
            next IST midnight.
    """
    current = (now or utcnow()).astimezone(IST)
    day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    limit = autofits_per_day_limit(db, settings, _plan_override(db, user_id))
    used = autofit_runs_today(db, user_id, now)
    if used + 1 > limit:
        retry_after = max(1, int((day_start + timedelta(days=1) - current).total_seconds()))
        raise ApiError(
            429,
            "QUOTA_EXCEEDED",
            f"the limit of {limit:,} Auto-fit runs per day is used up; try again tomorrow",
            {"code": "QUOTA_EXCEEDED", "quota": "autofits_per_day", "limit": limit, "used": used},
            {"Retry-After": str(retry_after)},
        )


def set_plan_override(db: Session, user: User, override: str | None, actor: User | None) -> None:
    """Set (or clear) the account's plan override, then audit the change as ``user.plan_change``.

    Only an admin (P5) or the entitlement connector calls this: the name must match a row in
    ``plans``; ``None`` clears the override. The audit detail carries the old and the new name
    and nothing else.
    """
    if override is not None and db.scalar(select(Plan).where(Plan.name == override)) is None:
        raise ApiError(422, "VALIDATION_ERROR", f"there is no plan named {override!r}")
    detail = {"from": user.plan_override, "to": override}
    user.plan_override = override
    audit.record(
        db,
        "user.plan_change",
        "user",
        user_id=None if actor is None else actor.id,
        entity_id=user.id,
        detail=detail,
    )
