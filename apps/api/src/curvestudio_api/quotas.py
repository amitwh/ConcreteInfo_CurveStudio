"""Per-user quotas from the ``plans`` row (``self-hosted``), overridable by settings."""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from curvestudio_api.errors import ApiError
from curvestudio_api.orm import Dataset, Fit, Job, Plan
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, utcnow

PLAN_NAME = "self-hosted"
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


def plan_limits(db: Session) -> dict[str, object]:
    """The ``limits`` JSON of the self-hosted plan (empty if the row is missing)."""
    plan = db.scalar(select(Plan).where(Plan.name == PLAN_NAME))
    return dict(plan.limits) if plan is not None else {}


def effective_limits(db: Session, settings: Settings) -> Limits:
    """Settings override the plan row, which overrides the built-in defaults."""
    stored = plan_limits(db)

    def pick(override: int | None, key: str) -> int:
        if override is not None:
            return override
        value = stored.get(key)
        return value if isinstance(value, int) and value > 0 else DEFAULT_LIMITS[key]

    return Limits(
        max_datasets=pick(settings.quota_max_datasets, "max_datasets"),
        max_dataset_bytes=pick(settings.quota_max_dataset_bytes, "max_dataset_bytes"),
        max_fits=pick(settings.quota_max_fits, "max_fits"),
    )


def usage(db: Session, user_id: uuid.UUID) -> Usage:
    datasets, nbytes = db.execute(
        select(
            func.count(Dataset.id), func.coalesce(func.sum(func.length(Dataset.payload)), 0)
        ).where(Dataset.user_id == user_id)
    ).one()
    fits = db.scalar(select(func.count(Fit.id)).where(Fit.user_id == user_id)) or 0
    return Usage(datasets=int(datasets), dataset_bytes=int(nbytes), fits=int(fits))


def active_jobs(db: Session, user_id: uuid.UUID) -> int:
    """Queued or running jobs that will store a fit (an Auto-fit run stores none)."""
    return (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.user_id == user_id,
                Job.kind != "autofit",
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
    limits = effective_limits(db, settings)
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
                "limit": limit,
                "limits": asdict(limits),
                "usage": asdict(used),
            },
        )


def autofits_per_day_limit(db: Session, settings: Settings) -> int:
    """Auto-fit runs one user may start per day: the setting, else the plan row, else 100."""
    if settings.quota_max_autofits_per_day is not None:
        return settings.quota_max_autofits_per_day
    value = plan_limits(db).get("max_autofits_per_day")
    return value if isinstance(value, int) and value > 0 else DEFAULT_MAX_AUTOFITS_PER_DAY


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
    limit = autofits_per_day_limit(db, settings)
    used = (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.user_id == user_id, Job.kind == "autofit", Job.created_at >= day_start
            )
        )
        or 0
    )
    if used + 1 > limit:
        retry_after = max(1, int((day_start + timedelta(days=1) - current).total_seconds()))
        raise ApiError(
            429,
            "QUOTA_EXCEEDED",
            f"the limit of {limit:,} Auto-fit runs per day is used up; try again tomorrow",
            {"code": "QUOTA_EXCEEDED", "quota": "autofits_per_day", "limit": limit, "used": used},
            {"Retry-After": str(retry_after)},
        )
