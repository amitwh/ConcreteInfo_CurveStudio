"""Per-user quotas from the ``plans`` row (``self-hosted``), overridable by settings."""

import uuid
from dataclasses import asdict, dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from curvestudio_api.errors import ApiError
from curvestudio_api.orm import Dataset, Fit, Job, Plan
from curvestudio_api.settings import Settings

PLAN_NAME = "self-hosted"
DEFAULT_LIMITS = {"max_datasets": 200, "max_dataset_bytes": 500_000_000, "max_fits": 2000}
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
    return (
        db.scalar(
            select(func.count(Job.id)).where(
                Job.user_id == user_id, Job.status.in_(("queued", "running"))
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
