"""The bodies of the admin API: what the console lists and shows, and what it sends. An error
event never carries an account identifier here: there is none to carry."""

import uuid
from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from curvestudio_api.modules.installs.schemas import InstallSummaryOut
from curvestudio_api.schemas import UsageOut, _Body
from curvestudio_api.timeutil import IstDatetime

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100

Role = Literal["member", "admin"]
ErrorStatus = Literal["new", "acknowledged", "resolved", "ignored"]


class PageOut[T](BaseModel):
    """One page of a list. ``page`` counts from 1; ``page_size`` is 25 unless asked for and
    never more than 100."""

    items: list[T]
    total: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=MAX_PAGE_SIZE)


class AdminUserOut(BaseModel):
    """An account as the console lists it. ``plan`` is the resolved plan's name."""

    id: uuid.UUID
    email: str
    name: str
    role: Role
    plan: str
    plan_override: str | None
    is_active: bool
    email_verified: bool
    created_at: IstDatetime
    last_login_at: IstDatetime | None
    last_active_on: date | None
    usage: UsageOut


class AdminUserUpdateIn(_Body):
    """What to change on an account. ``plan_override`` set to ``null`` clears it; a key left
    out keeps its value."""

    role: Role | None = None
    plan_override: str | None = Field(default=None, max_length=50)
    is_active: bool | None = None

    @model_validator(mode="after")
    def _names_a_change(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one of role, plan_override and is_active is required")
        return self


class AuditOut(BaseModel):
    """One audit row."""

    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    entity: str
    entity_id: str | None
    detail: dict[str, Any]
    request_id: str | None
    created_at: IstDatetime


class AdminUserDetailOut(BaseModel):
    """An account and the last fifty rows of its audit trail, newest first."""

    user: AdminUserOut
    audit: list[AuditOut]


class ErrorEventOut(BaseModel):
    """An error event as the console lists it: without the large texts."""

    id: uuid.UUID
    source: str
    category: str
    severity: str
    error_type: str
    message: str
    fingerprint: str
    route: str | None
    method: str | None
    status_code: int | None
    request_id: str | None
    app_version: str | None
    status: ErrorStatus
    occurrence_count: int
    first_seen_at: IstDatetime
    last_seen_at: IstDatetime


class ErrorEventDetailOut(ErrorEventOut):
    """An error event with everything stored about it."""

    stack: str | None
    context_json: dict[str, Any] | None
    console_log: str | None
    user_agent: str | None


class ErrorStatusUpdateIn(_Body):
    """The new status of an error event."""

    status: ErrorStatus


class CountOut(BaseModel):
    """One day of the overview's series."""

    day: date
    signups: int
    active: int


class AccountCountsOut(BaseModel):
    total: int
    verified: int
    new_30d: int


class ActiveUsersOut(BaseModel):
    """Accounts active in the last 1, 7 and 30 days; today counts."""

    d1: int
    d7: int
    d30: int


class OverviewOut(BaseModel):
    """The figures of the console's front page."""

    accounts: AccountCountsOut
    active_users: ActiveUsersOut
    installs: InstallSummaryOut
    open_errors: int
    autofit_runs_24h: int
    series: list[CountOut]


class JobsOut(BaseModel):
    running: bool
    workers: int
    queue_depth: int
    queue_max: int


class DatabaseOut(BaseModel):
    """``size_bytes`` is the database's size on PostgreSQL and ``null`` elsewhere;
    ``migration_head`` is the schema revision the database is at."""

    size_bytes: int | None
    migration_head: str


class SystemOut(BaseModel):
    version: str
    current_release: str
    started_at: IstDatetime
    uptime_s: float
    jobs: JobsOut
    database: DatabaseOut


class PlanLimitsIn(_Body):
    """A plan's four limits, each at least 1."""

    max_datasets: int = Field(ge=1)
    max_dataset_bytes: int = Field(ge=1)
    max_fits: int = Field(ge=1)
    max_autofits_per_day: int = Field(ge=1)


class PlanUpdateIn(_Body):
    """What to change on a plan."""

    limits: PlanLimitsIn


class PlanAdminOut(BaseModel):
    """A plan row with the limits that apply to it."""

    name: str
    limits: PlanLimitsIn
