"""``/api/v1/admin``: the administrator API.

Every handler takes ``AdminUser``, so a member is answered 403 and a caller without a token
401 before any handler runs. The rules live in the service and no query is written here.
``id`` in a path is the account's or the error event's id; ``name`` a plan's name.
"""

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query, Response
from fastapi.responses import PlainTextResponse

from curvestudio_api.deps import AdminUser, Ctx, Db
from curvestudio_api.modules.admin import service
from curvestudio_api.modules.admin.schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AdminUserDetailOut,
    AdminUserOut,
    AdminUserUpdateIn,
    AuditOut,
    ErrorEventDetailOut,
    ErrorEventOut,
    ErrorStatus,
    ErrorStatusUpdateIn,
    InstallSummaryOut,
    OverviewOut,
    PageOut,
    PlanAdminOut,
    PlanUpdateIn,
    Role,
    SystemOut,
)

router = APIRouter(prefix="/admin", tags=["admin"])

Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]
Search = Annotated[str | None, Query(max_length=100)]

_BUNDLE = 'attachment; filename="error-{id}.md"'


class MarkdownResponse(PlainTextResponse):
    media_type = "text/markdown"


@router.get(
    "/overview", response_model=OverviewOut, summary="The figures of the console's front page"
)
def overview(user: AdminUser, ctx: Ctx, db: Db) -> OverviewOut:
    """Accounts, active users, installs, open errors, Auto-fit runs and a ninety-day series."""
    return service.overview(db, ctx.settings)


@router.get("/users", response_model=PageOut[AdminUserOut], summary="Accounts, filtered and paged")
def users(
    user: AdminUser,
    ctx: Ctx,
    db: Db,
    q: Search = None,
    plan: Annotated[str | None, Query(max_length=50)] = None,
    role: Role | None = None,
    active: bool | None = None,
    page: Page = 1,
    page_size: PageSize = DEFAULT_PAGE_SIZE,
) -> PageOut[AdminUserOut]:
    """The accounts matching the filters, newest first. ``q`` is a case-insensitive substring
    of the name or the address; ``plan`` is the resolved plan's name."""
    return service.list_users(
        db,
        ctx.settings,
        q=q,
        plan=plan,
        role=role,
        active=active,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/users/{id}", response_model=AdminUserDetailOut, summary="One account and its audit trail"
)
def user_detail(id: uuid.UUID, user: AdminUser, ctx: Ctx, db: Db) -> AdminUserDetailOut:
    """The account, with the last fifty rows of its audit trail, newest first."""
    return service.user_detail(db, ctx.settings, service.get_user(db, id))


@router.patch(
    "/users/{id}",
    response_model=AdminUserOut,
    summary="Change an account's role, plan or active flag",
)
def update_user(
    id: uuid.UUID, body: AdminUserUpdateIn, user: AdminUser, ctx: Ctx, db: Db
) -> AdminUserOut:
    """Each change is audited. An administrator cannot change their own role or active flag."""
    account = service.get_user(db, id)
    changed = service.update_user(db, ctx.settings, user, account, body)
    db.commit()
    return changed


@router.delete("/users/{id}", status_code=204, summary="Erase an account")
def delete_user(id: uuid.UUID, user: AdminUser, ctx: Ctx, db: Db) -> Response:
    """The account and everything in it goes; the audit trail stays, anonymised."""
    service.delete_account(ctx, db, user, id)
    db.commit()
    return Response(status_code=204)


@router.get("/installs/summary", response_model=InstallSummaryOut, summary="The install figures")
def installs(user: AdminUser, ctx: Ctx, db: Db) -> InstallSummaryOut:
    """What the desktop installations reported, counted the way ``modules.installs`` does."""
    return service.install_figures(db, ctx.settings)


@router.get(
    "/errors", response_model=PageOut[ErrorEventOut], summary="Error events, filtered and paged"
)
def errors(
    user: AdminUser,
    db: Db,
    status: ErrorStatus | None = None,
    category: Annotated[str | None, Query(max_length=16)] = None,
    page: Page = 1,
    page_size: PageSize = DEFAULT_PAGE_SIZE,
) -> PageOut[ErrorEventOut]:
    """The error events matching the filters, newest first. Nothing here names an account."""
    return service.list_errors(db, status=status, category=category, page=page, page_size=page_size)


@router.get("/errors/{id}", response_model=ErrorEventDetailOut, summary="One error event")
def error_detail(id: uuid.UUID, user: AdminUser, db: Db) -> ErrorEventDetailOut:
    """The event with its stack, context, console log and user agent."""
    return service.event_detail(db, id)


@router.patch(
    "/errors/{id}", response_model=ErrorEventOut, summary="Change an error event's status"
)
def update_error(
    id: uuid.UUID, body: ErrorStatusUpdateIn, user: AdminUser, db: Db
) -> ErrorEventOut:
    """Audited as ``error.status_change``. A repeat of a resolved or ignored event reopens it."""
    event = service.get_event(db, id)
    changed = service.set_error_status(db, user, event, body.status)
    db.commit()
    return changed


@router.get(
    "/errors/{id}/bundle",
    response_class=MarkdownResponse,
    summary="One error event as a Markdown attachment",
)
def error_bundle(id: uuid.UUID, user: AdminUser, db: Db) -> Response:
    """Every field of the event, its stack, its context and its console lines."""
    return MarkdownResponse(
        service.error_bundle_markdown(db, id),
        headers={"Content-Disposition": _BUNDLE.format(id=id)},
    )


@router.get(
    "/audit", response_model=PageOut[AuditOut], summary="The audit trail, filtered and paged"
)
def audit(
    user: AdminUser,
    db: Db,
    user_id: uuid.UUID | None = None,
    action: Annotated[str | None, Query(max_length=50)] = None,
    from_day: Annotated[date | None, Query(alias="from")] = None,
    to_day: Annotated[date | None, Query(alias="to")] = None,
    page: Page = 1,
    page_size: PageSize = DEFAULT_PAGE_SIZE,
) -> PageOut[AuditOut]:
    """The audit rows matching the filters, newest first. ``from`` and ``to`` are IST dates,
    both inclusive."""
    return service.list_audit(
        db,
        user_id=user_id,
        action=action,
        from_day=from_day,
        to_day=to_day,
        page=page,
        page_size=page_size,
    )


@router.get("/system", response_model=SystemOut, summary="Version, uptime, jobs and database")
def system(user: AdminUser, ctx: Ctx) -> SystemOut:
    """How this instance is running."""
    return service.system(ctx)


@router.get("/plans", response_model=list[PlanAdminOut], summary="The plan rows")
def plans(user: AdminUser, db: Db) -> list[PlanAdminOut]:
    """Every plan with the limits that apply to it, by name."""
    return service.plans(db)


@router.patch("/plans/{name}", response_model=PlanAdminOut, summary="Change a plan's limits")
def update_plan(name: str, body: PlanUpdateIn, user: AdminUser, db: Db) -> PlanAdminOut:
    """Sets the plan's four limits, each at least 1. Audited as ``plan.update``."""
    changed = service.update_plan(db, user, service.get_plan(db, name), body)
    db.commit()
    return changed
