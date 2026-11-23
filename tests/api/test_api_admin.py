"""The administrator API: the guard on every route, the overview figures, accounts, error
events, the audit trail, the system report and plan limits."""

import dataclasses
import json
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from alembic.script import ScriptDirectory
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from starlette.types import Scope

from conftest import linear_csv, make_settings, new_email, signup, upload
from curvestudio_api import __version__, quotas
from curvestudio_api.app import create_app
from curvestudio_api.middleware import is_api_request, matched_route
from curvestudio_api.migrations import _config
from curvestudio_api.modules.admin import service
from curvestudio_api.modules.admin.models import RequestStatDaily
from curvestudio_api.modules.admin.requeststats import BUCKET_COUNT, P50, P95, bucket_of, percentile
from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.modules.errors.service import record_error_event
from curvestudio_api.modules.installs.models import InstallEvent, InstallRecord
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
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, utcnow

API = "/api/v1"
ADMIN = f"{API}/admin"
VALID_ID = "3f2a9c1e-7b4d-4e8a-9c21-5d6f7a8b9c0d"
FREE_LIMITS = {
    "max_datasets": 200,
    "max_dataset_bytes": 500_000_000,
    "max_fits": 2000,
    "max_autofits_per_day": 100,
}
PLUS_LIMITS = {
    "max_datasets": 7,
    "max_dataset_bytes": 8_000,
    "max_fits": 9,
    "max_autofits_per_day": 1,
}
LIMITS = {"max_datasets": 10, "max_dataset_bytes": 1000, "max_fits": 20, "max_autofits_per_day": 3}
USER_KEYS = {
    "id",
    "email",
    "name",
    "role",
    "plan",
    "plan_override",
    "is_active",
    "email_verified",
    "created_at",
    "last_login_at",
    "last_active_on",
    "usage",
}
ERROR_KEYS = {
    "id",
    "source",
    "category",
    "severity",
    "error_type",
    "message",
    "fingerprint",
    "route",
    "method",
    "status_code",
    "request_id",
    "app_version",
    "status",
    "occurrence_count",
    "first_seen_at",
    "last_seen_at",
}
DETAIL_KEYS = ERROR_KEYS | {"stack", "context_json", "console_log", "user_agent"}
ENVELOPE_KEYS = {"error", "message", "detail"}


# -- helpers ---------------------------------------------------------------------------------------


def _ist(day: date, hour: int = 0, minute: int = 0) -> datetime:
    """The UTC instant of that time on ``day`` by the IST calendar."""
    return datetime.combine(day, time(hour, minute), tzinfo=IST).astimezone(UTC)


def _account(
    email: str,
    *,
    name: str = "",
    created_at: datetime | None = None,
    verified: bool = True,
    active_on: date | None = None,
    role: str = "member",
    is_active: bool = True,
    plan_override: str | None = None,
) -> User:
    """An account as it stands in the table, with the ages the test wants."""
    at = utcnow()
    return User(
        email=email,
        password_hash="not-a-real-hash",
        name=name or email.split("@", 1)[0],
        created_at=at if created_at is None else created_at,
        email_verified_at=at if verified else None,
        last_active_on=active_on,
        role=role,
        is_active=is_active,
        plan_override=plan_override,
    )


def admin_app(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    **overrides: Any,
) -> tuple[TestClient, dict[str, str]]:
    """A fresh app whose first account is an administrator: the client and its headers."""
    owner = new_email()
    client = client_factory(settings_factory(admin_emails=[owner], **overrides))
    return client, signup(client, owner)


def account_id(client: TestClient, headers: dict[str, str]) -> uuid.UUID:
    return uuid.UUID(client.get(f"{API}/me", headers=headers).json()["id"])


def audit_rows(client: TestClient, action: str) -> list[AuditLog]:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return list(
            db.scalars(
                select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.created_at)
            )
        )


def count(db, model: Any, *where: Any) -> int:
    return int(db.scalar(select(func.count()).select_from(model).where(*where)) or 0)


# -- the guard -------------------------------------------------------------------------------------


# The guard test is parametrised over the routes the application really registers, so the
# table has to exist at collection time. The same app then serves the module-scoped fixture.
_scratch = tempfile.TemporaryDirectory(prefix="curvestudio-admin-routes-")
OWNER = "owner@admin.example.org"
_app: FastAPI = create_app(make_settings(Path(_scratch.name), admin_emails=[OWNER]))

_BODIES: dict[tuple[str, str], dict[str, Any]] = {
    ("PATCH", f"{ADMIN}/users/{{id}}"): {"role": "member"},
    ("PATCH", f"{ADMIN}/errors/{{id}}"): {"status": "acknowledged"},
    ("PATCH", f"{ADMIN}/plans/{{name}}"): {"limits": LIMITS},
}


def _api_routes(routes: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every API route under ``routes`` as ``(full path, route)``.

    This FastAPI version keeps each included router in a container that holds the original
    router and the prefix it was included under, so the tree is walked; a version that lists
    the routes flat is walked the same way, since a plain route has no inner router.
    """
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is None:
            yield prefix + getattr(route, "path", ""), route
            continue
        context = getattr(route, "include_context", None)
        yield from _api_routes(inner.routes, prefix + (getattr(context, "prefix", "") or ""))


def _route_table(app: FastAPI) -> list[tuple[str, str, dict[str, Any] | None]]:
    """Every admin route of ``app``: its method, a concrete path and a valid body where the
    route needs one."""
    table = []
    for path, route in _api_routes(app.routes):
        if not path.startswith(f"{ADMIN}/"):
            continue
        concrete = path.replace("{id}", VALID_ID).replace("{name}", "free")
        for method in sorted(getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}):
            table.append((method, concrete, _BODIES.get((method, path))))
    return table


ROUTE_TABLE = _route_table(_app)

CONTRACT_ROUTES = {
    ("GET", f"{ADMIN}/overview"),
    ("GET", f"{ADMIN}/users"),
    ("GET", f"{ADMIN}/users/{{id}}"),
    ("PATCH", f"{ADMIN}/users/{{id}}"),
    ("DELETE", f"{ADMIN}/users/{{id}}"),
    ("GET", f"{ADMIN}/installs/summary"),
    ("GET", f"{ADMIN}/errors"),
    ("GET", f"{ADMIN}/errors/{{id}}"),
    ("PATCH", f"{ADMIN}/errors/{{id}}"),
    ("GET", f"{ADMIN}/errors/{{id}}/bundle"),
    ("GET", f"{ADMIN}/audit"),
    ("GET", f"{ADMIN}/system"),
    ("GET", f"{ADMIN}/system/latency"),
    ("GET", f"{ADMIN}/plans"),
    ("PATCH", f"{ADMIN}/plans/{{name}}"),
}


def test_the_admin_routes_are_exactly_the_fifteen_of_the_contract() -> None:
    """A parametrised test over an empty table is skipped, not failed: this one makes sure the
    table read from the application is the contract's, so the guard test below covers every
    admin route there is."""
    found = {
        (method, path.replace(VALID_ID, "{id}").replace("/plans/free", "/plans/{name}"))
        for method, path, _ in ROUTE_TABLE
    }
    assert found == CONTRACT_ROUTES


@pytest.fixture(scope="module")
def guarded() -> Iterator[tuple[TestClient, dict[str, str]]]:
    """The module's app, running, with its owner signed in as an administrator."""
    ctx = _app.state.ctx
    ctx.ip_limiter.limit = ctx.install_limiter.limit = ctx.error_limiter.limit = 100_000
    with TestClient(_app) as client:
        yield client, signup(client, OWNER)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    ROUTE_TABLE,
    ids=[f"{method} {path}" for method, path, _ in ROUTE_TABLE],
)
def test_every_admin_route_refuses_a_member_and_an_anonymous_caller(
    guarded: tuple[TestClient, dict[str, str]],
    method: str,
    path: str,
    body: dict[str, Any] | None,
) -> None:
    client, _admin = guarded
    member = signup(client)

    refused = client.request(method, path, headers=member, json=body)
    assert refused.status_code == 403, refused.text
    assert refused.json()["error"] == "FORBIDDEN"
    assert set(refused.json()) == ENVELOPE_KEYS  # no data rides along

    anonymous = client.request(method, path, json=body)
    assert anonymous.status_code == 401, anonymous.text
    assert anonymous.json()["error"] == "UNAUTHORIZED"
    assert set(anonymous.json()) == ENVELOPE_KEYS


# -- the overview ----------------------------------------------------------------------------------


def test_overview_figures_equal_direct_counts(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory)
    ctx = client.app.state.ctx
    now = utcnow()
    today = now.astimezone(IST).date()
    ten_days_ago, eleven_days_ago = today - timedelta(days=10), today - timedelta(days=11)
    active = [
        _account("active-today@example.org", active_on=today),
        _account("active-yesterday@example.org", active_on=today - timedelta(days=1)),
        _account("active-day-6@example.org", active_on=today - timedelta(days=6)),
        _account("active-day-7@example.org", active_on=today - timedelta(days=7)),
        _account("active-day-29@example.org", active_on=today - timedelta(days=29)),
        _account("active-day-30@example.org", active_on=today - timedelta(days=30)),
        _account("never-active@example.org"),
    ]
    installs = [
        _install("a" * 64, "0.11.0", now - timedelta(days=40)),
        _install("b" * 64, __version__, now - timedelta(days=10)),
        _install("c" * 64, __version__, now - timedelta(days=10)),
    ]
    with ctx.session_factory() as db:
        signed_up = [
            _account("old-verified@example.org", created_at=now - timedelta(days=100)),
            _account(
                "old-unverified@example.org",
                created_at=now - timedelta(days=100),
                verified=False,
            ),
            _account("new-verified@example.org", created_at=now - timedelta(days=5)),
            # Sign-ups around IST midnights: the same IST day on both sides of one.
            _account("ten-days-morning@example.org", created_at=_ist(ten_days_ago, 10, 0)),
            _account("ten-days-night@example.org", created_at=_ist(ten_days_ago, 23, 30)),
            _account("ten-days-first-minute@example.org", created_at=_ist(ten_days_ago)),
            _account(
                "eleven-days-last-minute@example.org", created_at=_ist(eleven_days_ago, 23, 59)
            ),
            _account("eleven-days-night@example.org", created_at=_ist(eleven_days_ago, 0, 15)),
            _account(
                "first-day-of-series@example.org",
                created_at=_ist(today - timedelta(days=89), 12),
            ),
            _account("before-series@example.org", created_at=_ist(today - timedelta(days=90), 12)),
        ]
        db.add_all([*signed_up, *active, *installs])
        db.flush()
        db.add_all(
            [
                UserDailyActivity(user_id=active[0].id, day=today - timedelta(days=3)),
                UserDailyActivity(user_id=active[1].id, day=today - timedelta(days=3)),
                UserDailyActivity(user_id=active[0].id, day=today - timedelta(days=1)),
                UserDailyActivity(user_id=active[2].id, day=today - timedelta(days=89)),
            ]
            + [
                InstallEvent(install_record_id=record.id, event="install", to_version=__version__)
                for record in installs
            ]
        )
        # Two new events and one each in the other three statuses.
        for index, name in enumerate("abcde"):
            event, _ = record_error_event(
                db,
                source="client",
                category="crash",
                severity="error",
                error_type="TypeError",
                message=f"error {name}",
                now=now,
            )
            if index >= 2:
                event.status = ("acknowledged", "resolved", "ignored")[index - 2]
        db.add_all(
            [
                AuditLog(
                    user_id=None,
                    action="autofit.submit",
                    entity="autofit",
                    created_at=at,
                )
                for at in (
                    now - timedelta(hours=23),
                    now - timedelta(hours=1),
                    now,
                    now - timedelta(hours=25),
                    now - timedelta(days=3),
                )
            ]
            + [AuditLog(user_id=None, action="fit.submit", entity="fit", created_at=now)]
        )
        db.commit()

    response = client.get(f"{ADMIN}/overview", headers=admin)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "accounts",
        "active_users",
        "installs",
        "open_errors",
        "autofit_runs_24h",
        "series",
    }

    with ctx.session_factory() as db:
        total = count(db, User)
        verified = count(db, User, User.email_verified_at.is_not(None))
        new_30d = count(db, User, User.created_at >= now - timedelta(days=30))
        d1 = count(db, User, User.last_active_on >= today)
        d7 = count(db, User, User.last_active_on >= today - timedelta(days=6))
        d30 = count(db, User, User.last_active_on >= today - timedelta(days=29))
        open_errors = count(db, ErrorEvent, ErrorEvent.status == "new")
        runs_24h = count(
            db,
            AuditLog,
            AuditLog.action == "autofit.submit",
            AuditLog.created_at >= now - timedelta(hours=24),
        )
        installs_total = count(db, InstallRecord)
        installs_new = count(
            db, InstallRecord, InstallRecord.first_seen_at >= now - timedelta(days=30)
        )
        on_latest = count(db, InstallRecord, InstallRecord.app_version == __version__)
        # The series straight from the rows: sign-ups by the IST day of the instant,
        # activity by the day itself.
        signups = Counter(
            created.astimezone(IST).date() for created in db.scalars(select(User.created_at))
        )
        active_days = Counter(db.scalars(select(UserDailyActivity.day)))

    assert body["accounts"] == {"total": total, "verified": verified, "new_30d": new_30d}
    assert (total, verified, new_30d) == (18, 17, 14)
    assert body["active_users"] == {"d1": d1, "d7": d7, "d30": d30}
    assert (d1, d7, d30) == (2, 4, 6)  # the windows end with today, which counts
    assert body["open_errors"] == open_errors == 2
    assert body["autofit_runs_24h"] == runs_24h == 3
    installs_out = body["installs"]
    assert (
        (installs_out["total"], installs_out["new_30d"]) == (installs_total, installs_new) == (3, 2)
    )
    assert installs_out["latest_release"] == __version__
    assert installs_out["latest_share"] == on_latest / installs_total == 2 / 3

    series = body["series"]
    assert [row["day"] for row in series] == [
        (today - timedelta(days=back)).isoformat() for back in range(89, -1, -1)
    ]
    assert all(set(row) == {"day", "signups", "active"} for row in series)
    assert [row["signups"] for row in series] == [
        signups.get(date.fromisoformat(row["day"]), 0) for row in series
    ]
    assert [row["active"] for row in series] == [
        active_days.get(date.fromisoformat(row["day"]), 0) for row in series
    ]
    by_day = {row["day"]: row for row in series}
    assert by_day[ten_days_ago.isoformat()]["signups"] == 3
    assert by_day[eleven_days_ago.isoformat()]["signups"] == 2
    assert by_day[(today - timedelta(days=89)).isoformat()] == {
        "day": (today - timedelta(days=89)).isoformat(),
        "signups": 1,
        "active": 1,
    }
    # Today holds the administrator's sign-up, the accounts seeded "now" and its own activity.
    assert by_day[today.isoformat()]["signups"] == 1 + len(active)
    assert by_day[today.isoformat()]["active"] == 1
    assert by_day[(today - timedelta(days=3)).isoformat()]["active"] == 2
    assert by_day[(today - timedelta(days=1)).isoformat()]["active"] == 1
    # Three accounts are older than the window: the two hundred-day-old ones and the one
    # from the day before it starts.
    assert sum(row["signups"] for row in series) == total - 3


def _install(key: str, version: str, first_seen: datetime) -> InstallRecord:
    return InstallRecord(
        install_key=key,
        app_version=version,
        os_name="linux",
        os_version="6.8",
        arch="x86_64",
        package="appimage",
        session="wayland",
        first_seen_at=first_seen,
        last_seen_at=first_seen,
    )


# -- accounts --------------------------------------------------------------------------------------


def test_users_search_filter_and_paginate(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory, billing_enabled=True)
    ctx = client.app.state.ctx
    admin_id = account_id(client, admin)
    now = utcnow()
    alice = _account(
        "alice.anderson@example.org", created_at=now - timedelta(days=4), plan_override="plus"
    )
    bob = _account(
        "BOB@example.org", name="Bob Brown", created_at=now - timedelta(days=3), role="admin"
    )
    carol = _account("carol@example.org", created_at=now - timedelta(days=2), is_active=False)
    # An override naming no plan row resolves to ``free``.
    dave = _account("dave@example.org", created_at=now - timedelta(days=1), plan_override="nope")
    with ctx.session_factory() as db:
        db.add(Plan(name="plus", limits=PLUS_LIMITS))
        db.add_all([alice, bob, carol, dave])
        db.commit()
        seeded = [user.id for user in (alice, bob, carol, dave)]

    def page(**params: Any) -> dict[str, Any]:
        return client.get(f"{ADMIN}/users", headers=admin, params=params).json()

    admin_email = client.get(f"{API}/me", headers=admin).json()["email"]
    # No filter: every account, newest first, on pages of 25.
    listed = page()
    assert (listed["total"], listed["page"], listed["page_size"]) == (5, 1, 25)
    assert [item["id"] for item in listed["items"]] == [str(admin_id)] + [
        str(i) for i in reversed(seeded)
    ]
    assert set(listed["items"][0]) == USER_KEYS
    assert set(listed["items"][0]["usage"]) == {
        "datasets",
        "dataset_bytes",
        "fits",
        "autofits_today",
    }

    # Pages of two run through the same order without repeating or skipping anybody.
    seen = [item["id"] for number in (1, 2, 3) for item in page(page=number, page_size=2)["items"]]
    assert seen == [item["id"] for item in listed["items"]]
    assert page(page=2, page_size=2)["total"] == 5
    assert page(page_size=2)["page_size"] == 2

    # ``q`` is a case-insensitive substring of the name or the address.
    assert page(q="ali")["items"][0]["email"] == "alice.anderson@example.org"
    assert page(q="BROWN")["items"][0]["email"] == "BOB@example.org"  # as stored
    assert page(q=" example.ORG ")["total"] == 5  # every address, padded and capitalised
    assert page(q="nobody") == {"items": [], "total": 0, "page": 1, "page_size": 25}

    def emails(**params: Any) -> list[str]:
        return sorted(item["email"] for item in page(**params)["items"])

    assert emails(role="admin") == sorted([admin_email, "BOB@example.org"])
    assert emails(active="false") == ["carol@example.org"]
    assert emails(active="true", role="member") == [
        "alice.anderson@example.org",
        "dave@example.org",
    ]
    # The plan filter follows the resolved plan, with billing on.
    assert emails(plan="plus") == ["alice.anderson@example.org"]
    assert emails(plan="free") == sorted(
        [admin_email, "BOB@example.org", "carol@example.org", "dave@example.org"]
    )
    assert page(plan="self-hosted")["total"] == 0

    for params in (
        {"role": "owner"},
        {"plan": "nope"},
        {"active": "maybe"},
        {"page": 0},
        {"page_size": 101},
    ):
        refused = client.get(f"{ADMIN}/users", headers=admin, params=params)
        assert refused.status_code == 422, params
        assert refused.json()["error"] == "VALIDATION_ERROR"

    # With billing off every account resolves to ``self-hosted``, whatever its override says.
    with ctx.session_factory() as db:
        off = ctx.settings.model_copy(update={"billing_enabled": False})
        assert service.list_users(db, off, plan="self-hosted").total == 5
        assert service.list_users(db, off, plan="free").total == 0

    # Each account's plan is the resolved one; the override is shown as stored.
    by_email = {item["email"]: item for item in page()["items"]}
    assert by_email["alice.anderson@example.org"]["plan"] == "plus"
    assert by_email["dave@example.org"]["plan"] == "free"
    assert by_email["dave@example.org"]["plan_override"] == "nope"
    assert by_email["carol@example.org"]["is_active"] is False
    assert by_email["BOB@example.org"]["created_at"].endswith("+05:30")


def test_user_detail_shows_the_account_and_its_last_audit_rows(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory)
    ctx = client.app.state.ctx
    member = signup(client)
    member_id = account_id(client, member)
    now = utcnow()
    with ctx.session_factory() as db:
        db.add_all(
            AuditLog(
                user_id=member_id,
                action="user.name_change",
                entity="user",
                detail={"n": index},
                created_at=now + timedelta(microseconds=index),
            )
            for index in range(60)
        )
        db.commit()

    detail = client.get(f"{ADMIN}/users/{member_id}", headers=admin)
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert set(body) == {"user", "audit"}
    assert set(body["user"]) == USER_KEYS and body["user"]["id"] == str(member_id)
    audit = body["audit"]
    assert len(audit) == 50  # the last fifty rows, newest first
    assert [row["detail"]["n"] for row in audit[:3]] == [59, 58, 57]
    assert set(audit[0]) == {
        "id",
        "user_id",
        "action",
        "entity",
        "entity_id",
        "detail",
        "request_id",
        "created_at",
    }
    assert audit[0]["user_id"] == str(member_id) and audit[0]["created_at"].endswith("+05:30")
    assert client.get(f"{ADMIN}/users/{uuid.uuid4()}", headers=admin).status_code == 404
    assert client.get(f"{ADMIN}/users/{VALID_ID}", headers=admin).json()["error"] == "NOT_FOUND"


def test_patch_user_audits_each_change(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory, billing_enabled=True)
    ctx = client.app.state.ctx
    admin_id = account_id(client, admin)
    member = signup(client)
    member_id = account_id(client, member)
    with ctx.session_factory() as db:
        db.add(Plan(name="plus", limits=PLUS_LIMITS))
        db.commit()
    url = f"{ADMIN}/users/{member_id}"

    def patch(body: dict[str, Any]) -> Any:
        return client.patch(url, headers=admin, json=body)

    promoted = patch({"role": "admin"})
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["role"] == "admin"
    assert set(promoted.json()) == USER_KEYS

    planned = patch({"plan_override": "plus"})
    assert planned.status_code == 200, planned.text
    assert (planned.json()["plan"], planned.json()["plan_override"]) == ("plus", "plus")

    cleared = patch({"plan_override": None})
    assert cleared.status_code == 200, cleared.text
    assert (cleared.json()["plan"], cleared.json()["plan_override"]) == ("free", None)

    off = patch({"is_active": False})
    assert off.status_code == 200 and off.json()["is_active"] is False
    assert client.get(f"{API}/me", headers=member).status_code == 401  # at once, token or not
    on = patch({"is_active": True})
    assert on.status_code == 200 and on.json()["is_active"] is True
    assert client.get(f"{API}/me", headers=member).status_code == 200

    def rows_of(action: str) -> list[AuditLog]:
        """The audit rows of ``action`` about the member (the administrator's own promotion is
        a ``user.role_change`` row too, but about the administrator)."""
        with ctx.session_factory() as db:
            return [r for r in audit_rows_of(db, action) if r.entity_id == str(member_id)]

    assert [(r.user_id, r.detail) for r in rows_of("user.role_change")] == [
        (admin_id, {"from": "member", "to": "admin"})
    ]
    plan_changes = rows_of("user.plan_change")
    assert [r.detail for r in plan_changes] == [
        {"from": None, "to": "plus"},
        {"from": "plus", "to": None},
    ]
    assert all(r.user_id == admin_id for r in plan_changes)
    assert len(rows_of("user.deactivate")) == len(rows_of("user.activate")) == 1
    assert rows_of("user.deactivate")[0].user_id == admin_id
    assert rows_of("user.deactivate")[0].detail == {}

    # A body that names nothing, an unknown key, an unknown plan and an unknown role are
    # refused, and nothing is written.
    for body in ({}, {"nickname": "x"}, {"plan_override": "nope"}, {"role": "owner"}):
        refused = patch(body)
        assert refused.status_code == 422, body
        assert refused.json()["error"] == "VALIDATION_ERROR"
    with ctx.session_factory() as db:
        user = db.get(User, member_id)
        assert user is not None and user.plan_override is None and user.role == "admin"
    assert len(rows_of("user.plan_change")) == 2

    # The values the account already has are not a change: nothing is written.
    for body in ({"role": "admin"}, {"plan_override": None}, {"is_active": True}):
        assert patch(body).status_code == 200
    assert len(rows_of("user.role_change")) == 1
    assert len(rows_of("user.plan_change")) == 2
    assert len(rows_of("user.activate")) == 1

    unknown = client.patch(f"{ADMIN}/users/{uuid.uuid4()}", headers=admin, json={"role": "member"})
    assert unknown.status_code == 404 and unknown.json()["error"] == "NOT_FOUND"


def audit_rows_of(db, action: str) -> list[AuditLog]:
    return list(
        db.scalars(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.created_at))
    )


def test_an_administrator_cannot_demote_or_deactivate_themselves(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory, billing_enabled=True)
    ctx = client.app.state.ctx
    admin_id = account_id(client, admin)
    with ctx.session_factory() as db:
        db.add(Plan(name="plus", limits=PLUS_LIMITS))
        db.commit()
    url = f"{ADMIN}/users/{admin_id}"

    for body in ({"role": "member"}, {"is_active": False}):
        refused = client.patch(url, headers=admin, json=body)
        assert refused.status_code == 422, refused.text
        assert refused.json()["error"] == "VALIDATION_ERROR"
        assert refused.json()["detail"] == {"code": "SELF_CHANGE_REFUSED"}
    # Naming both a plan and the refused field writes nothing at all.
    both = client.patch(url, headers=admin, json={"plan_override": "plus", "role": "member"})
    assert both.status_code == 422 and both.json()["detail"] == {"code": "SELF_CHANGE_REFUSED"}

    with ctx.session_factory() as db:
        me = db.get(User, admin_id)
        assert me is not None and me.role == "admin" and me.is_active and me.plan_override is None
        for action in ("user.role_change", "user.plan_change", "user.deactivate"):
            mine = [
                r
                for r in audit_rows_of(db, action)
                if r.user_id == admin_id and r.entity_id == str(admin_id)
            ]
            assert len(mine) == (1 if action == "user.role_change" else 0)  # the one promotion row
    assert (
        client.get(f"{ADMIN}/overview", headers=admin).status_code == 200
    )  # still an administrator

    # Their own plan is theirs to change.
    allowed = client.patch(url, headers=admin, json={"plan_override": "plus"})
    assert allowed.status_code == 200, allowed.text
    assert (allowed.json()["plan"], allowed.json()["role"]) == ("plus", "admin")
    with ctx.session_factory() as db:
        changes = audit_rows_of(db, "user.plan_change")
        assert [(r.user_id, r.entity_id, r.detail) for r in changes] == [
            (admin_id, str(admin_id), {"from": None, "to": "plus"})
        ]


def test_admin_deletion_erases_like_delete_me(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, admin = admin_app(client_factory, settings_factory)
    ctx = client.app.state.ctx
    admin_id = account_id(client, admin)
    member = signup(client)
    member_id = account_id(client, member)
    dataset = upload(client, member, linear_csv()).json()
    cancelled: list[uuid.UUID] = []
    monkeypatch.setattr(ctx.jobs, "cancel_for_user", cancelled.append)
    with ctx.session_factory() as db:
        db.add_all(
            [
                Fit(
                    user_id=member_id,
                    dataset_id=uuid.UUID(dataset["id"]),
                    spec={},
                    bundle={},
                    status="succeeded",
                ),
                Job(
                    user_id=member_id,
                    dataset_id=uuid.UUID(dataset["id"]),
                    kind="fit",
                    status="queued",
                    spec={},
                ),
            ]
        )
        db.commit()
        before = count(db, AuditLog, AuditLog.user_id == member_id)
        assert before >= 2  # at least the sign-up and the sign-in
    assert count_rows(client, RefreshToken, member_id) >= 1  # the session the member signed in with

    erased = client.delete(f"{ADMIN}/users/{member_id}", headers=admin)
    assert erased.status_code == 204 and erased.content == b""
    assert cancelled == [member_id]
    assert client.get(f"{API}/me", headers=member).status_code == 401

    with ctx.session_factory() as db:
        assert db.get(User, member_id) is None
        for model in (Dataset, Fit, Job, RefreshToken):
            assert count(db, model, model.user_id == member_id) == 0
        assert count(db, AuditLog, AuditLog.user_id == member_id) == 0
        # The account's rows are still there, anonymised.
        anonymised = count(
            db, AuditLog, AuditLog.user_id.is_(None), AuditLog.action == "auth.login"
        )
        assert anonymised == 1
        erased_rows = audit_rows_of(db, "user.delete")
        assert len(erased_rows) == 1
        assert (erased_rows[0].user_id, erased_rows[0].entity_id) == (admin_id, str(member_id))

    assert client.delete(f"{ADMIN}/users/{uuid.uuid4()}", headers=admin).status_code == 404
    assert client.delete(f"{ADMIN}/users/{VALID_ID}", headers=admin).json()["error"] == "NOT_FOUND"


def count_rows(client: TestClient, model: Any, user_id: uuid.UUID) -> int:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return count(db, model, model.user_id == user_id)


# -- error events ----------------------------------------------------------------------------------


def _report(
    client: TestClient,
    message: str,
    *,
    category: str = "crash",
    at: datetime | None = None,
    **extra: Any,
) -> ErrorEvent:
    """One client event in ``client``'s database, through the errors service."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        event, _ = record_error_event(
            db,
            source="client",
            category=category,
            severity="error",
            error_type="TypeError",
            message=message,
            stack="TypeError: boom\n    at render (app.js:1:2)",
            now=at or utcnow(),
            **extra,
        )
        db.commit()
        return event


def test_errors_list_filter_and_status_changes(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory)
    admin_id = account_id(client, admin)
    now = utcnow()
    oldest = _report(client, "first", at=now - timedelta(hours=3))
    middle = _report(client, "second", category="ui", at=now - timedelta(hours=2))
    newest = _report(client, "third", at=now - timedelta(hours=1))
    order = [str(event.id) for event in (newest, middle, oldest)]

    def page(**params: Any) -> dict[str, Any]:
        return client.get(f"{ADMIN}/errors", headers=admin, params=params).json()

    listed = page()
    assert (listed["total"], listed["page"], listed["page_size"]) == (3, 1, 25)
    assert [item["id"] for item in listed["items"]] == order  # newest first
    assert set(listed["items"][0]) == ERROR_KEYS
    assert listed["items"][0]["status"] == "new" and listed["items"][0]["occurrence_count"] == 1
    assert "stack" not in listed["items"][0] and "user_agent" not in listed["items"][0]
    assert [item["id"] for item in page(category="ui")["items"]] == [str(middle.id)]
    assert page(category="network")["total"] == 0
    assert [item["id"] for item in page(status="new", page_size=2)["items"]] == order[:2]
    for params in ({"status": "fixed"}, {"page_size": 500}):
        assert client.get(f"{ADMIN}/errors", headers=admin, params=params).status_code == 422

    # A status change is audited under the administrator, against the event.
    url = f"{ADMIN}/errors/{oldest.id}"
    resolved = client.patch(url, headers=admin, json={"status": "resolved"})
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["status"] == "resolved" and set(resolved.json()) == ERROR_KEYS
    assert [item["id"] for item in page(status="resolved")["items"]] == [str(oldest.id)]
    assert [item["id"] for item in page(status="new")["items"]] == order[:2]
    changes = audit_rows(client, "error.status_change")
    assert len(changes) == 1
    assert (changes[0].user_id, changes[0].entity_id, changes[0].detail) == (
        admin_id,
        str(oldest.id),
        {"from": "new", "to": "resolved"},
    )
    assert changes[0].entity == "error_event"

    # The same status again is not a change.
    assert client.patch(url, headers=admin, json={"status": "resolved"}).status_code == 200
    assert len(audit_rows(client, "error.status_change")) == 1

    # A repeat of the resolved error reopens it, through the errors service.
    _report(client, "first")
    after = client.get(url, headers=admin)
    assert after.status_code == 200
    detail = after.json()
    assert set(detail) == DETAIL_KEYS
    assert (detail["status"], detail["occurrence_count"]) == ("new", 2)
    assert detail["stack"] == "TypeError: boom\n    at render (app.js:1:2)"
    assert detail["context_json"] is None and detail["console_log"] is None
    assert detail["first_seen_at"] < detail["last_seen_at"]  # both with the +05:30 offset
    assert [item["id"] for item in page(status="new")["items"]] == [str(oldest.id), *order[:2]]

    for status in ("fixed", ""):
        assert client.patch(url, headers=admin, json={"status": status}).status_code == 422
    assert client.patch(url, headers=admin, json={"state": "resolved"}).status_code == 422
    assert client.get(f"{ADMIN}/errors/{uuid.uuid4()}", headers=admin).status_code == 404
    unknown = client.patch(f"{ADMIN}/errors/{uuid.uuid4()}", headers=admin, json={"status": "new"})
    assert unknown.status_code == 404 and unknown.json()["error"] == "NOT_FOUND"


def test_error_bundle_is_markdown(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory)
    context = {"component": "Plot", "rows": 3, "note": "a | b"}
    event = _report(
        client,
        "Cannot read properties of undefined (reading 'x') | in Plot",
        route="/api/v1/plots/{id}",
        method="GET",
        status_code=500,
        request_id="req-42",
        user_agent="UA/1.0",
        app_version="0.12.0",
        context=context,
        console_log="warn: first\nwarn: second",
    )

    response = client.get(f"{ADMIN}/errors/{event.id}/bundle", headers=admin)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == f'attachment; filename="error-{event.id}.md"'
    text = response.text
    assert text.startswith("# Error event\n")
    for needle in (
        str(event.id),
        event.fingerprint,
        "TypeError",
        "Cannot read properties of undefined (reading 'x') \\| in Plot",  # a cell escapes pipes
        "/api/v1/plots/{id}",
        "GET",
        "500",
        "req-42",
        "UA/1.0",
        "0.12.0",
        "new",
        "client",
        "crash",
        "error",
        '"a | b"',  # the JSON block keeps its text as it is
    ):
        assert needle in text, needle
    assert "+05:30" in text  # timestamps in IST
    assert "```text\nTypeError: boom\n    at render (app.js:1:2)\n```" in text
    assert (
        "```json\n" + json.dumps(context, indent=2, ensure_ascii=False, sort_keys=True) + "\n```"
        in text
    )
    assert "```text\nwarn: first\nwarn: second\n```" in text
    assert text.index("## Stack") < text.index("## Context") < text.index("## Console log")
    assert client.get(f"{ADMIN}/errors/{uuid.uuid4()}/bundle", headers=admin).status_code == 404


# -- the audit trail -------------------------------------------------------------------------------


def test_audit_filters(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory)
    ctx = client.app.state.ctx
    admin_id = account_id(client, admin)
    # Rows on both sides of an IST midnight: 23:30 on the 20th is that day, 00:30 on the 21st
    # (still the 20th by UTC) is the next one.
    day = (utcnow() - timedelta(days=10)).astimezone(IST).date()
    seeded = [
        ("auth.login", admin_id, _ist(day, 23, 30)),
        ("user.register", admin_id, _ist(day + timedelta(days=1), 0, 30)),
        ("autofit.submit", None, _ist(day + timedelta(days=1), 12, 0)),
        ("plan.update", None, _ist(day + timedelta(days=2), 9, 0)),
    ]
    with ctx.session_factory() as db:
        db.add_all(
            AuditLog(user_id=user_id, action=action, entity="user", created_at=at)
            for action, user_id, at in seeded
        )
        db.commit()

    def page(**params: Any) -> dict[str, Any]:
        return client.get(f"{ADMIN}/audit", headers=admin, params=params).json()

    def actions(body: dict[str, Any]) -> list[str]:
        return [row["action"] for row in body["items"]]

    everything = page()
    # Newest first. The administrator's own three rows (sign-up, sign-in and the promotion
    # the admin list earned at sign-in) are all from now; which of the last two came first is
    # not something the rows record, so only the group is asserted, then the seeded rows.
    assert everything["total"] == 7
    assert set(actions(everything)[:3]) == {"auth.login", "user.register", "user.role_change"}
    assert actions(everything)[3:] == [
        "plan.update",
        "autofit.submit",
        "user.register",
        "auth.login",
    ]
    assert set(everything["items"][0]) == {
        "id",
        "user_id",
        "action",
        "entity",
        "entity_id",
        "detail",
        "request_id",
        "created_at",
    }
    first = page(page_size=2)
    assert first["items"] == everything["items"][:2] and first["total"] == 7
    assert page(page=3, page_size=2)["items"] == everything["items"][4:6]
    assert page(page=4, page_size=2)["items"] == everything["items"][-1:]
    assert page(page=5, page_size=2)["items"] == []

    next_day = (day + timedelta(days=1)).isoformat()
    assert actions(page(**{"from": next_day, "to": next_day})) == [
        "autofit.submit",
        "user.register",
    ]
    # From that day on: the administrator's own rows from now come first.
    from_that_day = page(**{"from": next_day})
    assert from_that_day["total"] == 6
    assert set(actions(from_that_day)[:3]) == {"auth.login", "user.register", "user.role_change"}
    assert actions(from_that_day)[3:] == ["plan.update", "autofit.submit", "user.register"]
    assert actions(page(**{"to": day.isoformat()})) == ["auth.login"]
    assert actions(page(action="user.register")) == ["user.register", "user.register"]
    assert actions(page(user_id=str(admin_id), action="auth.login")) == ["auth.login", "auth.login"]
    assert page(user_id=str(uuid.uuid4()))["total"] == 0
    # The audit rows of the administrator's own account, as the detail view shows them.
    detail = client.get(f"{ADMIN}/users/{admin_id}", headers=admin).json()["audit"]
    assert {row["action"] for row in detail} == {"user.register", "auth.login", "user.role_change"}

    for params in (
        {"from": (day + timedelta(days=2)).isoformat(), "to": day.isoformat()},
        {"from": "yesterday"},
        {"user_id": "nope"},
        {"page": -1},
    ):
        refused = client.get(f"{ADMIN}/audit", headers=admin, params=params)
        assert refused.status_code == 422, params
        assert refused.json()["error"] == "VALIDATION_ERROR"


# -- the system report and plans -------------------------------------------------------------------


def test_system_reports_workers_queue_and_database(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(
        client_factory, settings_factory, job_workers=2, job_queue_max=7, current_release="0.12.0"
    )
    ctx = client.app.state.ctx
    response = client.get(f"{ADMIN}/system", headers=admin)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"version", "current_release", "started_at", "uptime_s", "jobs", "database"}
    assert body["version"] == __version__
    assert body["current_release"] == "0.12.0"
    assert body["started_at"].endswith("+05:30")
    assert 0 <= body["uptime_s"] < 60
    assert body["jobs"] == {
        "running": True,
        "workers": 2,
        "queue_depth": ctx.jobs.queue_size(),
        "queue_max": 7,
    }
    assert ctx.jobs.worker_count() == 2
    head = ScriptDirectory.from_config(_config(ctx.engine)).get_current_head()
    assert body["database"] == {"size_bytes": None, "migration_head": head}  # SQLite has no size

    # An empty setting means the API's own version.
    plain = dataclasses.replace(
        ctx, settings=ctx.settings.model_copy(update={"current_release": ""})
    )
    assert service.system(plain).current_release == __version__


def test_plans_update_validates_limits(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client, admin = admin_app(client_factory, settings_factory, billing_enabled=True)
    ctx = client.app.state.ctx
    admin_id = account_id(client, admin)

    listed = client.get(f"{ADMIN}/plans", headers=admin)
    assert listed.status_code == 200, listed.text
    plans = {plan["name"]: plan for plan in listed.json()}
    assert list(plans) == ["free", "self-hosted"]  # by name
    assert plans["free"] == {"name": "free", "limits": FREE_LIMITS}
    # The self-hosted row stores no Auto-fit allowance: the default is what applies and is shown.
    assert plans["self-hosted"]["limits"] == FREE_LIMITS

    url = f"{ADMIN}/plans/free"
    for limits in (
        {**LIMITS, "max_datasets": 0},
        {**LIMITS, "max_fits": -1},
        {key: value for key, value in LIMITS.items() if key != "max_fits"},
        {**LIMITS, "extra": 1},
    ):
        refused = client.patch(url, headers=admin, json={"limits": limits})
        assert refused.status_code == 422, limits
        assert refused.json()["error"] == "VALIDATION_ERROR"
    missing = client.patch(url, headers=admin, json={})
    assert missing.status_code == 422
    unknown = client.patch(f"{ADMIN}/plans/nope", headers=admin, json={"limits": LIMITS})
    assert unknown.status_code == 404 and unknown.json()["error"] == "NOT_FOUND"
    assert audit_rows(client, "plan.update") == []
    with ctx.session_factory() as db:
        assert db.scalar(select(Plan).where(Plan.name == "free")).limits == FREE_LIMITS

    changed = client.patch(url, headers=admin, json={"limits": LIMITS})
    assert changed.status_code == 200, changed.text
    assert changed.json() == {"name": "free", "limits": LIMITS}
    with ctx.session_factory() as db:
        free = db.scalar(select(Plan).where(Plan.name == "free"))
        assert free is not None and free.limits == LIMITS
        rows = audit_rows(client, "plan.update")
        assert len(rows) == 1
        assert (rows[0].user_id, rows[0].entity_id, rows[0].entity) == (
            admin_id,
            str(free.id),
            "plan",
        )
        assert rows[0].detail == {"name": "free", "limits": LIMITS}
        assert quotas.autofits_per_day_limit(db, ctx.settings, "free") == 3

    # A member's limits follow the row at once.
    member = signup(client)
    assert client.get(f"{API}/me", headers=member).json()["plan"] == {"name": "free", **LIMITS}
    assert client.get(f"{ADMIN}/plans", headers=admin).json()[0]["limits"] == LIMITS

    # The same limits again leave the row as it is: nothing is written.
    again = client.patch(url, headers=admin, json={"limits": LIMITS})
    assert again.status_code == 200 and again.json()["limits"] == LIMITS
    assert len(audit_rows(client, "plan.update")) == 1


def test_only_api_requests_are_counted_in_the_statistics(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    """The web app's own pages and assets are served by one catch-all route. A figure that
    mixed them with API calls would say nothing about the API, so they are left out."""
    client, _admin = admin_app(client_factory, settings_factory)

    @client.app.get("/{path:path}", include_in_schema=False)
    def _page(path: str) -> dict[str, str]:
        return {"path": path}

    # Setting the administrator up made its own API calls; they are not this test's.
    client.app.state.ctx.stats.snapshot_and_reset()
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get(f"{API}/health").status_code == 200

    assert set(client.app.state.ctx.stats.snapshot_and_reset()) == {f"{API}/health"}


def test_the_matched_route_is_read_as_its_full_template() -> None:
    """What the middleware derives from a routed scope, for the statistics and the error
    capture alike."""
    nested = {"path": f"{API}/datasets/3f2a9c1e-0000-4000-8000-000000000000", "route": None}
    nested["route"] = SimpleNamespace(path_format="/datasets/{dataset_id}")
    assert matched_route(cast("Scope", nested)) == f"{API}/datasets/{{dataset_id}}"
    assert is_api_request(cast("Scope", nested)) is True

    own = {"path": f"{API}/boom/1", "route": SimpleNamespace(path_format=f"{API}/boom/{{id}}")}
    assert matched_route(cast("Scope", own)) == f"{API}/boom/{{id}}"

    page = {"path": "/assets/app.js", "route": SimpleNamespace(path_format="/{path:path}")}
    assert matched_route(cast("Scope", page)) == "/{path:path}"
    assert is_api_request(cast("Scope", page)) is False

    assert matched_route(cast("Scope", {"path": "/nope", "route": None})) is None
    assert is_api_request(cast("Scope", {"path": "/api/v10/health", "route": None})) is False


def test_route_latency_reads_the_percentiles_off_the_summed_days(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    """Over a window the days' histograms are summed and the percentiles read off the sum.
    They are never an average of the days' percentiles: a slow day and a fast day do not make
    a medium one."""
    client, admin = admin_app(client_factory, settings_factory)
    today = utcnow().astimezone(IST).date()
    fast = [0] * BUCKET_COUNT
    fast[bucket_of(5)] = 10
    slow = [0] * BUCKET_COUNT
    slow[bucket_of(2_000)] = 10
    with client.app.state.ctx.session_factory() as db:
        for day, histogram, errors, slowest in (
            (today, fast, 1, 6),
            (today - timedelta(days=1), slow, 0, 2_100),
            (today - timedelta(days=6), fast, 0, 6),  # in a week, not in a day
            (today - timedelta(days=29), slow, 0, 2_100),  # in 30 days, not in a week
            (today - timedelta(days=30), fast, 0, 6),  # in no window
            (today - timedelta(days=40), slow, 0, 2_100),
        ):
            db.add(
                RequestStatDaily(
                    day=day,
                    route_template="/api/v1/a",
                    count=20,
                    errors=errors,
                    histogram=histogram,
                    p50_ms=0,
                    p95_ms=0,
                    max_ms=slowest,
                )
            )
        db.add(
            RequestStatDaily(
                day=today,
                route_template="/api/v1/b",
                count=3,
                errors=0,
                histogram=[0] * BUCKET_COUNT,
                p50_ms=0,
                p95_ms=0,
                max_ms=0,
            )
        )
        db.commit()

    def window(days: int | None) -> list[dict[str, Any]]:
        response = client.get(
            f"{ADMIN}/system/latency", params={} if days is None else {"days": days}, headers=admin
        )
        assert response.status_code == 200, response.text
        return response.json()

    by_route = {row["route_template"]: row for row in window(1)}
    assert by_route["/api/v1/a"] == {
        "route_template": "/api/v1/a",
        "count": 20,
        "errors": 1,
        "p50_ms": percentile(fast, P50),
        "p95_ms": percentile(fast, P95),
        "max_ms": 6,
    }
    assert window(None)[0]["count"] == 20  # the default window is one day

    # a week ending today holds six days ago: two fast days and one slow
    a_week = {row["route_template"]: row for row in window(7)}["/api/v1/a"]
    summed = [2 * x + y for x, y in zip(fast, slow, strict=True)]
    assert (a_week["count"], a_week["errors"], a_week["max_ms"]) == (60, 1, 2_100)
    assert a_week["p50_ms"] == percentile(summed, P50)
    assert a_week["p95_ms"] == percentile(summed, P95)
    assert a_week["p95_ms"] != (percentile(fast, P95) + percentile(slow, P95)) // 2

    # 30 days hold 29 days ago, and nothing older
    assert {row["route_template"]: row for row in window(30)}["/api/v1/a"]["count"] == 80
    assert [row["route_template"] for row in window(7)] == ["/api/v1/a", "/api/v1/b"]

    refused = client.get(f"{ADMIN}/system/latency", params={"days": 2}, headers=admin)
    assert refused.status_code == 422 and refused.json()["error"] == "VALIDATION_ERROR"
