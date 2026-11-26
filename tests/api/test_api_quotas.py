"""Per-user quotas from the plan row, /me usage, housekeeping and audit capping."""

import logging
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import api_job_workers
import pytest
from alembic import command
from api_test_helpers import linear_csv, new_email, signup, upload
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text

from curvestudio_api.app import create_app
from curvestudio_api.errors import ApiError
from curvestudio_api.housekeeping import purge
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.modules.admin.models import RequestStatDaily
from curvestudio_api.modules.admin.requeststats import BUCKET_COUNT
from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.modules.installs.models import InstallEvent, InstallRecord
from curvestudio_api.orm import AuditLog, Dataset, Fit, Job, Plan, RefreshToken, User
from curvestudio_api.quotas import autofits_per_day_limit, resolve_plan, set_plan_override
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, utcnow

API = "/api/v1"
POLY1 = {"model": {"kind": "polynomial", "degree": 1}}


def user_id(client: TestClient, headers: dict[str, str]) -> uuid.UUID:
    return uuid.UUID(client.get(f"{API}/me", headers=headers).json()["id"])


def test_me_reports_the_plan_and_usage(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    empty = shared_client.get(f"{API}/me", headers=headers).json()
    assert empty["plan"] == {
        "name": "self-hosted",
        "max_datasets": 200,
        "max_dataset_bytes": 500_000_000,
        "max_fits": 2000,
        "max_autofits_per_day": 100,
    }
    assert empty["usage"] == {"datasets": 0, "dataset_bytes": 0, "fits": 0, "autofits_today": 0}
    upload(shared_client, headers, linear_csv(30))
    used = shared_client.get(f"{API}/me", headers=headers).json()["usage"]
    assert used["datasets"] == 1
    assert used["dataset_bytes"] > 100
    assert used["fits"] == 0


def test_me_counts_the_days_autofit_runs_against_the_plan_limit(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    """The profile shows today's Auto-fit runs beside the plan's daily allowance: the same
    count the endpoint that refuses a run compares, and only today's (IST) rows."""
    client = client_factory(settings_factory(quota_max_autofits_per_day=3))
    headers = signup(client)
    account_id = user_id(client, headers)
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        for started_at in (utcnow(), utcnow(), utcnow() - timedelta(days=1)):
            db.add(
                AuditLog(
                    user_id=account_id,
                    action="autofit.submit",
                    entity="job",
                    created_at=started_at,
                )
            )
        db.commit()

    body = client.get(f"{API}/me", headers=headers).json()
    assert body["plan"]["max_autofits_per_day"] == 3
    assert body["usage"]["autofits_today"] == 2


def test_dataset_count_quota(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(quota_max_datasets=2))
    headers, other = signup(client), signup(client)
    assert upload(client, headers, linear_csv()).status_code == 201
    assert upload(client, headers, linear_csv()).status_code == 201
    refused = upload(client, headers, linear_csv())
    assert refused.status_code == 409
    body = refused.json()
    assert body["error"] == "QUOTA_EXCEEDED"
    assert body["detail"]["quota"] == "datasets"
    assert body["detail"]["limit"] == 2
    assert body["detail"]["usage"]["datasets"] == 2
    assert body["detail"]["limits"]["max_fits"] == 2000
    assert upload(client, other, linear_csv()).status_code == 201  # quotas are per user
    # deleting makes room again
    first = client.get(f"{API}/datasets", headers=headers).json()[0]["id"]
    assert client.delete(f"{API}/datasets/{first}", headers=headers).status_code == 204
    assert upload(client, headers, linear_csv()).status_code == 201


def test_dataset_bytes_quota(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(quota_max_dataset_bytes=1500))
    headers = signup(client)
    assert upload(client, headers, linear_csv(30)).status_code == 201
    size = client.get(f"{API}/me", headers=headers).json()["usage"]["dataset_bytes"]
    assert 0 < size < 1500
    assert upload(client, headers, linear_csv(30)).status_code == 201 if 2 * size <= 1500 else True
    refused = upload(client, headers, linear_csv(300))
    assert refused.status_code == 413
    assert refused.json()["error"] == "QUOTA_EXCEEDED"
    assert refused.json()["detail"]["plan"] == "self-hosted"
    assert refused.json()["detail"]["quota"] == "dataset_bytes"
    assert refused.json()["detail"]["limit"] == 1500


def test_fit_quota_counts_stored_fits_and_jobs_in_flight(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(
        settings_factory(quota_max_fits=2), job_worker=api_job_workers.sleeping_worker
    )
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(
            Fit(
                id=uuid.uuid4(),
                user_id=user_id(client, headers),
                dataset_id=uuid.UUID(dataset["id"]),
                spec={},
                bundle={},
                status="converged",
            )
        )
        db.commit()
    first = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=POLY1)
    assert first.status_code == 202
    refused = client.post(f"{API}/datasets/{dataset['id']}/fits", headers=headers, json=POLY1)
    assert refused.status_code == 409
    assert refused.json()["error"] == "QUOTA_EXCEEDED"
    assert refused.json()["detail"]["quota"] == "fits"
    assert refused.json()["detail"]["usage"]["fits"] == 1


def test_the_plan_row_sets_the_limits_and_settings_override_it(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any], tmp_path: Path
) -> None:
    client = client_factory()
    headers = signup(client)
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        plan = db.scalar(select(Plan).where(Plan.name == "self-hosted"))
        assert plan is not None
        plan.limits = {"max_datasets": 1, "max_dataset_bytes": 123_456, "max_fits": 7}
        db.commit()
    assert client.get(f"{API}/me", headers=headers).json()["plan"]["max_datasets"] == 1
    assert upload(client, headers, linear_csv()).status_code == 201
    assert upload(client, headers, linear_csv()).status_code == 409
    overridden = client_factory(
        settings_factory(quota_max_fits=9, database_url=f"sqlite:///{tmp_path}/second.db")
    )
    plan_of = overridden.get(f"{API}/me", headers=signup(overridden)).json()["plan"]
    assert plan_of["max_fits"] == 9 and plan_of["max_datasets"] == 200


# -- plan resolution -------------------------------------------------------------------------------

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
SMALL = {"families": ["polynomial"], "max_parameters": 2, "cv_folds": 3, "n_starts": 1}


def test_billing_off_resolves_self_hosted(shared_client: TestClient) -> None:
    """The default: every install resolves to the ``self-hosted`` row and /me says so."""
    headers = signup(shared_client)
    assert shared_client.get(f"{API}/me", headers=headers).json()["plan"]["name"] == "self-hosted"
    ctx = shared_client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.session_factory() as db:
        user = db.get(User, user_id(shared_client, headers))
        assert user is not None and user.plan_override is None
        assert resolve_plan(db, ctx.settings, user.plan_override).name == "self-hosted"


def test_billing_on_resolves_free_and_the_seeded_row(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    """Billing on with no override: the account is on ``free``, seeded by migration 0006."""
    client = client_factory(settings_factory(billing_enabled=True))
    headers = signup(client)
    assert client.get(f"{API}/me", headers=headers).json()["plan"]["name"] == "free"
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.session_factory() as db:
        plan = resolve_plan(db, ctx.settings, None)
        assert plan.name == "free"
        assert plan.limits == FREE_LIMITS


def test_a_known_override_resolves_to_its_row(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(billing_enabled=True))
    headers = signup(client)
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(Plan(name="plus", limits=PLUS_LIMITS))
        user = db.get(User, user_id(client, headers))
        assert user is not None
        user.plan_override = "plus"
        db.commit()
    reported = client.get(f"{API}/me", headers=headers).json()["plan"]
    assert reported["name"] == "plus"
    assert reported["max_datasets"] == 7 and reported["max_fits"] == 9
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.session_factory() as db:
        assert resolve_plan(db, ctx.settings, "plus").limits["max_autofits_per_day"] == 1


def test_an_unknown_override_falls_back_to_free(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    """An override naming no row must degrade to ``free``, never to an error."""
    client = client_factory(settings_factory(billing_enabled=True))
    headers = signup(client)
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.get(User, user_id(client, headers))
        assert user is not None
        user.plan_override = "nope"
        db.commit()
    response = client.get(f"{API}/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["plan"]["name"] == "free"
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.session_factory() as db:
        assert resolve_plan(db, ctx.settings, "nope").name == "free"


def test_the_autofit_cap_follows_the_resolved_plan(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    """Two plans with different ``max_autofits_per_day`` give different daily caps."""
    client = client_factory(settings_factory(billing_enabled=True))
    headers = signup(client)
    dataset = upload(client, headers, linear_csv()).json()
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(Plan(name="plus", limits=PLUS_LIMITS))
        db.commit()
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    url = f"{API}/datasets/{dataset['id']}/autofit"
    with ctx.session_factory() as db:
        # the seeded free row allows 100 a day; the plus row allows one
        assert autofits_per_day_limit(db, ctx.settings, None) == 100
        assert autofits_per_day_limit(db, ctx.settings, "plus") == 1
    assert client.post(url, headers=headers, json=SMALL).status_code == 202
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None
        user.plan_override = "plus"
        db.commit()
    refused = client.post(url, headers=headers, json=SMALL)
    assert refused.status_code == 429
    body = refused.json()
    assert body["error"] == "QUOTA_EXCEEDED"
    assert body["detail"]["quota"] == "autofits_per_day" and body["detail"]["limit"] == 1


def test_migration_0006_can_be_rolled_back_and_reapplied(
    settings_factory: Callable[..., Any],
) -> None:
    app = create_app(settings_factory())
    engine = app.state.ctx.engine
    upgrade_to_head(engine)
    command.downgrade(_config(engine), "0005")
    with engine.connect() as connection:
        columns = {c["name"] for c in inspect(connection).get_columns("users")}
        plan_names = {row[0] for row in connection.execute(text("SELECT name FROM plans"))}
    assert "plan_override" not in columns
    assert plan_names == {"self-hosted"}
    upgrade_to_head(engine)
    with engine.connect() as connection:
        columns = {c["name"] for c in inspect(connection).get_columns("users")}
        plan_names = {row[0] for row in connection.execute(text("SELECT name FROM plans"))}
    assert "plan_override" in columns
    assert plan_names == {"free", "self-hosted"}


# -- plan override setter --------------------------------------------------------------------------


def test_set_plan_override_writes_and_audits(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    """A known name is written and audited; ``None`` clears the override, also audited."""
    client = client_factory(settings_factory(billing_enabled=True))
    headers = signup(client)
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.session_factory() as db:
        db.add(Plan(name="plus", limits=PLUS_LIMITS))
        db.commit()
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None
        set_plan_override(db, user, "plus", actor=None)
        db.commit()
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None and user.plan_override == "plus"
        changes = list(db.scalars(select(AuditLog).where(AuditLog.action == "user.plan_change")))
        assert [c.detail for c in changes] == [{"from": None, "to": "plus"}]
        assert changes[0].entity_id == str(user.id)
        assert changes[0].user_id is None  # no actor was passed
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None
        set_plan_override(db, user, None, actor=user)
        db.commit()
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None and user.plan_override is None
        changes = list(db.scalars(select(AuditLog).where(AuditLog.action == "user.plan_change")))
        assert [c.detail for c in changes] == [
            {"from": None, "to": "plus"},
            {"from": "plus", "to": None},
        ]
        assert changes[1].user_id == user.id  # the actor is the one recorded


def test_set_plan_override_refuses_an_unknown_name(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(billing_enabled=True))
    headers = signup(client)
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None
        with pytest.raises(ApiError) as raised:
            set_plan_override(db, user, "nope", actor=None)
        assert raised.value.status_code == 422
        assert raised.value.code == "VALIDATION_ERROR"
        db.rollback()
    with ctx.session_factory() as db:
        user = db.get(User, user_id(client, headers))
        assert user is not None and user.plan_override is None
        assert db.scalar(select(AuditLog).where(AuditLog.action == "user.plan_change")) is None


# -- housekeeping ----------------------------------------------------------------------------------


def seed_housekeeping(client: TestClient) -> dict[str, uuid.UUID]:
    now = utcnow()
    ids = {
        name: uuid.uuid4()
        for name in (
            "expired_long_ago",
            "expired_recently",
            "revoked_old_still_valid",
            "revoked_old_expired",
            "job_old_done",
            "job_new_done",
            "job_old_running",
            "user",
            "dataset",
        )
    }
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(
            User(id=ids["user"], email=f"hk{uuid.uuid4().hex[:6]}@example.org", password_hash="x")
        )
        db.flush()
        db.add(
            Dataset(
                id=ids["dataset"],
                user_id=ids["user"],
                name="d",
                n_rows=3,
                columns=["x", "y"],
                x_column="x",
                y_column="y",
                fingerprint="f",
                payload=b"x,y\n1,2\n",
            )
        )
        db.flush()

        def token(name: str, expires: timedelta, revoked: timedelta | None = None) -> None:
            db.add(
                RefreshToken(
                    id=ids[name],
                    user_id=ids["user"],
                    token_hash=uuid.uuid4().hex + uuid.uuid4().hex,
                    expires_at=now + expires,
                    revoked_at=None if revoked is None else now + revoked,
                )
            )

        token("expired_long_ago", -timedelta(days=40))
        token("expired_recently", -timedelta(days=10))
        token("revoked_old_still_valid", timedelta(days=5), -timedelta(days=40))
        token("revoked_old_expired", -timedelta(days=35), -timedelta(days=40))

        def job(name: str, status: str, finished: timedelta | None) -> None:
            db.add(
                Job(
                    id=ids[name],
                    user_id=ids["user"],
                    dataset_id=ids["dataset"],
                    kind="fit",
                    status=status,
                    spec={},
                    finished_at=None if finished is None else now + finished,
                )
            )

        job("job_old_done", "succeeded", -timedelta(days=40))
        job("job_new_done", "failed", -timedelta(days=5))
        job("job_old_running", "running", None)
        db.commit()
    return ids


def test_purge_removes_only_what_is_old_enough(shared_client: TestClient) -> None:
    ids = seed_housekeeping(shared_client)
    result = purge(shared_client.app.state.ctx.session_factory)  # type: ignore[attr-defined]
    assert result.refresh_tokens >= 2 and result.jobs >= 1
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        tokens = {
            t.id
            for t in db.scalars(select(RefreshToken).where(RefreshToken.user_id == ids["user"]))
        }
        jobs = {j.id for j in db.scalars(select(Job).where(Job.user_id == ids["user"]))}
    assert tokens == {ids["expired_recently"], ids["revoked_old_still_valid"]}
    assert jobs == {ids["job_new_done"], ids["job_old_running"]}


def test_housekeeping_runs_on_its_own(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Any]
) -> None:
    client = client_factory(settings_factory(housekeeping_interval_s=0.2))
    ids = seed_housekeeping(client)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
            if db.get(RefreshToken, ids["expired_long_ago"]) is None:
                return
        time.sleep(0.1)
    raise AssertionError("the housekeeping task did not purge the old token")


def _error_event(status: str, last_seen: datetime) -> ErrorEvent:
    return ErrorEvent(
        source="client",
        category="crash",
        severity="error",
        error_type="TypeError",
        message=f"{status}, last seen {last_seen.isoformat()}",
        fingerprint=uuid.uuid4().hex,
        status=status,
        occurrence_count=1,
        first_seen_at=last_seen,
        last_seen_at=last_seen,
    )


def _stat_row(day: date) -> RequestStatDaily:
    return RequestStatDaily(
        day=day,
        route_template="/api/v1/health",
        count=1,
        errors=0,
        histogram=[1] + [0] * (BUCKET_COUNT - 1),
        p50_ms=0,
        p95_ms=0,
        max_ms=1,
    )


def test_the_housekeeper_merges_request_stats_and_purges_by_age(
    client_factory: Callable[..., TestClient],
    settings_factory: Callable[..., Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One run merges what the middleware counted, removes error events and request statistics
    past their retention, and leaves the install records alone."""
    client = client_factory(settings_factory(housekeeping_interval_s=0.05))
    ctx = client.app.state.ctx
    now = utcnow()
    today = now.astimezone(IST).date()
    day = timedelta(days=1)
    gone = [
        _error_event("resolved", now - 91 * day),
        _error_event("ignored", now - 100 * day),
        _error_event("new", now - 181 * day),  # whatever its status, nothing is kept for ever
    ]
    kept = [
        _error_event("resolved", now - 89 * day),
        _error_event("new", now - 100 * day),  # still open: kept until the 180-day rule
        _error_event("acknowledged", now - 179 * day),
    ]
    with ctx.session_factory() as db:
        db.add_all(gone + kept)
        # 400 days back is kept; the day before it goes.
        db.add_all(
            [
                _stat_row(today - 401 * day),
                _stat_row(today - 400 * day),
                _stat_row(today - 399 * day),
            ]
        )
        record = InstallRecord(
            install_key="k" * 64,
            app_version="0.11.1",
            os_name="linux",
            os_version="6.8",
            arch="x86_64",
            package="appimage",
            session="wayland",
            first_seen_at=now - 1000 * day,
            last_seen_at=now - 1000 * day,
        )
        db.add(record)
        db.flush()
        db.add(InstallEvent(install_record_id=record.id, event="install", to_version="0.11.1"))
        db.commit()
        gone_ids, kept_ids = [event.id for event in gone], [event.id for event in kept]

    # What the middleware counted since the previous run.
    ctx.stats.record("/api/v1/health", 12.5, error=False)
    ctx.stats.record("/api/v1/health", 30_000.0, error=True)
    ctx.stats.record(None, 5.0, error=False)  # a request that matched no route

    with caplog.at_level(logging.INFO, logger="curvestudio_api"):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with ctx.session_factory() as db:
                merged = db.scalar(select(RequestStatDaily).where(RequestStatDaily.day == today))
                if (
                    db.get(ErrorEvent, gone_ids[0]) is None
                    and merged is not None
                    and merged.count == 2
                ):
                    break
            time.sleep(0.05)
        else:
            raise AssertionError("the housekeeping task did not merge and purge")

    assert (merged.route_template, merged.count, merged.errors, merged.max_ms) == (
        "/api/v1/health",
        2,
        1,
        30_000,
    )
    assert sum(merged.histogram) == 2
    assert merged.p50_ms < merged.p95_ms <= merged.max_ms
    with ctx.session_factory() as db:
        assert set(db.scalars(select(ErrorEvent.id))) == set(kept_ids)
        assert set(db.scalars(select(RequestStatDaily.day))) == {
            today,
            today - 399 * day,
            today - 400 * day,
        }
        # Install records and their events are never purged.
        assert db.scalar(select(func.count()).select_from(InstallRecord)) == 1
        assert db.scalar(select(func.count()).select_from(InstallEvent)) == 1

    purged = [r for r in caplog.records if r.getMessage() == "purged"]
    lines = [r.fields for r in purged if r.fields["error_events_closed"] == 2]  # type: ignore[attr-defined]
    assert lines, [r.getMessage() for r in caplog.records]
    assert (lines[0]["error_events_expired"], lines[0]["request_stats"]) == (1, 1)
    assert {"tokens", "jobs"} <= set(lines[0])
    assert any(r.fields["stats_merged"] >= 1 for r in purged)  # type: ignore[attr-defined]


def test_repeated_failed_logins_write_a_bounded_number_of_audit_rows(
    shared_client: TestClient,
) -> None:
    email = new_email()
    shared_client.post(
        f"{API}/auth/register", json={"email": email, "password": "correct horse battery staple"}
    )
    for _ in range(104):
        shared_client.post(
            f"{API}/auth/login", json={"email": email, "password": "wrong password!!"}
        )
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        user = db.scalar(select(User).where(User.email == email))
        actions = [
            a.action for a in db.scalars(select(AuditLog).where(AuditLog.user_id == user.id))
        ]
    assert actions.count("auth.login_failed") == 100
    assert actions.count("auth.login_failed_capped") == 1
    # a successful login is still audited
    shared_client.post(
        f"{API}/auth/login", json={"email": email, "password": "correct horse battery staple"}
    )
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert "auth.login" in [
            a.action for a in db.scalars(select(AuditLog).where(AuditLog.user_id == user.id))
        ]
