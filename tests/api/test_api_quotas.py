"""Per-user quotas from the plan row, /me usage, housekeeping and audit capping."""

import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import api_job_workers
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text

from conftest import linear_csv, new_email, signup, upload
from curvestudio_api.app import create_app
from curvestudio_api.housekeeping import purge
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.orm import AuditLog, Dataset, Fit, Job, Plan, RefreshToken, User
from curvestudio_api.quotas import autofits_per_day_limit, resolve_plan
from curvestudio_api.timeutil import utcnow

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
    }
    assert empty["usage"] == {"datasets": 0, "dataset_bytes": 0, "fits": 0}
    upload(shared_client, headers, linear_csv(30))
    used = shared_client.get(f"{API}/me", headers=headers).json()["usage"]
    assert used["datasets"] == 1
    assert used["dataset_bytes"] > 100
    assert used["fits"] == 0


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
