"""The P5 schema and shared plumbing: the 0008-0010 migrations' round trips, the model
registration, the install and error limiters, the per-path body limits and the
current-release setting."""

import asyncio
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from api_test_helpers import make_settings
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from curvestudio_api.app import JSON_BODY_LIMIT, create_app
from curvestudio_api.db import make_session_factory
from curvestudio_api.deps import ERROR_REPORTS_PER_HOUR, INSTALL_RECORDS_PER_HOUR
from curvestudio_api.errors import ApiError
from curvestudio_api.middleware import BodyLimitMiddleware
from curvestudio_api.migrations import _config, upgrade_to_head
from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.modules.installs.models import InstallEvent, InstallRecord
from curvestudio_api.ratelimit import SlidingWindowLimiter
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import utcnow

API = "/api/v1"

COLUMNS = {
    "install_records": {
        "id",
        "install_key",
        "app_version",
        "os_name",
        "os_version",
        "arch",
        "package",
        "session",
        "first_seen_at",
        "last_seen_at",
        "created_at",
        "updated_at",
    },
    "install_events": {
        "id",
        "install_record_id",
        "event",
        "from_version",
        "to_version",
        "created_at",
    },
    "error_events": {
        "id",
        "source",
        "category",
        "severity",
        "error_type",
        "message",
        "fingerprint",
        "stack",
        "route",
        "method",
        "status_code",
        "request_id",
        "user_agent",
        "app_version",
        "status",
        "occurrence_count",
        "first_seen_at",
        "last_seen_at",
        "context_json",
        "console_log",
        "created_at",
        "updated_at",
    },
    "request_stats_daily": {
        "id",
        "day",
        "route_template",
        "count",
        "errors",
        "histogram",
        "p50_ms",
        "p95_ms",
        "max_ms",
        "created_at",
        "updated_at",
    },
}
INDEXES = {
    "install_events": {"ix_install_events_install_record_id"},
    "error_events": {"ix_error_events_status", "ix_error_events_last_seen_at"},
}
UNIQUE_CONSTRAINTS = {
    "install_records": {"uq_install_records_install_key"},
    "error_events": {"uq_error_events_fingerprint"},  # one row per error (0011)
    "request_stats_daily": {"uq_request_stats_daily_day"},
}
FOREIGN_KEYS = {"install_events": {"fk_install_events_install_record_id_install_records"}}


# -- migrations ------------------------------------------------------------------------------------


def _upgraded_engine(tmp_path: Path) -> Engine:
    """An application's database brought to head."""
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    return engine


def _assert_complete(engine: Engine, tables: list[str]) -> None:
    """Each table is there with its columns, indexes and constraints under their convention
    names (the names the migrations and the models must agree on)."""
    inspector = inspect(engine)
    for table in tables:
        columns = {column["name"] for column in inspector.get_columns(table)}
        assert columns == COLUMNS[table]
        indexes = {index["name"] for index in inspector.get_indexes(table)}
        assert indexes == INDEXES.get(table, set())
        unique = {item["name"] for item in inspector.get_unique_constraints(table)}
        assert unique == UNIQUE_CONSTRAINTS.get(table, set())
        foreign = {item["name"] for item in inspector.get_foreign_keys(table)}
        assert foreign == FOREIGN_KEYS.get(table, set())


def _round_trip(engine: Engine, revision: str, below: str, created: list[str]) -> None:
    """From head, back to ``revision`` and the one step down it undoes: exactly ``created`` goes
    and no other table changes; back to head, every table of ``created`` is back, complete."""
    config = _config(engine)
    at_head = set(inspect(engine).get_table_names())
    command.downgrade(config, revision)
    before = set(inspect(engine).get_table_names())
    assert set(created) <= before

    command.downgrade(config, below)
    assert set(inspect(engine).get_table_names()) == before - set(created)

    upgrade_to_head(engine)
    assert set(inspect(engine).get_table_names()) == at_head
    _assert_complete(engine, created)


def test_migration_0008_round_trip(tmp_path: Path) -> None:
    _round_trip(_upgraded_engine(tmp_path), "0008", "0007", ["install_records", "install_events"])


def test_migration_0009_round_trip(tmp_path: Path) -> None:
    _round_trip(_upgraded_engine(tmp_path), "0009", "0008", ["error_events"])


def test_migration_0010_round_trip(tmp_path: Path) -> None:
    _round_trip(_upgraded_engine(tmp_path), "0010", "0009", ["request_stats_daily"])


def _fingerprint_unique_constraint(engine: Engine) -> set[str]:
    """The names of the unique constraints on ``error_events.fingerprint``."""
    return {
        item["name"]
        for item in inspect(engine).get_unique_constraints("error_events")
        if item["column_names"] == ["fingerprint"]
    }


def test_migration_0011_round_trip(tmp_path: Path) -> None:
    """The plain fingerprint index becomes the unique constraint; the rollback puts the plain
    index back and takes the constraint away."""
    engine = _upgraded_engine(tmp_path)
    assert _fingerprint_unique_constraint(engine) == {"uq_error_events_fingerprint"}
    command.downgrade(_config(engine), "0010")
    assert _fingerprint_unique_constraint(engine) == set()
    assert "ix_error_events_fingerprint" in {
        index["name"] for index in inspect(engine).get_indexes("error_events")
    }
    upgrade_to_head(engine)
    _assert_complete(engine, ["error_events"])
    assert _fingerprint_unique_constraint(engine) == {"uq_error_events_fingerprint"}


def _same_error() -> ErrorEvent:
    return ErrorEvent(
        source="client",
        category="crash",
        severity="error",
        error_type="TypeError",
        message="boom",
        fingerprint="0" * 64,
        status="new",
        occurrence_count=1,
        first_seen_at=utcnow(),
        last_seen_at=utcnow(),
    )


def test_migration_0011_leaves_one_row_per_fingerprint(tmp_path: Path) -> None:
    """A database at 0010 may already hold two rows for one error: they are collapsed before
    the unique constraint is created, and a third with the same fingerprint is then refused."""
    engine = _upgraded_engine(tmp_path)
    command.downgrade(_config(engine), "0010")
    factory = make_session_factory(engine)
    with factory() as db:
        db.add_all([_same_error(), _same_error()])
        db.commit()
    with factory() as db:
        assert len(list(db.scalars(select(ErrorEvent)))) == 2

    upgrade_to_head(engine)

    with factory() as db:
        rows = list(db.scalars(select(ErrorEvent)))
        assert len(rows) == 1
        with pytest.raises(IntegrityError):
            db.add(_same_error())
            db.flush()


def test_an_install_event_is_deleted_with_its_record(tmp_path: Path) -> None:
    """``ondelete="CASCADE"`` is on the database, so a record deleted without its events (no
    ORM cascade is declared) still takes them along."""
    with Session(_upgraded_engine(tmp_path)) as db:
        now = utcnow()
        record = InstallRecord(
            install_key="k" * 64,
            app_version="0.11.1",
            os_name="linux",
            os_version="6.8",
            arch="x86_64",
            package="appimage",
            session="s" * 16,
            first_seen_at=now,
            last_seen_at=now,
        )
        db.add(record)
        db.flush()
        db.add(InstallEvent(install_record_id=record.id, event="install", to_version="0.11.1"))
        db.commit()
        events = select(func.count()).select_from(InstallEvent)
        assert db.scalar(events) == 1

        db.delete(record)
        db.commit()
        assert db.scalar(events) == 0


# -- limiters --------------------------------------------------------------------------------------


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_two_new_limiters_refuse_the_twenty_first_request_in_an_hour(tmp_path: Path) -> None:
    ctx = create_app(make_settings(tmp_path)).state.ctx
    assert INSTALL_RECORDS_PER_HOUR == ERROR_REPORTS_PER_HOUR == 20
    for limiter in (ctx.install_limiter, ctx.error_limiter):
        assert (limiter.limit, limiter.window_s) == (20, 3600.0)
        for _ in range(limiter.limit):
            limiter.check("a")
        with pytest.raises(ApiError) as refused:
            limiter.check("a")
        assert (refused.value.status_code, refused.value.code) == (429, "RATE_LIMITED")
        limiter.check("b")  # the count is per key

    # The window is the hour ``create_app`` asked for, not the limiter's one-minute default:
    # on a clock moved by hand, the twenty-first request is still refused a minute later.
    clock = _Clock()
    limiter = SlidingWindowLimiter(ctx.install_limiter.limit, ctx.install_limiter.window_s, clock)
    for _ in range(limiter.limit):
        limiter.check("a")
    clock.now = 61.0
    with pytest.raises(ApiError):
        limiter.check("a")
    clock.now = 3600.0
    limiter.check("a")


# -- body limits -----------------------------------------------------------------------------------


def _json_body(size: int) -> bytes:
    """A JSON object of exactly ``size`` bytes."""
    overhead = len(json.dumps({"pad": ""}))
    return json.dumps({"pad": "x" * (size - overhead)}).encode()


def _post(client: TestClient, path: str, size: int) -> Any:
    return client.post(path, content=_json_body(size), headers={"Content-Type": "application/json"})


def _assert_refused_by(response: Any, limit: int) -> None:
    """A 413 whose message names ``limit``: that limit, and no other, refused the body."""
    assert response.status_code == 413
    assert response.json() == {
        "error": "TOO_LARGE",
        "message": f"the request body is larger than the limit of {limit:,} bytes",
        "detail": {"code": "TOO_LARGE"},
    }


def test_a_body_over_two_kilobytes_to_the_installs_route_is_refused_before_any_handler(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    _assert_refused_by(_post(client, f"{API}/installs", 3 * 1024), 2048)

    # A body within the limit passes the middleware and reaches the handler: this one is not a
    # valid install report, so validation answers 422 where the limit would have said 413.
    allowed = _post(client, f"{API}/installs", 2048)
    assert allowed.status_code == 422
    assert allowed.json()["error"] == "VALIDATION_ERROR"


def test_a_body_over_sixty_four_kilobytes_to_the_errors_route_is_refused(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    _assert_refused_by(_post(client, f"{API}/errors", 65 * 1024), 65_536)
    # Within the limit the body reaches the handler, which refuses the unknown "pad" field.
    within = _post(client, f"{API}/errors", 65_536)
    assert within.status_code == 422
    assert within.json()["error"] == "VALIDATION_ERROR"


def test_other_paths_keep_the_default_body_limit(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    assert _post(client, f"{API}/nothing", 65 * 1024).status_code == 404
    _assert_refused_by(_post(client, f"{API}/nothing", JSON_BODY_LIMIT + 1), JSON_BODY_LIMIT)
    # The upload path keeps its own, larger limit.
    assert _post(client, f"{API}/datasets", 65 * 1024).status_code != 413


def test_a_path_limit_applies_ahead_of_the_upload_limit() -> None:
    sent: list[Any] = []

    async def send(message: Any) -> None:
        sent.append(message)

    async def receive() -> Any:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def unreachable(*_args: object) -> None:
        raise AssertionError("an oversized body must not reach the app")

    middleware = BodyLimitMiddleware(
        unreachable,
        default_limit=1_000,
        upload_limit=5_000,
        upload_path="/up",
        path_limits={"/up": 100},
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/up",
        "headers": [(b"content-length", b"101")],
    }
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


# -- settings --------------------------------------------------------------------------------------


def test_settings_read_the_current_release(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(env="test").current_release == ""
    assert Settings(env="test", current_release="0.12.0").current_release == "0.12.0"
    monkeypatch.setenv("CURVESTUDIO_CURRENT_RELEASE", "0.12.0")
    assert Settings(env="test").current_release == "0.12.0"


# -- model registration ----------------------------------------------------------------------------


def _in_a_fresh_interpreter(code: str) -> None:
    """Run ``code`` in a new Python process, failing with its stderr when it fails.

    Only a fresh process starts with the models unregistered. In this one ``create_app`` has
    registered them in every app a test built, so nothing here could tell a ``register_models``
    that does nothing, or an Alembic environment that never calls it, from the real thing.
    """
    run = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_register_models_puts_every_p5_table_on_the_metadata() -> None:
    _in_a_fresh_interpreter(
        """
        from curvestudio_api.modules import register_models
        from curvestudio_api.orm import Base

        p5_tables = {"install_records", "install_events", "error_events", "request_stats_daily"}
        assert not p5_tables & set(Base.metadata.tables)
        register_models()
        assert p5_tables <= set(Base.metadata.tables)
        """
    )


def test_alembic_find_no_difference_between_the_models_and_a_migrated_database(
    tmp_path: Path,
) -> None:
    """``command.check`` runs the real ``env.py`` and raises on any difference, so the
    environment is what registers the models here: nothing else in the process does."""
    _in_a_fresh_interpreter(
        f"""
        from alembic import command
        from curvestudio_api.migrations import _config, upgrade_to_head
        from sqlalchemy import create_engine

        engine = create_engine("sqlite:///{tmp_path}/api.db")
        upgrade_to_head(engine)
        command.check(_config(engine))
        """
    )
