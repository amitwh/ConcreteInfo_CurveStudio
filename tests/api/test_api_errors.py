"""The error-events domain: the report endpoint, the two server capture hooks and the roll-up."""

import asyncio
import json
import logging
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from api_test_helpers import make_settings
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from curvestudio_api import housekeeping
from curvestudio_api.app import create_app
from curvestudio_api.db import SessionFactory
from curvestudio_api.deps import Ctx
from curvestudio_api.logging_setup import LOGGER_NAME, log_fields
from curvestudio_api.middleware import RequestContextMiddleware
from curvestudio_api.modules.errors.capture import (
    CAPTURE_TIMEOUT_MS,
    CAPTURED_FIELD,
    ErrorLogHandler,
    capture_request_error,
    error_capture_sink,
    install_error_log_handler,
)
from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.modules.errors.service import record_error_event
from curvestudio_api.routers import system as system_module
from curvestudio_api.timeutil import utcnow

API = "/api/v1"
UUID = "3f2a9c1e-7b4d-4e8a-9c21-5d6f7a8b9c0d"
BEARER = "Authorization: Bearer s3cr3t-access-token-value"
PASSWORD = "password=hunter2-secret"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
ENVELOPE = {"error": "INTERNAL_ERROR", "message": "an internal error occurred", "detail": {}}


# -- helpers ---------------------------------------------------------------------------------------


def event(**overrides: Any) -> dict[str, Any]:
    """One client event as the network sends it."""
    body: dict[str, Any] = {
        "category": "crash",
        "error_type": "TypeError",
        "message": "Cannot read properties of undefined (reading 'x')",
        "stack": "TypeError: Cannot read properties of undefined\n    at render (app.js:1:2345)",
    }
    body.update(overrides)
    return body


def rows(client: TestClient) -> list[ErrorEvent]:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return list(db.scalars(select(ErrorEvent).order_by(ErrorEvent.created_at, ErrorEvent.id)))


def wait_for_rows(client: TestClient, timeout: float = 10.0) -> list[ErrorEvent]:
    """The rows of ``client``'s database once at least one exists (a background thread writes
    them), or whatever is there when ``timeout`` runs out."""
    deadline = time.monotonic() + timeout
    while True:
        found = rows(client)
        if found or time.monotonic() > deadline:
            return found
        time.sleep(0.02)


def mount_boom(client: TestClient, path: str = "/boom") -> str:
    """A scratch route that raises, as the admin-guard tests mount theirs."""

    @client.app.get(path)  # type: ignore[attr-defined]
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    return path


def broken_factory() -> Session:
    raise OperationalError("INSERT INTO error_events", {}, Exception("database is locked"))


def unhandled_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "unhandled error"]


# -- the request hook ------------------------------------------------------------------------------


def test_an_unhandled_exception_records_one_event_and_keeps_the_envelope(
    client_factory: Callable[..., TestClient], caplog: pytest.LogCaptureFixture
) -> None:
    client = client_factory()
    mount_boom(client)

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        response = client.get("/boom", headers={"X-Request-ID": "req-42", "User-Agent": "UA/1.0"})
    assert response.status_code == 500
    assert response.json() == ENVELOPE
    assert response.headers["x-request-id"] == "req-42"
    assert "secret internal detail" not in response.text

    found = rows(client)
    assert len(found) == 1  # the hook wrote it; the log handler did not write it again
    row = found[0]
    assert row.request_id == "req-42"
    assert row.route == "/boom"
    assert row.method == "GET"
    assert row.status_code == 500
    assert row.source == "server"
    assert row.severity == "fatal"
    assert row.category == "crash"
    assert row.error_type == "RuntimeError"
    assert row.message == "secret internal detail"
    assert row.stack is not None and "RuntimeError: secret internal detail" in row.stack
    assert row.user_agent == "UA/1.0"
    assert row.status == "new"
    assert row.occurrence_count == 1

    recorded = unhandled_records(caplog)
    assert len(recorded) == 1
    assert recorded[0].exc_info is not None and recorded[0].exc_info[0] is RuntimeError
    assert recorded[0].fields[CAPTURED_FIELD] is True  # type: ignore[attr-defined]


def test_a_uuid_in_the_route_is_stored_as_a_template(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    # a literal UUID in the route's own path: the template replaces it, as it would a client's
    path = mount_boom(client, f"/boom/{UUID}")
    assert client.get(path).status_code == 500
    assert [row.route for row in rows(client)] == ["/boom/{id}"]

    template = mount_boom(client, "/boom/{item_id}/parts")
    assert client.get(f"{template.replace('{item_id}', UUID)}").status_code == 500
    assert [row.route for row in rows(client)] == ["/boom/{id}", "/boom/{item_id}/parts"]


def test_a_crash_in_a_real_api_route_is_stored_with_its_full_template(
    client_factory: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API's routes sit under a router with the ``/api/v1`` prefix, and this FastAPI keeps a
    matched route's own path without it. The stored route is the full template."""

    def _broken(ctx: Ctx) -> bool:
        raise RuntimeError("boom inside a real handler")

    monkeypatch.setattr(system_module, "_database_up", _broken)
    client = client_factory()

    assert client.get(f"{API}/health").status_code == 500
    assert [row.route for row in rows(client)] == [f"{API}/health"]


def test_capture_failure_never_changes_the_response(
    client_factory: Callable[..., TestClient],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    client = client_factory()
    mount_boom(client)
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    monkeypatch.setattr(client.app.state.ctx, "session_factory", broken_factory)  # type: ignore[attr-defined]

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        response = client.get("/boom")
    assert response.status_code == 500
    assert response.json() == ENVELOPE
    assert len(response.headers["x-request-id"]) == 32

    recorded = unhandled_records(caplog)
    assert len(recorded) == 1
    record = recorded[0]
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    rendered = "".join(traceback.format_exception(*record.exc_info))
    assert "RuntimeError: secret internal detail" in rendered
    assert record.fields[CAPTURED_FIELD] is False  # type: ignore[attr-defined]

    # The hook's session was the broken one: nothing came from it, and one line said so. The
    # record was not marked as captured, so the log handler had its turn through the module
    # default, which is this app's real factory: one row, with the handler's level of detail.
    with factory() as db:
        found = list(db.scalars(select(ErrorEvent)))
    assert [(r.severity, r.error_type, r.status_code) for r in found] == [
        ("error", "RuntimeError", None)
    ]
    assert "error capture failed" in capfd.readouterr().err


def test_capture_request_error_classifies_sql_errors_and_never_raises(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    sql_error = OperationalError("SELECT 1", {}, Exception("database is locked"))
    written = capture_request_error(
        factory, exc=sql_error, method="GET", route="/x", request_id="r1"
    )
    assert written is True
    assert (
        capture_request_error(
            factory, exc=ValueError("boom"), method="GET", route=None, request_id=None
        )
        is True
    )
    assert [(r.category, r.error_type) for r in rows(client)] == [
        ("sql", "OperationalError"),
        ("crash", "ValueError"),
    ]
    assert (
        capture_request_error(
            broken_factory, exc=ValueError("boom"), method="GET", route="/x", request_id=None
        )
        is False
    )


def test_a_database_error_never_keeps_its_bound_parameters(
    client_factory: Callable[..., TestClient],
) -> None:
    """``str()`` of a database error echoes the statement's parameters, which can hold an
    e-mail address: a failed ``INSERT INTO users`` must not put one in the table. The capture
    drops that block and keeps the statement."""
    client = client_factory()
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    failure = IntegrityError(
        "INSERT INTO users (email, name) VALUES (:email, :name)",
        {"email": "bob@example.org", "name": "Bob"},
        Exception("UNIQUE constraint failed: users.email"),
    )
    assert (
        capture_request_error(
            factory, exc=failure, method="POST", route="/auth/register", request_id=None
        )
        is True
    )
    with error_capture_sink(factory):
        try:
            raise failure
        except IntegrityError:
            logging.getLogger("curvestudio_api.jobs").exception("could not record a row")

    found = rows(client)
    assert len(found) == 2
    for row in found:
        stored = row.message + (row.stack or "")
        assert "bob@example.org" not in stored
        assert "'name': 'Bob'" not in stored
        assert "[parameters: [REDACTED]]" in stored
        assert "[SQL: INSERT INTO users" in stored
        assert row.category == "sql"


def test_a_failed_capture_with_no_stderr_still_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one stderr line is best effort: a detached stderr must not turn a failed capture
    into an exception."""
    monkeypatch.setattr(sys, "stderr", None)
    assert (
        capture_request_error(
            broken_factory, exc=ValueError("boom"), method="GET", route=None, request_id=None
        )
        is False
    )
    with error_capture_sink(broken_factory):
        logging.getLogger("curvestudio_api.jobs").exception("job scheduler tick failed")


def test_a_scope_without_the_context_still_gets_the_envelope() -> None:
    """The hook finds no ``Context`` on a bare ASGI scope: nothing is recorded, and the
    response is the envelope all the same."""

    async def boom(scope: Any, receive: Any, send: Any) -> None:
        raise RuntimeError("secret internal detail")

    async def receive() -> Any:
        return {"type": "http.request", "body": b"", "more_body": False}

    messages: list[Any] = []

    async def send(message: Any) -> None:
        messages.append(message)

    scope: Any = {"type": "http", "method": "GET", "path": "/boom", "headers": []}
    with error_capture_sink(broken_factory):  # the log handler's own attempt goes nowhere
        asyncio.run(RequestContextMiddleware(boom)(scope, receive, send))
    started, body = messages
    assert started["status"] == 500
    assert dict(started["headers"])[b"x-request-id"]  # type: ignore[index]
    assert json.loads(body["body"]) == ENVELOPE


def _noting_factory(engine: Any, seen: list[int]) -> SessionFactory:
    """Sessions on ``engine`` that, as they close, note their own connection's ``PRAGMA
    busy_timeout``: the value the capture left on the very connection it used."""

    class Noting(Session):
        def close(self) -> None:
            seen.append(int(self.connection().scalar(text("PRAGMA busy_timeout"))))
            super().close()

    return lambda: Noting(engine)


ENGINE_BUSY_TIMEOUT_MS = 30_000  # what make_engine gives every connection


def test_a_capture_on_a_healthy_database_writes_its_row_at_once(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    seen: list[int] = []
    factory = _noting_factory(client.app.state.ctx.engine, seen)  # type: ignore[attr-defined]

    started = time.monotonic()
    written = capture_request_error(
        factory, exc=ValueError("boom"), method="GET", route="/boom", request_id="r-healthy"
    )

    assert written is True
    assert time.monotonic() - started < CAPTURE_TIMEOUT_MS / 1000
    assert [(r.error_type, r.request_id) for r in rows(client)] == [("ValueError", "r-healthy")]
    assert seen == [ENGINE_BUSY_TIMEOUT_MS]  # the capture's limit did not outlive its session


def test_a_capture_gives_up_quickly_when_the_database_is_locked(
    client_factory: Callable[..., TestClient], capfd: pytest.CaptureFixture[str]
) -> None:
    """Another connection holds the write lock. The capture waits for it only up to its limit,
    then gives up: the request still gets its envelope after seconds, not the default half
    minute, and the pooled connection keeps the engine's own wait."""
    client = client_factory()
    mount_boom(client)
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    seen: list[int] = []
    factory = _noting_factory(ctx.engine, seen)
    blocker = ctx.engine.connect()
    blocker.exec_driver_sql("BEGIN IMMEDIATE")  # a write, begun and kept open
    try:
        started = time.monotonic()
        written = capture_request_error(
            factory, exc=ValueError("boom"), method="GET", route="/boom", request_id=None
        )
        seconds = time.monotonic() - started
        started = time.monotonic()
        response = client.get("/boom")
        request_seconds = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()

    assert written is False
    # it really waited for the lock, and then gave up at its limit rather than the engine's
    assert CAPTURE_TIMEOUT_MS / 1000 <= seconds < 3 * CAPTURE_TIMEOUT_MS / 1000
    assert response.status_code == 500
    assert response.json() == ENVELOPE
    assert request_seconds < 6 * CAPTURE_TIMEOUT_MS / 1000
    assert "error capture failed" in capfd.readouterr().err
    assert seen == [ENGINE_BUSY_TIMEOUT_MS]


# -- the log handler -------------------------------------------------------------------------------


def test_a_background_error_is_recorded_by_the_log_handler(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    jobs = logging.getLogger("curvestudio_api.jobs")
    health = logging.getLogger("curvestudio_api.health")

    with error_capture_sink(factory):
        try:
            raise RuntimeError("scheduler boom")
        except RuntimeError:
            jobs.exception("job scheduler tick failed")
        try:
            raise OperationalError("SELECT 1", {}, Exception("database is locked"))
        except OperationalError:
            health.exception("database check failed")
        jobs.error("already recorded", extra=log_fields(**{CAPTURED_FIELD: True}))
        jobs.info("not an error")
        jobs.error("no traceback")

    found = rows(client)
    assert [(r.message, r.category, r.error_type) for r in found] == [
        ("job scheduler tick failed", "crash", "RuntimeError"),
        ("database check failed", "sql", "OperationalError"),
        ("no traceback", "crash", "Error"),
    ]
    row = found[0]
    assert row.source == "server"
    assert row.severity == "error"
    assert row.stack is not None and "RuntimeError: scheduler boom" in row.stack
    assert row.status_code is None and row.route is None and row.request_id is None
    assert row.method is None and row.user_agent is None and row.context_json is None
    assert found[2].stack is None


def test_a_failure_on_the_scheduler_thread_is_recorded_into_its_own_app(
    client_factory: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    earlier = client_factory(make_settings(tmp_path / "earlier"))
    # built last, so the module's default sink is this app's database: a thread that did not
    # bind its own factory would write here
    later = client_factory(make_settings(tmp_path / "later"))
    jobs = earlier.app.state.ctx.jobs  # type: ignore[attr-defined]
    original = jobs._dispatch
    failures: list[int] = []

    def fail_once() -> None:
        if not failures:
            failures.append(1)
            raise RuntimeError("dispatch boom")
        original()

    monkeypatch.setattr(jobs, "_dispatch", fail_once)
    try:
        found = wait_for_rows(earlier)
    finally:
        monkeypatch.setattr(jobs, "_dispatch", original)

    assert len(found) == 1
    assert found[0].message == "job scheduler tick failed"
    assert found[0].stack is not None and "RuntimeError: dispatch boom" in found[0].stack
    assert found[0].occurrence_count == 1
    assert rows(later) == []


def test_a_failure_on_the_housekeeping_thread_is_recorded_into_its_own_app(
    client_factory: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    earlier = client_factory(make_settings(tmp_path / "earlier", housekeeping_interval_s=0.05))
    later = client_factory(make_settings(tmp_path / "later"))  # the module's default sink
    original = housekeeping.purge
    failures: list[int] = []

    def fail_once(session_factory: Any) -> Any:
        if not failures:
            failures.append(1)
            raise RuntimeError("purge boom")
        return original(session_factory)

    monkeypatch.setattr(housekeeping, "purge", fail_once)
    try:
        found = wait_for_rows(earlier)
    finally:
        monkeypatch.setattr(housekeeping, "purge", original)

    assert [r.message for r in found] == ["housekeeping failed"]
    assert found[0].stack is not None and "RuntimeError: purge boom" in found[0].stack
    assert rows(later) == []


def test_the_handler_survives_a_database_that_is_down(
    client_factory: Callable[..., TestClient], capfd: pytest.CaptureFixture[str]
) -> None:
    client = client_factory()

    with error_capture_sink(broken_factory):
        logging.getLogger("curvestudio_api.jobs").exception("job scheduler tick failed")
        logging.getLogger("curvestudio_api.jobs").error("no traceback either")

    assert rows(client) == []
    err = capfd.readouterr().err
    assert err.count("error capture failed") == 2
    assert "database is locked" in err


def test_installing_the_handler_twice_adds_one(
    client_factory: Callable[..., TestClient],
) -> None:
    client_factory()
    client_factory()
    real = logging.getLogger(LOGGER_NAME)
    installed = [h for h in real.handlers if isinstance(h, ErrorLogHandler)]
    assert len(installed) == 1

    scratch = logging.getLogger("curvestudio_api.scratch-capture")
    first = install_error_log_handler(scratch)
    try:
        assert install_error_log_handler(scratch) is first
        assert [h for h in scratch.handlers if isinstance(h, ErrorLogHandler)] == [first]
    finally:
        scratch.removeHandler(first)


# -- the roll-up -----------------------------------------------------------------------------------


REPEAT = {
    "source": "client",
    "category": "crash",
    "severity": "error",
    "error_type": "TypeError",
    "message": "boom",
    "route": "/x",
}


def lookup_misses_once(db: Session) -> None:
    """Make the service's first lookup of a fingerprint miss, as it does when another caller
    committed the same error a moment later: the row is already there."""

    real_scalar = db.scalar
    misses = [1]

    def scalar(statement: Any, *args: Any, **kwargs: Any) -> Any:
        if misses:
            misses.pop()
            return None
        return real_scalar(statement, *args, **kwargs)

    db.scalar = scalar  # type: ignore[method-assign]


def ordered_rows(factory: SessionFactory) -> list[ErrorEvent]:
    with factory() as db:
        return list(db.scalars(select(ErrorEvent).order_by(ErrorEvent.created_at, ErrorEvent.id)))


def test_the_same_new_error_reported_by_two_callers_at_once_is_one_row(
    client_factory: Callable[..., TestClient],
) -> None:
    """Both callers look the fingerprint up before either has committed, so both insert. The
    unique fingerprint lets one through; the other becomes a repeat of it, and the work it had
    already done in its transaction is kept."""
    client = client_factory()
    factory: SessionFactory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    with factory() as first:
        won, how = record_error_event(first, **REPEAT)  # type: ignore[arg-type]
        first.commit()
        assert how == "created"

    with factory() as second:
        earlier, how = record_error_event(second, **{**REPEAT, "message": "another error"})  # type: ignore[arg-type]
        assert how == "created" and earlier.id != won.id  # pending in this transaction
        lookup_misses_once(second)
        lost, how = record_error_event(second, **REPEAT)  # type: ignore[arg-type]
        assert how == "incremented"
        assert lost.id == won.id
        second.commit()

    assert [(r.message, r.occurrence_count) for r in ordered_rows(factory)] == [
        ("boom", 2),
        ("another error", 1),
    ]


def test_a_resolved_error_reported_twice_at_once_is_one_new_row_two_further_occurrences_on(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    factory: SessionFactory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    with factory() as db:
        resolved, _ = record_error_event(db, **REPEAT)  # type: ignore[arg-type]
        resolved.status = "resolved"
        db.commit()
        resolved_id = resolved.id

    outcomes = []
    for _ in range(2):  # two callers, each sure the fingerprint is not there yet
        with factory() as db:
            lookup_misses_once(db)
            _, how = record_error_event(db, **REPEAT)  # type: ignore[arg-type]
            db.commit()
            outcomes.append(how)

    assert outcomes == ["reopened", "incremented"]
    found = ordered_rows(factory)
    assert [r.id for r in found] == [resolved_id]
    assert found[0].status == "new"
    assert found[0].occurrence_count == 3


def test_a_repeat_increments_and_a_resolved_event_reopens(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]

    with factory() as db:
        created, how = record_error_event(db, **REPEAT, console_log="first")  # type: ignore[arg-type]
        db.commit()
        assert how == "created"
        assert created.status == "new" and created.occurrence_count == 1
        assert created.first_seen_at == created.last_seen_at
        first_seen = created.first_seen_at

    later = utcnow() + timedelta(minutes=5)
    with factory() as db:
        event, how = record_error_event(db, **REPEAT, console_log="second", now=later)  # type: ignore[arg-type]
        db.commit()
        assert how == "incremented"
        assert event.id == created.id
        assert event.occurrence_count == 2
        assert event.first_seen_at == first_seen and event.last_seen_at == later
        assert event.console_log == "second"  # refreshed from the latest
        assert event.status == "new"

    event_id = event.id
    for closed in ("resolved", "ignored"):
        with factory() as db:
            stored = db.get(ErrorEvent, event_id)
            assert stored is not None
            stored.status = closed
            db.commit()
        with factory() as db:
            event, how = record_error_event(db, **REPEAT)  # type: ignore[arg-type]
            db.commit()
            assert how == "reopened"
            assert event.status == "new"
    assert event.occurrence_count == 4


def test_an_acknowledged_event_stays_acknowledged(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    with factory() as db:
        record_error_event(db, **REPEAT)  # type: ignore[arg-type]
        db.commit()
        event = db.scalar(select(ErrorEvent))
        assert event is not None
        event.status = "acknowledged"
        db.commit()

    with factory() as db:
        event, how = record_error_event(db, **REPEAT)  # type: ignore[arg-type]
        db.commit()
        assert how == "incremented"
        assert event.status == "acknowledged"
        assert event.occurrence_count == 2


def test_fields_are_cut_to_their_caps_and_blank_optionals_become_none(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    factory = client.app.state.ctx.session_factory  # type: ignore[attr-defined]
    fixed = datetime(2026, 11, 23, 10, 30, tzinfo=UTC)
    with factory() as db:
        event, _ = record_error_event(
            db,
            source="client",
            category="ui",
            severity="error",
            error_type="T" * 250,
            message="m" * 2_500,
            stack="s" * 100_500,
            route="/r" * 300,
            method="PATCH EXTRA",
            request_id="q" * 100,
            user_agent="u" * 600,
            app_version="v" * 50,
            console_log="c" * 5_000,
            context={"blob": "x" * 30_000, "small": "keep me"},
            now=fixed,
        )
        db.commit()
        assert event.error_type == "T" * 200
        assert len(event.message) == 2_000
        assert len(event.stack or "") == 100_000
        assert event.route == "/r" * 100
        assert event.method == "PATCH EX"
        assert event.request_id == "q" * 64
        assert len(event.user_agent or "") == 500
        assert event.app_version == "v" * 40
        assert len(event.console_log or "") == 4_000
        assert len(json.dumps(event.context_json)) <= 10 * 1024
        assert event.context_json is not None and event.context_json["small"] == "keep me"
        assert event.first_seen_at == fixed and event.last_seen_at == fixed

        blank, _ = record_error_event(
            db,
            source="server",
            category="crash",
            severity="fatal",
            error_type="Error",
            message="x",
            stack="  ",
            route="",
            request_id="",
            user_agent="   ",
            app_version="",
            console_log="\n",
        )
        db.commit()
        for field in ("stack", "route", "request_id", "user_agent", "app_version", "console_log"):
            assert getattr(blank, field) is None, field
        assert blank.context_json is None


# -- the report endpoint ---------------------------------------------------------------------------


def test_secrets_never_reach_the_database(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    response = client.post(
        f"{API}/errors",
        json=event(
            message=f"request failed with {BEARER}",
            stack=f"Error: boom\n    at save (app.js:1:2) {JWT}",
            console_log=f"console: {PASSWORD}",
            context={"form": PASSWORD, "headers": f"{BEARER}", "query": f"token={JWT}"},
        ),
    )
    assert response.status_code == 204
    (row,) = rows(client)
    stored = "\n".join(
        [row.message, row.stack or "", row.console_log or "", json.dumps(row.context_json)]
    )
    for secret in ("s3cr3t-access-token-value", "hunter2-secret", JWT):
        assert secret not in stored
    assert row.message == "request failed with Authorization: Bearer [REDACTED]"
    assert row.context_json is not None and row.context_json["form"] == "password=[REDACTED]"


def test_the_report_endpoint_takes_one_event_or_ten(
    client_factory: Callable[..., TestClient], tmp_path: Any
) -> None:
    client = client_factory()
    one = client.post(f"{API}/errors", json=event())
    assert one.status_code == 204
    assert one.content == b""
    assert [(r.source, r.category, r.severity) for r in rows(client)] == [
        ("client", "crash", "error")
    ]

    ten = client.post(f"{API}/errors", json={"events": [event(message=f"m{i}") for i in range(10)]})
    assert ten.status_code == 204
    assert len(rows(client)) == 11

    eleven = client.post(
        f"{API}/errors", json={"events": [event(message=f"e{i}") for i in range(11)]}
    )
    assert eleven.status_code == 422
    assert eleven.json()["error"] == "VALIDATION_ERROR"
    assert client.post(f"{API}/errors", json={"events": []}).status_code == 422

    smuggled = client.post(f"{API}/errors", json=event(source="server"))
    assert smuggled.status_code == 422
    assert client.post(f"{API}/errors", json={"events": [event(source="server")]}).status_code == (
        422
    )

    long_console = client.post(f"{API}/errors", json=event(console_log="x" * 4_001))
    assert long_console.status_code == 422
    big_context = client.post(f"{API}/errors", json=event(context={"blob": "x" * 11_000}))
    assert big_context.status_code == 422
    assert len(rows(client)) == 11

    too_big = client.post(
        f"{API}/errors",
        content=b'{"pad": "' + b"x" * 70_000 + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert too_big.status_code == 413
    assert too_big.json()["error"] == "TOO_LARGE"
    assert len(rows(client)) == 11

    # the real limit: an app whose error limiter was not widened for the tests
    strict = create_app(make_settings(tmp_path / "strict"))
    with TestClient(strict) as tight:
        for _ in range(20):
            assert tight.post(f"{API}/errors", json=event()).status_code == 204
        refused = tight.post(f"{API}/errors", json=event())
        assert refused.status_code == 429
        assert refused.json()["error"] == "RATE_LIMITED"
        assert "retry-after" in refused.headers


def test_a_batch_of_ten_identical_events_is_one_row_counted_ten_times(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    response = client.post(f"{API}/errors", json={"events": [event() for _ in range(10)]})
    assert response.status_code == 204
    found = rows(client)
    assert len(found) == 1
    assert found[0].occurrence_count == 10


def test_a_client_route_with_a_uuid_is_stored_as_a_template_and_the_description_is_kept(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    response = client.post(
        f"{API}/errors",
        json=event(
            route=f"/datasets/{UUID}/fits?tab=1",
            description="I clicked Save and the page went blank",
            context={"tab": "fits"},
            severity="fatal",
            category="ui",
            user_agent="Mozilla/5.0 (X11)",
            app_version="1.2.3",
            request_id="abc-123",
        ),
    )
    assert response.status_code == 204
    (row,) = rows(client)
    assert row.route == "/datasets/{id}/fits"
    assert row.context_json == {
        "tab": "fits",
        "description": "I clicked Save and the page went blank",
    }
    assert row.severity == "fatal"
    assert row.category == "ui"
    assert row.user_agent == "Mozilla/5.0 (X11)"
    assert row.app_version == "1.2.3"
    assert row.request_id == "abc-123"
    assert row.source == "client"
