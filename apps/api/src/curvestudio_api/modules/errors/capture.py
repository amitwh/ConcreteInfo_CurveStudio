"""The two server-side hooks that record errors nobody reports: the unhandled exception of a
request (:func:`capture_request_error`) and every ERROR record of the ``curvestudio_api``
logger (:class:`ErrorLogHandler`), which is how failures on the scheduler and housekeeping
threads are caught.

Neither ever raises, and neither waits long on the database: a failure to record is one line
on stderr, never a different response, and a capture gives up at :data:`CAPTURE_TIMEOUT_MS`.
The handler finds its database through a ``ContextVar`` that falls back to a module-level
default, which :func:`curvestudio_api.app.create_app` sets for the app it is building. A
thread that must write to a particular app's database binds its own factory
(:func:`bind_error_capture_sink`), so several apps can live in one process.
"""

import logging
import re
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from curvestudio_api.db import SessionFactory
from curvestudio_api.modules.errors.service import REDACTED, record_error_event, scrub_secrets

CAPTURED_FIELD = "error_captured"
"""The ``fields`` key that marks a log record as already recorded for the request it belongs
to, so :class:`ErrorLogHandler` does not record it a second time. The request hook sets it to
what it got back: the handler skips the record only when a row was really written."""

CAPTURE_TIMEOUT_MS = 2_000
"""How long the capture's own statements may wait on the database before giving up, so that a
request which ended in an error still gets its answer promptly (see :func:`_bounded_wait`)."""

SERVER_SOURCE = "server"
FATAL_SEVERITY = "fatal"
ERROR_SEVERITY = "error"
FAILURE_PREFIX = "error capture failed"
"""How the one stderr line of a failed capture starts."""

_BOUND_PARAMETERS_RE = re.compile(
    r"\n\[parameters: .*?(?=\n\(Background on this error|\Z)", re.DOTALL
)

_bound_sink: ContextVar[SessionFactory | None] = ContextVar("error_capture_sink", default=None)
_default_sink: SessionFactory | None = None
_in_capture: ContextVar[bool] = ContextVar("error_capture_in_progress", default=False)


def set_default_error_capture_sink(session_factory: SessionFactory) -> None:
    """Bind ``session_factory`` for every context that has not bound one of its own.
    :func:`curvestudio_api.app.create_app` calls this for the app it is building."""
    global _default_sink
    _default_sink = session_factory


@contextmanager
def error_capture_sink(session_factory: SessionFactory) -> Iterator[None]:
    """Bind ``session_factory`` for the block."""
    token = _bound_sink.set(session_factory)
    try:
        yield
    finally:
        _bound_sink.reset(token)


def bind_error_capture_sink(session_factory: SessionFactory) -> None:
    """Bind ``session_factory`` for the rest of the current thread. A thread starts with an
    empty context, so a background thread binds the factory of the app that owns it."""
    _bound_sink.set(session_factory)


def _current_sink() -> SessionFactory | None:
    return _bound_sink.get() or _default_sink


def _category(exc: BaseException | None) -> str:
    return "sql" if isinstance(exc, SQLAlchemyError) else "crash"


def _traceback_of(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


def _without_bound_parameters(exc: BaseException | None, text: str) -> str:
    """A database error's text without the block of its bound parameters.

    ``str()`` of an ``SQLAlchemyError`` echoes them, and they can hold an e-mail address (a
    failed ``INSERT INTO users``); an error event never identifies an account. The statement
    itself stays: it is what tells the failures apart.
    """
    if isinstance(exc, SQLAlchemyError):
        return _BOUND_PARAMETERS_RE.sub(f"\n[parameters: {REDACTED}]", text)
    return text


def _report_failure(what: str, failure: BaseException) -> None:
    """One line on stderr; never raises."""
    text = _without_bound_parameters(failure, str(failure))
    detail = scrub_secrets(" ".join(text.split()))[:300]
    line = f"{FAILURE_PREFIX} while recording {what}: {type(failure).__name__}: {detail}"
    # stderr may be detached (``None``), closed (``ValueError``) or broken (``OSError``): there
    # is nowhere left to say it, and the caller must not fail because of that.
    stream = sys.stderr
    if stream is not None:
        with suppress(OSError, ValueError):
            stream.write(line + "\n")


@contextmanager
def _bounded_wait(db: Session) -> Iterator[None]:
    """Keep the capture from waiting long on the database.

    PostgreSQL: ``SET LOCAL statement_timeout`` ends any statement of this transaction that
    runs past :data:`CAPTURE_TIMEOUT_MS`. SQLite has no statement timeout: ``PRAGMA
    busy_timeout`` bounds the wait for the write lock, and as it belongs to the pooled
    connection, the previous value is put back afterwards, so the application's other sessions
    keep the engine's own wait. Other dialects get no statement, and so no limit.
    """
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        db.execute(text(f"SET LOCAL statement_timeout = {CAPTURE_TIMEOUT_MS}"))
        yield
    elif dialect == "sqlite":
        previous = db.scalar(text("PRAGMA busy_timeout"))
        db.execute(text(f"PRAGMA busy_timeout = {CAPTURE_TIMEOUT_MS}"))
        try:
            yield
        finally:
            # A statement that failed leaves the session unusable until it is rolled back; the
            # capture has committed or lost its one row by now either way.
            db.rollback()
            db.execute(text(f"PRAGMA busy_timeout = {previous}"))
    else:
        yield


def _write(session_factory: SessionFactory, **fields: Any) -> None:
    """Record one server event on a fresh session that gives up on the database quickly."""
    with session_factory() as db, _bounded_wait(db):
        record_error_event(db, source=SERVER_SOURCE, **fields)
        db.commit()


def capture_request_error(
    session_factory: SessionFactory,
    *,
    exc: BaseException,
    method: str,
    route: str | None,
    request_id: str | None,
    user_agent: str = "",
) -> bool:
    """Record ``exc``, the exception a request ended on, as a fatal server event with status
    500: category ``sql`` for a database error, ``crash`` for anything else.

    Commits on its own session, and waits on the database for at most
    :data:`CAPTURE_TIMEOUT_MS` per statement. Never raises: on any failure it writes one line
    to stderr and returns ``False``; ``True`` means a row was written.
    """
    try:
        _write(
            session_factory,
            category=_category(exc),
            severity=FATAL_SEVERITY,
            error_type=type(exc).__name__,
            message=_without_bound_parameters(exc, str(exc) or type(exc).__name__),
            stack=_without_bound_parameters(exc, _traceback_of(exc)),
            route=route,
            method=method,
            status_code=500,
            request_id=request_id,
            user_agent=user_agent or None,
        )
        return True
    except Exception as failure:
        _report_failure("a request error", failure)
        return False


def _record(sink: SessionFactory, record: logging.LogRecord) -> None:
    exc = record.exc_info[1] if record.exc_info else None
    _write(
        sink,
        category=_category(exc),
        severity=ERROR_SEVERITY,
        error_type=type(exc).__name__ if exc is not None else "Error",
        message=_without_bound_parameters(exc, record.getMessage()),
        stack=_without_bound_parameters(exc, _traceback_of(exc)) if exc is not None else None,
        request_id=getattr(record, "request_id", None),
    )


class ErrorLogHandler(logging.Handler):
    """Records every ERROR record of the ``curvestudio_api`` logger as a server error event.

    ``emit`` does nothing for a record that is below ERROR, is already inside a capture (a
    re-entry guard), carries :data:`CAPTURED_FIELD` as true (the request hook wrote it), or has
    no session factory bound to write to. It never raises: a failure of its own is one line on
    stderr.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR or _in_capture.get():
            return
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields.get(CAPTURED_FIELD) is True:
            return
        sink = _current_sink()
        if sink is None:
            return
        token = _in_capture.set(True)
        try:
            _record(sink, record)
        except Exception as failure:
            _report_failure("an error log record", failure)
        finally:
            _in_capture.reset(token)


def install_error_log_handler(logger: logging.Logger) -> ErrorLogHandler:
    """Attach the capture handler to ``logger`` unless it already has one; returns it. Flagged
    like the JSON handler in ``logging_setup``, so many ``create_app`` calls in one process
    leave one handler."""
    for handler in logger.handlers:
        if getattr(handler, "_curvestudio_error_capture", False):
            return cast(ErrorLogHandler, handler)
    handler = ErrorLogHandler()
    handler._curvestudio_error_capture = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return handler
