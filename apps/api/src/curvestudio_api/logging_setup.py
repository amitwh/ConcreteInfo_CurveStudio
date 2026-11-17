"""Structured JSON logging with a per-request ``request_id``; no bodies or headers."""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

from curvestudio_api.timeutil import to_ist_iso, utcnow

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

LOGGER_NAME = "curvestudio_api"
_FIELDS_ATTRIBUTE = "fields"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ``ts`` (IST), ``level``, ``logger``, ``message``, ``request_id``"""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": to_ist_iso(utcnow()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        extra = getattr(record, _FIELDS_ATTRIBUTE, None)
        if isinstance(extra, dict):
            entry.update(extra)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def log_fields(**fields: Any) -> dict[str, Any]:
    """``extra=`` argument that attaches ``fields`` to the JSON log line."""
    return {_FIELDS_ATTRIBUTE: fields}


def _install_request_id_factory() -> None:
    """Stamp every log record with the current request id when it is created."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_curvestudio_request_id", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    factory._curvestudio_request_id = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach the JSON handler to the ``curvestudio_api`` logger (idempotent)."""
    _install_request_id_factory()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(h, "_curvestudio_json", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._curvestudio_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
