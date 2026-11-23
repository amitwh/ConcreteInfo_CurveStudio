"""The error-event table. An error event never identifies an account, and the table has no
column for one: a row holds the failure and the request that carried it, never a person."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from curvestudio_api.db import UtcDateTime
from curvestudio_api.orm import Base, Timestamps


class ErrorEvent(Timestamps, Base):
    """A reported failure, with the request that carried it where there was one."""

    __tablename__ = "error_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(8))
    category: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(8))
    error_type: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    # Unique: one row per error. Two callers reporting the same new error at once both insert;
    # the second is turned away here and continues as a repeat of the first (service.py).
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    stack: Mapped[str | None] = mapped_column(Text, default=None)
    route: Mapped[str | None] = mapped_column(String(200), default=None)
    method: Mapped[str | None] = mapped_column(String(8), default=None)
    status_code: Mapped[int | None] = mapped_column(Integer, default=None)
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    user_agent: Mapped[str | None] = mapped_column(String(500), default=None)
    app_version: Mapped[str | None] = mapped_column(String(40), default=None)
    status: Mapped[str] = mapped_column(String(16), default="new", index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True)
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    console_log: Mapped[str | None] = mapped_column(Text, default=None)
