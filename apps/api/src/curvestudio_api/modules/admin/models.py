"""The admin console's own table: request statistics per route and day."""

import uuid
from datetime import date

from sqlalchemy import JSON, Date, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from curvestudio_api.orm import Base, Timestamps


class RequestStatDaily(Timestamps, Base):
    """One route's requests on one day: how many, how many were errors, and how long they took
    (``histogram`` is a list of integers; the percentiles are in milliseconds)."""

    __tablename__ = "request_stats_daily"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    day: Mapped[date] = mapped_column(Date)
    route_template: Mapped[str] = mapped_column(String(200))
    count: Mapped[int] = mapped_column(Integer)
    errors: Mapped[int] = mapped_column(Integer)
    histogram: Mapped[list[int]] = mapped_column(JSON)
    p50_ms: Mapped[int] = mapped_column(Integer)
    p95_ms: Mapped[int] = mapped_column(Integer)
    max_ms: Mapped[int] = mapped_column(Integer)

    __table_args__ = (UniqueConstraint("day", "route_template"),)
