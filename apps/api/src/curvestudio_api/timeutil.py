"""Time handling: stored UTC-aware, serialised as ISO 8601 with the IST (+05:30) offset."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Annotated

from pydantic import PlainSerializer

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def utcnow() -> datetime:
    """The current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_ist_iso(value: datetime) -> str:
    """``value`` as ISO 8601 with the +05:30 offset. Naive values are taken to be UTC."""
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(IST).isoformat()


IstDatetime = Annotated[datetime, PlainSerializer(to_ist_iso, return_type=str, when_used="always")]
"""A datetime that always serialises with the +05:30 offset (also in the OpenAPI schema)."""
