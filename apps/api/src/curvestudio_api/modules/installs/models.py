"""The install-record tables. Neither holds an address or an account: an installation is known
by its ``install_key`` alone, so no row can be tied back to a person."""

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from curvestudio_api.db import UtcDateTime
from curvestudio_api.orm import Base, Timestamps
from curvestudio_api.timeutil import utcnow


class InstallRecord(Timestamps, Base):
    """One desktop installation, by release, operating system and architecture."""

    __tablename__ = "install_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    install_key: Mapped[str] = mapped_column(String(64), unique=True)
    app_version: Mapped[str] = mapped_column(String(40))
    os_name: Mapped[str] = mapped_column(String(16))
    os_version: Mapped[str] = mapped_column(String(40))
    arch: Mapped[str] = mapped_column(String(16))
    package: Mapped[str] = mapped_column(String(20))
    session: Mapped[str] = mapped_column(String(16))
    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime)


class InstallEvent(Base):
    """One event an installation reported (``event`` names it), with the releases it moved
    between where that applies. It goes when its record goes."""

    __tablename__ = "install_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    install_record_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("install_records.id", ondelete="CASCADE"), index=True
    )
    event: Mapped[str] = mapped_column(String(16))
    from_version: Mapped[str | None] = mapped_column(String(40), default=None)
    to_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
