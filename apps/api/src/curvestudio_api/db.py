"""Engine and session plumbing (SQLAlchemy 2, SQLite for tests and small installs, Postgres)."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Engine, TypeDecorator, create_engine, event
from sqlalchemy.engine import Dialect, make_url
from sqlalchemy.orm import Session, sessionmaker

from curvestudio_api.settings import Settings

SessionFactory = Callable[[], Session]


class UtcDateTime(TypeDecorator[datetime]):
    """A timezone-aware UTC datetime on every backend (SQLite drops zones; this restores UTC)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not stored; use a timezone-aware value")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def make_engine(settings: Settings) -> Engine:
    """An engine for ``settings.database_url``; SQLite gets foreign keys, WAL and a busy wait."""
    url = make_url(settings.database_url)
    if url.get_backend_name() != "sqlite":
        return create_engine(url, pool_pre_ping=True)
    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions that never expire objects on commit (handlers return what they just wrote)."""
    return sessionmaker(bind=engine, expire_on_commit=False)
