"""The rules of the install-record domain: what a report stores, and the figures the admin
console shows. Nothing here is tied to a person: an installation is its hashed ``install_id``."""

import hashlib
import re
import uuid
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

from sqlalchemy import and_, case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import InstrumentedAttribute, Session
from sqlalchemy.sql import ColumnElement

from curvestudio_api import __version__
from curvestudio_api.modules.installs.models import InstallEvent, InstallRecord
from curvestudio_api.modules.installs.schemas import (
    InstallRecordIn,
    InstallSummaryOut,
    NameCount,
    WeekCount,
)
from curvestudio_api.timeutil import IST, utcnow

OS_NAMES = ("windows", "macos", "linux", "other")
ARCHES = ("x86_64", "arm64", "other")
PACKAGES = ("deb", "appimage", "tarball", "windows_setup", "windows_zip", "source", "other")
SESSIONS = ("x11", "wayland", "windows", "other")

NEW_DAYS = 30
WEEKS = 12

_RELEASE = re.compile(r"\d+\.\d+\.\d+")
"""A release of the form ``MAJOR.MINOR.PATCH``; anything else (``0.12.0.dev3``) is unreleased."""


def install_key(install_id: uuid.UUID) -> str:
    """The SHA-256 hex digest of ``install_id`` in its canonical lower-case text, so the same id
    hashes alike however the desktop app spelled it."""
    return hashlib.sha256(str(install_id).encode("ascii")).hexdigest()


def record_install(
    db: Session, body: InstallRecordIn, *, now: datetime | None = None
) -> Literal["install", "upgrade", "seen"]:
    """Store one report and say what it was: an installation's first report, its move to a
    version it had never reported, or one more sighting. Flushes; the caller commits."""
    at = now or utcnow()
    key = install_key(body.install_id)
    record = _existing(db, key)
    if record is None:
        candidate = InstallRecord(install_key=key, first_seen_at=at, last_seen_at=at)
        _describe(candidate, body)
        db.add(candidate)
        try:
            db.flush()  # the event needs the record's id
        except IntegrityError:
            # Two first launches of one installation reported at once and the other request
            # committed first. Its row stands; this report is handled below as a sighting.
            db.rollback()
            record = _existing(db, key)
            if record is None:
                raise
        else:
            db.add(
                InstallEvent(
                    install_record_id=candidate.id, event="install", to_version=body.app_version
                )
            )
            db.flush()
            return "install"

    previous = record.app_version
    _describe(record, body)
    record.last_seen_at = at
    if _has_arrived_at(db, record.id, body.app_version):
        db.flush()
        return "seen"
    db.add(
        InstallEvent(
            install_record_id=record.id,
            event="upgrade",
            from_version=previous,
            to_version=body.app_version,
        )
    )
    db.flush()
    return "upgrade"


def install_summary(
    db: Session, *, current_release: str, now: datetime | None = None
) -> InstallSummaryOut:
    """The figures the admin console shows, as of ``now``."""
    at = now or utcnow()
    total = _count(db)
    latest_release = current_release or __version__
    # An unreleased version is nobody's latest: a development build never counts as one.
    on_latest = (
        _count(db, InstallRecord.app_version == latest_release)
        if _RELEASE.fullmatch(latest_release)
        else 0
    )
    return InstallSummaryOut(
        total=total,
        new_30d=_count(db, InstallRecord.first_seen_at >= at - timedelta(days=NEW_DAYS)),
        latest_release=latest_release,
        latest_share=on_latest / total if total else 0.0,
        by_version=_per_value(db, InstallRecord.app_version),
        by_os=_per_value(db, InstallRecord.os_name),
        by_arch=_per_value(db, InstallRecord.arch),
        by_package=_per_value(db, InstallRecord.package),
        by_session=_per_value(db, InstallRecord.session),
        new_per_week=_new_per_week(db, at),
    )


def _existing(db: Session, key: str) -> InstallRecord | None:
    """The record of the installation whose key is ``key``; ``None`` before its first report."""
    return db.scalar(select(InstallRecord).where(InstallRecord.install_key == key))


def _known(value: str, names: tuple[str, ...]) -> str:
    """``value`` when it is exactly one of ``names``; otherwise ``other``."""
    return value if value in names else "other"


def _describe(record: InstallRecord, body: InstallRecordIn) -> None:
    """Set the record's version and environment from ``body``, the latest report."""
    record.app_version = body.app_version
    record.os_name = _known(body.os_name, OS_NAMES)
    record.os_version = body.os_version
    record.arch = _known(body.arch, ARCHES)
    record.package = _known(body.package, PACKAGES)
    record.session = _known(body.session, SESSIONS)


def _has_arrived_at(db: Session, record_id: uuid.UUID, version: str) -> bool:
    """Whether an event of the record already ends at ``version``."""
    event = db.scalar(
        select(InstallEvent.id)
        .where(InstallEvent.install_record_id == record_id, InstallEvent.to_version == version)
        .limit(1)
    )
    return event is not None


def _count(db: Session, *where: ColumnElement[bool]) -> int:
    """How many records satisfy ``where``; all of them when there is none."""
    return int(db.scalar(select(func.count()).select_from(InstallRecord).where(*where)) or 0)


def _per_value(db: Session, column: InstrumentedAttribute[str]) -> list[NameCount]:
    """The records counted per value of ``column``, most first and then by name."""
    rows = db.execute(
        select(column, func.count()).group_by(column).order_by(func.count().desc(), column)
    )
    return [NameCount(name=name, count=count) for name, count in rows]


def _new_per_week(db: Session, now: datetime) -> list[WeekCount]:
    """First sightings per week for the ``WEEKS`` weeks ending with the one ``now`` is in,
    oldest first; a week without any is a zero.

    A week starts on a Monday by the IST calendar day of ``first_seen_at``. IST keeps one
    offset all year, so each of its weeks is the same 168 hours in UTC: the bounds are worked
    out here and the database does the counting.
    """
    this_monday = _monday_of(now)
    mondays = [this_monday - timedelta(weeks=back) for back in range(WEEKS - 1, -1, -1)]
    seen_at = InstallRecord.first_seen_at
    week = case(
        *[
            (and_(seen_at >= low, seen_at < high), index)
            for index, (low, high) in enumerate(map(_utc_week, mondays))
        ],
        else_=None,
    )
    per_week = [0] * WEEKS
    for index, count in db.execute(select(week, func.count()).group_by(week)):
        if index is not None:  # None: first seen before these twelve weeks
            per_week[index] = count
    return [
        WeekCount(week_start=monday, count=per_week[index]) for index, monday in enumerate(mondays)
    ]


def _monday_of(moment: datetime) -> date:
    """The Monday of the IST week ``moment`` falls in."""
    day = moment.astimezone(IST).date()
    return day - timedelta(days=day.weekday())


def _utc_week(monday: date) -> tuple[datetime, datetime]:
    """The instants, in UTC, that the week starting on the IST ``monday`` runs between."""
    start = datetime.combine(monday, time.min, tzinfo=IST).astimezone(UTC)
    return start, start + timedelta(days=7)
