"""The install-record domain: what a report stores, what is refused, and the figures the admin
console is shown. A report names no person and no address, and the reply to one tells nothing."""

import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import InstrumentedAttribute, Session

from conftest import make_settings
from curvestudio_api import __version__
from curvestudio_api.app import create_app
from curvestudio_api.db import SessionFactory
from curvestudio_api.migrations import upgrade_to_head
from curvestudio_api.modules.installs import service
from curvestudio_api.modules.installs.models import InstallEvent, InstallRecord
from curvestudio_api.modules.installs.schemas import InstallRecordIn
from curvestudio_api.timeutil import IST

API = "/api/v1"
INSTALLS = f"{API}/installs"


def report(**changes: Any) -> dict[str, Any]:
    """A valid report from a desktop installation, with ``changes`` over its defaults."""
    return {
        "install_id": str(uuid.uuid4()),
        "app_version": "0.12.0",
        "os_name": "linux",
        "os_version": "6.8.0",
        "arch": "x86_64",
        "package": "appimage",
        "session": "wayland",
        **changes,
    }


@pytest.fixture
def sessions(tmp_path: Path) -> SessionFactory:
    """Sessions on a database of this test's own, migrated to head (no app is started)."""
    ctx = create_app(make_settings(tmp_path)).state.ctx
    upgrade_to_head(ctx.engine)
    return ctx.session_factory


@contextmanager
def rows(client: TestClient) -> Iterator[Session]:
    """A session on ``client``'s database, to read what the endpoint stored."""
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        yield db


def stored_events(db: Session) -> list[tuple[str, str | None, str]]:
    """Every event as ``(event, from_version, to_version)``, oldest first."""
    found = db.scalars(select(InstallEvent).order_by(InstallEvent.created_at))
    return [(event.event, event.from_version, event.to_version) for event in found]


def assert_nothing_stored(client: TestClient) -> None:
    with rows(client) as db:
        assert db.scalar(select(func.count()).select_from(InstallRecord)) == 0
        assert db.scalar(select(func.count()).select_from(InstallEvent)) == 0


def _row(
    version: str, os_name: str, arch: str, package: str, session: str, first_seen: datetime
) -> InstallRecord:
    """A record written straight to the table, under a key of its own."""
    return InstallRecord(
        install_key=uuid.uuid4().hex * 2,
        app_version=version,
        os_name=os_name,
        os_version="6.8.0",
        arch=arch,
        package=package,
        session=session,
        first_seen_at=first_seen,
        last_seen_at=first_seen,
    )


# -- what a report stores -------------------------------------------------------------------------


def test_a_first_report_creates_the_record_and_an_install_event(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    install_id = uuid.uuid4()
    # Sent in upper case: the key is taken from the canonical lower-case text.
    response = client.post(INSTALLS, json=report(install_id=str(install_id).upper()))

    assert response.status_code == 204
    assert response.content == b""
    with rows(client) as db:
        [record] = db.scalars(select(InstallRecord))
        assert record.install_key == hashlib.sha256(str(install_id).encode()).hexdigest()
        assert (
            record.app_version,
            record.os_name,
            record.os_version,
            record.arch,
            record.package,
            record.session,
        ) == ("0.12.0", "linux", "6.8.0", "x86_64", "appimage", "wayland")
        assert record.first_seen_at is not None
        assert record.first_seen_at == record.last_seen_at
        assert record.created_at is not None
        assert record.updated_at is not None
        [event] = db.scalars(select(InstallEvent))
        assert event.install_record_id == record.id
        assert (event.event, event.from_version, event.to_version) == ("install", None, "0.12.0")

    # Exactly these columns, so none for an address or an account.
    assert {column.name for column in InstallRecord.__table__.columns} == {
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
    }


def test_the_same_id_in_another_spelling_is_the_same_installation(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    install_id = uuid.uuid4()
    for spelling in (f"{{{install_id}}}", install_id.hex, str(install_id)):
        response = client.post(INSTALLS, json=report(install_id=spelling))
        assert response.status_code == 204
    with rows(client) as db:
        assert db.scalar(select(func.count()).select_from(InstallRecord)) == 1
        assert stored_events(db) == [("install", None, "0.12.0")]


def test_the_same_version_again_moves_last_seen_and_adds_no_event(
    sessions: SessionFactory,
) -> None:
    install_id = str(uuid.uuid4())
    first_seen = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
    seen_again = first_seen + timedelta(days=3, hours=2)
    with sessions() as db:
        first = service.record_install(
            db, InstallRecordIn(**report(install_id=install_id)), now=first_seen
        )
        again = service.record_install(
            db,
            InstallRecordIn(**report(install_id=install_id, os_version="6.9.1")),
            now=seen_again,
        )
        db.commit()
        assert (first, again) == ("install", "seen")
        [record] = db.scalars(select(InstallRecord))
        assert (record.first_seen_at, record.last_seen_at) == (first_seen, seen_again)
        assert record.os_version == "6.9.1"  # the environment is the one last reported
        assert stored_events(db) == [("install", None, "0.12.0")]


def test_a_new_version_adds_an_upgrade_event(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory()
    install_id = str(uuid.uuid4())
    for version in ("0.12.0", "0.12.1"):
        response = client.post(INSTALLS, json=report(install_id=install_id, app_version=version))
        assert response.status_code == 204
    with rows(client) as db:
        [record] = db.scalars(select(InstallRecord))
        assert record.app_version == "0.12.1"
        assert stored_events(db) == [
            ("install", None, "0.12.0"),
            ("upgrade", "0.12.0", "0.12.1"),
        ]


def test_a_version_seen_before_adds_no_second_event_for_it(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    install_id = str(uuid.uuid4())
    for version in ("0.12.0", "0.12.1", "0.12.0"):
        response = client.post(INSTALLS, json=report(install_id=install_id, app_version=version))
        assert response.status_code == 204
    with rows(client) as db:
        [record] = db.scalars(select(InstallRecord))
        assert record.app_version == "0.12.0"
        assert stored_events(db) == [
            ("install", None, "0.12.0"),
            ("upgrade", "0.12.0", "0.12.1"),
        ]


def test_a_value_outside_its_list_is_stored_as_other(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    response = client.post(
        INSTALLS,
        json=report(os_name="haiku", package="rpm", session="Wayland", arch="arm64"),
    )
    assert response.status_code == 204
    with rows(client) as db:
        [record] = db.scalars(select(InstallRecord))
        assert (record.os_name, record.package, record.session) == ("other", "other", "other")
        assert record.arch == "arm64"


def test_deleting_a_record_takes_its_events(sessions: SessionFactory) -> None:
    install_id = str(uuid.uuid4())
    with sessions() as db:
        service.record_install(db, InstallRecordIn(**report(install_id=install_id)))
        service.record_install(
            db, InstallRecordIn(**report(install_id=install_id, app_version="0.12.1"))
        )
        db.commit()
        [record] = db.scalars(select(InstallRecord))
        assert len(stored_events(db)) == 2

        db.delete(record)
        db.commit()
        assert stored_events(db) == []


# -- what is refused ------------------------------------------------------------------------------


def test_malformed_reports_are_refused_and_store_nothing(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    without_arch = report()
    del without_arch["arch"]
    for bad in (
        report(install_id=str(uuid.uuid1())),
        report(app_version="1" * 41),
        report(os_name=""),
        report(ip="203.0.113.7"),
        without_arch,
    ):
        response = client.post(INSTALLS, json=bad)
        assert response.status_code == 422, bad
        assert response.json()["error"] == "VALIDATION_ERROR"
    assert_nothing_stored(client)


def test_a_body_over_two_kilobytes_is_refused(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    content = json.dumps(report(os_version="x" * 40, pad="x" * 2048)).encode()
    assert len(content) > 2048
    response = client.post(INSTALLS, content=content, headers={"Content-Type": "application/json"})
    # The limit comes first: not the 422 that the extra "pad" key would otherwise earn.
    assert response.status_code == 413
    assert response.json()["error"] == "TOO_LARGE"
    assert_nothing_stored(client)


def test_twenty_one_reports_from_one_address_are_refused(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))  # the limits as served, not the tests' raised ones
    assert app.state.ctx.install_limiter.limit == 20
    behind_the_address = (str(uuid.uuid4()), str(uuid.uuid4()))
    with TestClient(app) as client:
        # Two installations behind one address draw on the one budget.
        for nth in range(20):
            response = client.post(INSTALLS, json=report(install_id=behind_the_address[nth % 2]))
            assert response.status_code == 204, nth
        for install_id in behind_the_address:
            refused = client.post(INSTALLS, json=report(install_id=install_id))
            assert refused.status_code == 429
            assert refused.json()["error"] == "RATE_LIMITED"
            assert int(refused.headers["Retry-After"]) >= 1
        with rows(client) as db:
            assert db.scalar(select(func.count()).select_from(InstallRecord)) == 2


# -- what the reply tells -------------------------------------------------------------------------


def test_the_reply_is_the_same_whether_or_not_the_installation_was_known(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory()
    known, other = str(uuid.uuid4()), str(uuid.uuid4())
    replies = [
        client.post(INSTALLS, json=report(install_id=known)),
        client.post(INSTALLS, json=report(install_id=known)),
        client.post(INSTALLS, json=report(install_id=known, app_version="0.12.1")),
        client.post(INSTALLS, json=report(install_id=other)),
    ]

    def steady(response: Any) -> dict[str, str]:
        return {name: value for name, value in response.headers.items() if name != "x-request-id"}

    assert "content-type" not in steady(replies[0])
    for response in replies:
        assert response.status_code == 204
        assert response.content == b""
        assert steady(response) == steady(replies[0])


# -- the summary ----------------------------------------------------------------------------------

# version, operating system, architecture, package, session, first seen this many days before "now"
SEEDED = [
    ("0.12.0", "linux", "x86_64", "appimage", "wayland", 10),
    ("0.12.0", "windows", "x86_64", "windows_setup", "windows", 10),
    ("0.11.1", "linux", "arm64", "deb", "x11", 10),
    ("0.11.1", "macos", "arm64", "other", "other", 10),
    ("0.11.0", "windows", "x86_64", "windows_zip", "windows", 40),
    ("0.11.0", "linux", "x86_64", "tarball", "x11", 40),
    ("0.10.2", "linux", "x86_64", "source", "wayland", 40),
    ("0.10.2", "windows", "arm64", "windows_setup", "windows", 40),
]


def _monday_of(moment: datetime) -> date:
    day = moment.astimezone(IST).date()
    return day - timedelta(days=day.weekday())


def test_the_summary_equals_direct_counts(sessions: SessionFactory) -> None:
    now = datetime(2026, 11, 29, 20, 0, tzinfo=UTC)  # Monday 01:30 in IST, still Sunday by UTC
    assert now.astimezone(IST).weekday() == 0
    assert now.astimezone(UTC).weekday() == 6
    this_week = _monday_of(now)
    # The week's edges in IST. Its first minute is still Sunday by UTC.
    week_starts = datetime.combine(this_week, time.min, tzinfo=IST)
    just_before = week_starts - timedelta(minutes=1)
    just_after = week_starts + timedelta(minutes=30)
    assert week_starts.astimezone(UTC).weekday() == 6

    with sessions() as db:
        edges = [
            _row("0.10.0", "linux", "x86_64", "appimage", "wayland", at)
            for at in (just_before, week_starts, just_after)
        ]
        # First seen before the twelve weeks: counted, but in no week.
        long_ago = _row("0.9.0", "other", "other", "other", "other", now - timedelta(days=100))
        db.add_all(
            [_row(*facts[:5], now - timedelta(days=facts[5])) for facts in SEEDED]
            + edges
            + [long_ago]
        )
        db.commit()
        summary = service.install_summary(db, current_release="0.12.0", now=now)

        def counted(column: InstrumentedAttribute[str]) -> list[tuple[str, int]]:
            """Per value, straight from the database, most first and then by name."""
            rows = db.execute(select(column, func.count()).group_by(column)).all()
            return sorted(
                ((name, count) for name, count in rows), key=lambda item: (-item[1], item[0])
            )

        total = db.scalar(select(func.count()).select_from(InstallRecord))
        assert summary.total == total == len(SEEDED) + 4
        thirty_days = db.scalar(
            select(func.count())
            .select_from(InstallRecord)
            .where(InstallRecord.first_seen_at >= now - timedelta(days=30))
        )
        assert summary.new_30d == thirty_days == 7

        assert summary.latest_release == "0.12.0"
        on_latest = db.scalar(
            select(func.count())
            .select_from(InstallRecord)
            .where(InstallRecord.app_version == "0.12.0")
        )
        assert summary.latest_share == on_latest / total

        for found, column in (
            (summary.by_version, InstallRecord.app_version),
            (summary.by_os, InstallRecord.os_name),
            (summary.by_arch, InstallRecord.arch),
            (summary.by_package, InstallRecord.package),
            (summary.by_session, InstallRecord.session),
        ):
            assert [(item.name, item.count) for item in found] == counted(column)
        # By hand, for the two lists that carry ties: the ties are real and broken by name.
        assert [(item.name, item.count) for item in summary.by_version] == [
            ("0.10.0", 3),
            ("0.10.2", 2),
            ("0.11.0", 2),
            ("0.11.1", 2),
            ("0.12.0", 2),
            ("0.9.0", 1),
        ]
        assert [(item.name, item.count) for item in summary.by_package] == [
            ("appimage", 4),
            ("other", 2),
            ("windows_setup", 2),
            ("deb", 1),
            ("source", 1),
            ("tarball", 1),
            ("windows_zip", 1),
        ]

        # Weeks, bucketed here by hand from the stored instants.
        weeks_back = Counter(
            (this_week - _monday_of(first_seen)).days // 7
            for first_seen in db.scalars(select(InstallRecord.first_seen_at))
        )
        expected = [weeks_back.get(back, 0) for back in range(11, -1, -1)]
        assert [week.week_start for week in summary.new_per_week] == [
            this_week - timedelta(weeks=back) for back in range(11, -1, -1)
        ]
        assert [week.count for week in summary.new_per_week] == expected
        # Sunday 23:59 IST is last week; Monday 00:00 and 00:30 IST (still Sunday by UTC) are this.
        # The twelve weeks hold eleven of the twelve records: the oldest is in none of them.
        assert expected == [0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 1, 2]


def test_a_development_version_is_never_the_latest(sessions: SessionFactory) -> None:
    was_first_seen = datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
    development_build = "0.12.0.dev3"
    with sessions() as db:
        # One installation runs the development build: it is still nobody's latest.
        db.add_all(
            [
                _row(development_build, "linux", "arm64", "deb", "x11", was_first_seen),
                _row(__version__, "linux", "x86_64", "appimage", "wayland", was_first_seen),
                _row("0.11.0", "windows", "x86_64", "windows_setup", "windows", was_first_seen),
            ]
        )
        db.commit()

        development = service.install_summary(db, current_release=development_build)
        assert development.latest_release == development_build
        assert development.latest_share == 0.0

        fallback = service.install_summary(db, current_release="")
        assert fallback.latest_release == __version__
        released = re.fullmatch(r"\d+\.\d+\.\d+", __version__) is not None
        assert fallback.latest_share == (1 / 3 if released else 0.0)


def test_the_summary_of_an_empty_table_is_all_zero(sessions: SessionFactory) -> None:
    with sessions() as db:
        summary = service.install_summary(db, current_release="0.12.0")
        assert (summary.total, summary.new_30d, summary.latest_release, summary.latest_share) == (
            0,
            0,
            "0.12.0",
            0.0,
        )
        assert (
            summary.by_version
            == summary.by_os
            == summary.by_arch
            == summary.by_package
            == summary.by_session
            == []
        )
        assert [week.count for week in summary.new_per_week] == [0] * 12


def test_two_first_reports_of_one_installation_racing_end_in_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two first launches of one installation can report at the same moment. The second request
    read "no record" just before the first one committed, so its insert meets the unique key."""
    app = create_app(make_settings(tmp_path))
    upgrade_to_head(app.state.ctx.engine)
    client = TestClient(app)
    body = report()
    assert client.post(INSTALLS, json=body).status_code == 204

    # The other request's row exists by now; only this request's earlier lookup missed it.
    lookups = iter([None])
    real_lookup = service._existing
    monkeypatch.setattr(service, "_existing", lambda db, key: next(lookups, real_lookup(db, key)))

    assert client.post(INSTALLS, json=body).status_code == 204

    with app.state.ctx.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(InstallRecord)) == 1
        events = db.scalars(select(InstallEvent.event)).all()
    assert events == ["install"]
