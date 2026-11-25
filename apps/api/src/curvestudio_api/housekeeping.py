"""Periodic clean-up, run in-process once an hour (and callable directly)."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, delete, or_

from curvestudio_api import licences
from curvestudio_api.db import SessionFactory
from curvestudio_api.logging_setup import get_logger, log_fields
from curvestudio_api.modules.admin.models import RequestStatDaily
from curvestudio_api.modules.admin.requeststats import RequestStats, merge_into_daily
from curvestudio_api.modules.errors.capture import bind_error_capture_sink
from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.modules.errors.service import CLOSED_STATUSES
from curvestudio_api.orm import Job, RefreshToken
from curvestudio_api.settings import Settings
from curvestudio_api.timeutil import IST, utcnow

_log = get_logger("housekeeping")

RETENTION = timedelta(days=30)
FINISHED = ("succeeded", "failed", "cancelled")
CLOSED_ERROR_RETENTION = timedelta(days=90)
ERROR_RETENTION = timedelta(days=180)
REQUEST_STATS_RETENTION = timedelta(days=400)
RECHECK_INTERVAL_S = 3600.0
"""The licence re-check runs at most this often, whatever the housekeeping interval is."""
Recheck = Callable[..., int]
"""``licences.recheck``: ``(db, settings, now=None) -> int``; tests inject a stub."""


@dataclass(frozen=True)
class PurgeResult:
    """What one purge removed."""

    refresh_tokens: int
    jobs: int
    error_events_closed: int
    """``resolved`` or ``ignored`` events last seen more than 90 days ago."""
    error_events_expired: int
    """Events not covered above, last seen more than 180 days ago."""
    request_stats: int
    """Daily request-statistic rows for days more than 400 days in the past."""


def purge(session_factory: SessionFactory, now: datetime | None = None) -> PurgeResult:
    """Delete what nobody needs any more.

    * refresh tokens that expired, or were revoked and have since expired, more than 30 days ago
      (a revoked but still valid token is kept: it is how a replay is recognised);
    * jobs that finished more than 30 days ago (their fits stay);
    * error events that are ``resolved`` or ``ignored`` and were last seen more than 90 days ago,
      and every other error event last seen more than 180 days ago;
    * request statistics for days more than 400 days in the past.

    ``audit_logs`` is append-only and is never purged; repeated failed logins are capped when
    they are written instead (see ``routers/auth.py``). Install records and events are kept for
    as long as the database is.
    """
    at = now or utcnow()
    cutoff = at - RETENTION
    with session_factory() as db:
        tokens = db.execute(
            delete(RefreshToken).where(
                or_(
                    RefreshToken.expires_at < cutoff,
                    and_(RefreshToken.revoked_at < cutoff, RefreshToken.expires_at < at),
                )
            )
        ).rowcount  # type: ignore[attr-defined]
        jobs = db.execute(
            delete(Job).where(Job.status.in_(FINISHED), Job.finished_at < cutoff)
        ).rowcount  # type: ignore[attr-defined]
        closed = db.execute(
            delete(ErrorEvent).where(
                ErrorEvent.status.in_(CLOSED_STATUSES),
                ErrorEvent.last_seen_at < at - CLOSED_ERROR_RETENTION,
            )
        ).rowcount  # type: ignore[attr-defined]
        expired = db.execute(
            delete(ErrorEvent).where(ErrorEvent.last_seen_at < at - ERROR_RETENTION)
        ).rowcount  # type: ignore[attr-defined]
        # The days of the statistics are IST calendar days, so the cut-off is one as well.
        stats_before = (at.astimezone(IST) - REQUEST_STATS_RETENTION).date()
        stats = db.execute(
            delete(RequestStatDaily).where(RequestStatDaily.day < stats_before)
        ).rowcount  # type: ignore[attr-defined]
        db.commit()
    return PurgeResult(
        refresh_tokens=int(tokens),
        jobs=int(jobs),
        error_events_closed=int(closed),
        error_events_expired=int(expired),
        request_stats=int(stats),
    )


class Housekeeper:
    """Runs :func:`purge`, the request-statistics merge and the licence re-check every
    ``interval_s`` seconds on a daemon thread.

    The merge sums what the middleware counted since the previous run into the current IST
    day's ``request_stats_daily`` rows. The counters are emptied first, so a merge that then
    cannot reach the database loses that run's counts; the purge has nothing to lose and simply
    runs again next time. The re-check runs only while the licence connector is configured,
    at most once an hour of wall clock (``RECHECK_INTERVAL_S``), and a portal it cannot reach
    is logged and waited for, never raised past the loop.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        interval_s: float,
        *,
        settings: Settings,
        stats: RequestStats,
        recheck: Recheck = licences.recheck,
    ) -> None:
        self._sessions = session_factory
        self._interval_s = interval_s
        self._settings = settings
        self._stats = stats
        self._recheck = recheck
        self._last_recheck: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="curvestudio-housekeeping", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def run_once(self) -> PurgeResult:
        """One run: purge, merge the request statistics into today's rows (by IST), then
        re-check the licences when their hour is up."""
        result = purge(self._sessions)
        merged = self._merge_stats()
        checked = self._recheck_licences()
        _log.info(
            "purged",
            extra=log_fields(
                tokens=result.refresh_tokens,
                jobs=result.jobs,
                error_events_closed=result.error_events_closed,
                error_events_expired=result.error_events_expired,
                request_stats=result.request_stats,
                stats_merged=merged,
                **({} if checked is None else {"licences_checked": checked}),
            ),
        )
        return result

    def _recheck_licences(self) -> int | None:
        """The licence re-check, when the connector is on and its hour is up; how many
        entitlements were checked, or ``None`` when it did not run."""
        if not licences.billing_on(self._settings):
            return None
        now = time.monotonic()
        if self._last_recheck is not None and now - self._last_recheck < RECHECK_INTERVAL_S:
            return None
        self._last_recheck = now
        try:
            with self._sessions() as db:
                return int(self._recheck(db, self._settings))
        except Exception:
            _log.exception("the licence re-check failed")
            return None

    def _merge_stats(self) -> int:
        """Empty the counters into today's rows on a fresh session; how many rows were touched."""
        snapshot = self._stats.snapshot_and_reset()
        if not snapshot:
            return 0
        today = utcnow().astimezone(IST).date()
        with self._sessions() as db:
            touched = merge_into_daily(db, snapshot, day=today)
            db.commit()
        return touched

    def _run(self) -> None:
        # A failure logged on this thread is recorded into this housekeeper's own database.
        bind_error_capture_sink(self._sessions)
        while not self._stop.wait(self._interval_s):
            try:
                self.run_once()
            except Exception:
                _log.exception("housekeeping failed")
