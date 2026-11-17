"""Periodic clean-up, run in-process once an hour (and callable directly)."""

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, delete, or_

from curvestudio_api.db import SessionFactory
from curvestudio_api.logging_setup import get_logger
from curvestudio_api.orm import Job, RefreshToken
from curvestudio_api.timeutil import utcnow

_log = get_logger("housekeeping")

RETENTION = timedelta(days=30)
FINISHED = ("succeeded", "failed", "cancelled")


@dataclass(frozen=True)
class PurgeResult:
    refresh_tokens: int
    jobs: int


def purge(session_factory: SessionFactory, now: datetime | None = None) -> PurgeResult:
    """Delete what nobody needs any more.

    * refresh tokens that expired, or were revoked and have since expired, more than 30 days ago
      (a revoked but still valid token is kept: it is how a replay is recognised);
    * jobs that finished more than 30 days ago (their fits stay).

    ``audit_logs`` is append-only and is never purged; repeated failed logins are capped when
    they are written instead (see ``routers/auth.py``).
    """
    cutoff = (now or utcnow()) - RETENTION
    current = now or utcnow()
    with session_factory() as db:
        tokens = db.execute(
            delete(RefreshToken).where(
                or_(
                    RefreshToken.expires_at < cutoff,
                    and_(RefreshToken.revoked_at < cutoff, RefreshToken.expires_at < current),
                )
            )
        ).rowcount  # type: ignore[attr-defined]
        jobs = db.execute(
            delete(Job).where(Job.status.in_(FINISHED), Job.finished_at < cutoff)
        ).rowcount  # type: ignore[attr-defined]
        db.commit()
    return PurgeResult(refresh_tokens=int(tokens), jobs=int(jobs))


class Housekeeper:
    """Runs :func:`purge` every ``interval_s`` seconds on a daemon thread."""

    def __init__(self, session_factory: SessionFactory, interval_s: float) -> None:
        self._sessions = session_factory
        self._interval_s = interval_s
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

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                result = purge(self._sessions)
                _log.info(
                    "purged",
                    extra={"fields": {"tokens": result.refresh_tokens, "jobs": result.jobs}},
                )
            except Exception:
                _log.exception("housekeeping failed")
