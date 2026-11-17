"""In-memory sliding-window rate limiter (single process; see the deployment notes)."""

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime

from curvestudio_api.errors import ApiError
from curvestudio_api.timeutil import to_ist_iso, utcnow


class SlidingWindowLimiter:
    """Allows at most ``limit`` hits per ``window_s`` seconds for each key."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record a hit for ``key``.

        Raises:
            ApiError: 429 ``RATE_LIMITED`` (with ``Retry-After``) when the window is full.
        """
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window_s - (now - hits[0])) + 1)
                raise ApiError(
                    429,
                    "RATE_LIMITED",
                    "too many attempts; try again later",
                    {"retry_after_s": retry_after},
                    {"Retry-After": str(retry_after)},
                )
            hits.append(now)
            if len(self._hits) > 10_000:
                self._prune(now)

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
            del self._hits[key]


class DailyCap:
    """Counts events per key per (IST) day, to cap how many audit rows one source can write."""

    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock
        self._counts: dict[str, int] = {}
        self._day = ""
        self._lock = threading.Lock()

    def hit(self, key: str) -> int:
        """Record an event for ``key`` and return how many it has had today."""
        today = to_ist_iso(self._clock())[:10]
        with self._lock:
            if today != self._day:
                self._counts.clear()
                self._day = today
            count = self._counts.get(key, 0) + 1
            if len(self._counts) < 100_000:
                self._counts[key] = count
            return count
