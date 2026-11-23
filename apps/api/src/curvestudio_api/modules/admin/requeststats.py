"""Per-route request statistics: counted in memory between two housekeeping runs, then summed
into the day's ``request_stats_daily`` rows.

The request middleware feeds :class:`RequestStats`; the housekeeper empties it once an hour
with :meth:`RequestStats.snapshot_and_reset` and hands the counts to :func:`merge_into_daily`.
Durations are kept as a fixed histogram, so a day's percentiles come from the summed histogram
of all its hours and are never averaged from the hours' own percentiles. The longest request of
each hour is kept beside the histogram, so ``max_ms`` stays a true maximum.
"""

import bisect
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from curvestudio_api.modules.admin.models import RequestStatDaily

EDGE_COUNT = 20
BUCKET_BOUNDS_MS: tuple[float, ...] = tuple(
    round(60_000.0 ** (index / (EDGE_COUNT - 1)), 3) for index in range(EDGE_COUNT)
)
"""The bucket edges in milliseconds: log-spaced from 1 to 60,000. The first bucket is everything
below the first edge, every later one runs up to its edge (an edge belongs to the bucket it
opens), and the last holds everything from the last edge up."""

BUCKET_COUNT = len(BUCKET_BOUNDS_MS) + 1

P50 = 0.5
P95 = 0.95


def bucket_of(duration_ms: float) -> int:
    """The histogram bucket a request of ``duration_ms`` milliseconds belongs to."""
    return bisect.bisect_right(BUCKET_BOUNDS_MS, duration_ms)


def _edges(bucket: int) -> tuple[float, float | None]:
    """Where bucket ``bucket`` starts and ends; the overflow bucket has no end."""
    if bucket == 0:
        return 0.0, BUCKET_BOUNDS_MS[0]
    lower = BUCKET_BOUNDS_MS[bucket - 1]
    return lower, None if bucket == BUCKET_COUNT - 1 else BUCKET_BOUNDS_MS[bucket]


def percentile(histogram: Sequence[int], q: float) -> int:
    """The ``q``-th percentile in milliseconds, read off a histogram of bucket counts.

    The requests of a bucket are taken to be spread evenly over it, so the answer is
    interpolated inside the bucket that holds the ``q``-th of them. The overflow bucket has no
    end and reports the last edge; an empty histogram is 0.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    total = sum(histogram)
    if total == 0:
        return 0
    rank = q * total  # where the q-th request stands among the sorted durations, from 1
    seen = 0
    for bucket, count in enumerate(histogram):
        if count == 0:
            continue
        seen += count
        if rank <= seen:
            lower, upper = _edges(bucket)
            if upper is None:
                return round(lower)
            # The k-th of n evenly spread requests sits at (k - 0.5) / n of the bucket's width.
            position = min(1.0, max(0.0, (rank - (seen - count) - 0.5) / count))
            return round(lower + position * (upper - lower))
    return round(BUCKET_BOUNDS_MS[-1])  # not reached: q is at most 1


@dataclass
class RouteCounts:
    """One route's requests since the last merge: how many, how many were errors, how long they
    took, and the longest of them (in milliseconds)."""

    count: int = 0
    errors: int = 0
    histogram: list[int] = field(default_factory=lambda: [0] * BUCKET_COUNT)
    max_ms: float = 0.0


class RequestStats:
    """The requests of every route since the last :meth:`snapshot_and_reset`.

    Fed from the event loop and emptied from the housekeeper thread, so every access holds
    the lock. A request that matched no route is not counted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._routes: dict[str, RouteCounts] = {}

    def record(self, route_template: str | None, duration_ms: float, *, error: bool) -> None:
        """Count one request of ``route_template`` (the matched route's path template) that took
        ``duration_ms`` milliseconds; ``error`` when it ended in a server error."""
        if route_template is None:
            return
        with self._lock:
            counts = self._routes.setdefault(route_template, RouteCounts())
            counts.count += 1
            counts.errors += int(error)
            counts.histogram[bucket_of(duration_ms)] += 1
            counts.max_ms = max(counts.max_ms, duration_ms)

    def snapshot_and_reset(self) -> dict[str, RouteCounts]:
        """Every route's counts since the last call, after which the counters start empty."""
        with self._lock:
            taken = self._routes
            self._routes = {}
        return taken


def merge_into_daily(db: Session, snapshot: Mapping[str, RouteCounts], *, day: date) -> int:
    """Sum ``snapshot`` into the ``request_stats_daily`` rows of ``day``, one per route.

    A route's row is created on first sight. Its counts and histogram are added to the day's
    and the percentiles are recomputed from the summed histogram; ``max_ms`` keeps the larger of
    the stored value and the snapshot's longest request. Returns the number of rows touched.
    Flushes; the caller commits.
    """
    for route_template, counts in snapshot.items():
        row = db.scalar(
            select(RequestStatDaily).where(
                RequestStatDaily.day == day, RequestStatDaily.route_template == route_template
            )
        )
        if row is None:
            row = RequestStatDaily(
                day=day,
                route_template=route_template,
                count=0,
                errors=0,
                histogram=[0] * BUCKET_COUNT,
                p50_ms=0,
                p95_ms=0,
                max_ms=0,
            )
            db.add(row)
        row.count += counts.count
        row.errors += counts.errors
        # A new list, never an in-place edit: a change inside a JSON column is not tracked.
        row.histogram = [a + b for a, b in zip(row.histogram, counts.histogram, strict=True)]
        row.p50_ms = percentile(row.histogram, P50)
        row.p95_ms = percentile(row.histogram, P95)
        row.max_ms = max(row.max_ms, round(counts.max_ms))
    db.flush()
    return len(snapshot)
