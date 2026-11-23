"""The in-memory per-route request statistics and their merge into ``request_stats_daily``."""

import itertools
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select

from curvestudio_api.db import SessionFactory, make_engine, make_session_factory
from curvestudio_api.migrations import upgrade_to_head
from curvestudio_api.modules.admin.models import RequestStatDaily
from curvestudio_api.modules.admin.requeststats import (
    BUCKET_BOUNDS_MS,
    BUCKET_COUNT,
    RequestStats,
    RouteCounts,
    bucket_of,
    capped_percentiles,
    merge_into_daily,
    percentile,
)
from curvestudio_api.settings import Settings

DAY = date(2026, 11, 23)
HEALTH = "/api/v1/health"


@pytest.fixture
def sessions(tmp_path: Path) -> SessionFactory:
    """Sessions on a database of this test's own, migrated to head (no app is started)."""
    settings = Settings(
        env="test", database_url=f"sqlite:///{tmp_path}/stats.db", data_dir=tmp_path
    )
    engine = make_engine(settings)
    upgrade_to_head(engine)
    return make_session_factory(engine)


def _fill(stats: RequestStats, route: str, durations: Sequence[float], *, error=False) -> None:
    for duration in durations:
        stats.record(route, duration, error=error)


def _summed(first: RouteCounts, second: RouteCounts) -> list[int]:
    return [a + b for a, b in zip(first.histogram, second.histogram, strict=True)]


def _raw_percentile(ordered: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of the durations themselves, interpolated between neighbours."""
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def test_two_snapshots_sum_into_one_daily_row(sessions: SessionFactory) -> None:
    morning, afternoon = RequestStats(), RequestStats()
    _fill(morning, HEALTH, [4.0, 4.5, 5.0])
    _fill(morning, "/api/v1/boom", [900.0], error=True)
    _fill(afternoon, HEALTH, [50.0, 51.0])
    one = morning.snapshot_and_reset()
    two = afternoon.snapshot_and_reset()

    with sessions() as db:
        assert merge_into_daily(db, one, day=DAY) == 2  # one row per route in the snapshot
        assert merge_into_daily(db, two, day=DAY) == 1  # the other route is not in this one
        db.commit()

    with sessions() as db:
        rows = {row.route_template: row for row in db.scalars(select(RequestStatDaily))}
    assert set(rows) == {HEALTH, "/api/v1/boom"}

    health = rows[HEALTH]
    assert (health.day, health.count, health.errors) == (DAY, 5, 0)
    combined = _summed(one[HEALTH], two[HEALTH])
    assert health.histogram == combined
    assert health.p50_ms == percentile(combined, 0.5)
    assert health.p95_ms == percentile(combined, 0.95)
    assert health.p50_ms < health.p95_ms  # the slow second hour moved the tail
    assert health.max_ms == 51  # the longest request of the day, not a percentile of the histogram

    boom = rows["/api/v1/boom"]
    assert (boom.count, boom.errors, boom.max_ms) == (1, 1, 900)


def test_percentile_of_the_combined_histogram_is_not_the_mean_of_percentiles(
    sessions: SessionFactory,
) -> None:
    fast, slow = 4.9, 50.1
    morning, afternoon = RequestStats(), RequestStats()
    _fill(morning, HEALTH, [fast] * 100)
    _fill(afternoon, HEALTH, [slow] * 100)
    one = morning.snapshot_and_reset()
    two = afternoon.snapshot_and_reset()
    separate = [percentile(counts.histogram, 0.95) for counts in (one[HEALTH], two[HEALTH])]
    combined = _summed(one[HEALTH], two[HEALTH])

    with sessions() as db:
        assert merge_into_daily(db, one, day=DAY) == 1
        assert merge_into_daily(db, two, day=DAY) == 1
        db.commit()
    with sessions() as db:
        stored = db.scalar(select(RequestStatDaily)).p95_ms

    from_raw = _raw_percentile(sorted([fast] * 100 + [slow] * 100), 0.95)
    assert from_raw == slow  # 190 of the 200 were fast: the 95th percentile is a slow one
    # The buckets' estimate would pass the slowest request there was, so it is capped by it.
    assert percentile(combined, 0.95) == 55
    assert stored == capped_percentiles(combined, round(slow))[1] == 50
    # The estimate sits in the bucket that holds the true 95th percentile.
    assert bucket_of(float(stored)) == bucket_of(from_raw)
    # Averaging the two hours' own 95th percentiles would have answered about half that.
    assert stored != pytest.approx(sum(separate) / 2, rel=0.25)


def test_overflow_bucket_and_reset() -> None:
    bounds = BUCKET_BOUNDS_MS
    last = bounds[-1]
    assert bounds[0] == 1 and last == 60_000
    assert 18 <= len(bounds) <= 22
    assert len(bounds) + 1 == BUCKET_COUNT
    ratios = [upper / lower for lower, upper in itertools.pairwise(bounds)]
    assert max(ratios) - min(ratios) < 0.01  # one ratio throughout: the spacing is logarithmic

    # Below the first edge is bucket 0; an edge belongs to the bucket it opens.
    assert bucket_of(0.0) == 0 and bucket_of(0.999) == 0 and bucket_of(1.0) == 1
    assert bucket_of(last - 1) == BUCKET_COUNT - 2
    assert bucket_of(last) == bucket_of(last * 100) == BUCKET_COUNT - 1

    stats = RequestStats()
    stats.record("/slow", last + 1, error=False)
    stats.record("/slow", last * 10, error=True)
    stats.record("/fast", 0.4, error=False)
    stats.record(None, 3.0, error=False)  # a request that matched no route counts for nothing

    snapshot = stats.snapshot_and_reset()
    assert set(snapshot) == {"/slow", "/fast"}
    slow = snapshot["/slow"]
    assert len(slow.histogram) == BUCKET_COUNT
    assert (slow.count, slow.errors) == (2, 1)
    assert slow.histogram[-1] == 2 and sum(slow.histogram) == 2
    # The overflow bucket has no end, so both percentiles report the last edge.
    assert percentile(slow.histogram, 0.5) == percentile(slow.histogram, 0.95) == 60_000
    assert snapshot["/fast"].histogram[0] == 1

    assert stats.snapshot_and_reset() == {}  # the counters were emptied by the first snapshot


def test_records_from_several_threads_are_all_kept() -> None:
    stats = RequestStats()
    threads, each = 8, 500

    def hammer(index: int) -> None:
        for n in range(each):
            stats.record(f"/api/v1/t{index % 3}", float(n), error=n % 10 == 0)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(hammer, range(threads)))

    snapshot = stats.snapshot_and_reset()
    assert set(snapshot) == {"/api/v1/t0", "/api/v1/t1", "/api/v1/t2"}
    assert sum(counts.count for counts in snapshot.values()) == threads * each
    assert sum(counts.errors for counts in snapshot.values()) == threads * (each // 10)
    assert all(sum(counts.histogram) == counts.count for counts in snapshot.values())
    assert all(counts.max_ms == each - 1 for counts in snapshot.values())


def test_percentile_of_an_empty_histogram_is_zero_and_q_is_bounded() -> None:
    assert percentile([0] * BUCKET_COUNT, 0.5) == 0
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1] + [0] * (BUCKET_COUNT - 1), 1.5)
    # One request in the first bucket and one in the overflow bucket: the extremes are the
    # start of the first and the last edge for the overflow.
    extremes = [0] * BUCKET_COUNT
    extremes[0] = extremes[-1] = 1
    assert percentile(extremes, 0.0) == 0
    assert percentile(extremes, 1.0) == 60_000


def test_a_percentile_never_exceeds_the_slowest_request(sessions: SessionFactory) -> None:
    """A percentile read off the buckets is an estimate. With one request in a wide bucket its
    interpolation passes the true maximum, which would read as nonsense beside the slowest."""
    stats = RequestStats()
    _fill(stats, HEALTH, [36.0])
    snapshot = stats.snapshot_and_reset()
    # the estimate on its own overshoots: the only request there was took 36 ms
    assert percentile(snapshot[HEALTH].histogram, 0.95) > 36

    with sessions() as db:
        merge_into_daily(db, snapshot, day=DAY)
        db.commit()
        row = db.scalar(select(RequestStatDaily))
    assert (row.count, row.max_ms, row.p50_ms, row.p95_ms) == (1, 36, 32, 36)
