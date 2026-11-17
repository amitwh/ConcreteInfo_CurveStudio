"""Order fitted candidates; pure functions over metrics, no fitting."""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from curvestudio.discovery.types import RankBy, RankedFit

TIE_DELTA = 2.0
"""Candidates within this many information-criterion units of the best form a tie group."""

OVERFIT_PARAMETER_GAP = 3
OVERFIT_R2_GAP = 0.01
SAME_CURVE_RTOL = 1e-8

_DESCENDING = {RankBy.R2: "r2", RankBy.ADJ_R2: "adj_r2"}
_ASCENDING = {
    RankBy.RMSE: "rmse",
    RankBy.AICC: "aicc",
    RankBy.BIC: "bic",
    RankBy.CV_RMSE: "cv_rmse",
}


def find_same_curves(
    fitted: Mapping[str, ArrayLike], scale: float, rtol: float = SAME_CURVE_RTOL
) -> dict[str, tuple[str, ...]]:
    """For each id, the other ids whose fitted values differ by at most ``rtol * scale``.

    ``fitted`` holds each candidate's fitted values at the same x; ``scale`` is the data
    range (y max minus y min; a non-positive or non-finite scale compares exactly).
    Candidates with differently sized vectors are never the same curve. The result has a
    key for every id (an empty tuple when no curve coincides), in sorted order.
    """
    tolerance = rtol * scale if math.isfinite(scale) and scale > 0.0 else 0.0
    ids = sorted(fitted)
    arrays = {i: np.asarray(fitted[i], dtype=np.float64) for i in ids}
    same: dict[str, list[str]] = {i: [] for i in ids}
    for pos, a in enumerate(ids):
        for b in ids[pos + 1 :]:
            xa, xb = arrays[a], arrays[b]
            if xa.shape == xb.shape and bool(np.all(np.abs(xa - xb) <= tolerance)):
                same[a].append(b)
                same[b].append(a)
    return {i: tuple(sorted(others)) for i, others in same.items()}


def _cv_key(row: RankedFit) -> tuple[bool, float]:
    cv = row.metrics.cv_rmse
    return (cv is None, 0.0 if cv is None else cv)


def _tie_order(row: RankedFit) -> tuple[bool, float, int, str]:
    """Inside a tie group: CV-RMSE (``None`` last), then fewer parameters, then id."""
    return (*_cv_key(row), row.metrics.k, row.model_id)


def _peel(rows: list[RankedFit], criterion: Callable[[RankedFit], float]) -> list[RankedFit]:
    """Repeatedly emit the tie group around the best remaining ``criterion`` (lower is better)."""
    remaining = sorted(rows, key=lambda r: (criterion(r), r.model_id))
    ordered: list[RankedFit] = []
    while remaining:
        best = criterion(remaining[0])
        group = [r for r in remaining if criterion(r) - best <= TIE_DELTA]
        ordered.extend(sorted(group, key=_tie_order))
        taken = {r.model_id for r in group}
        remaining = [r for r in remaining if r.model_id not in taken]
    return ordered


def best_overall_order(rows: Sequence[RankedFit]) -> list[RankedFit]:
    """Order ``rows`` best first by the AICc-led composite.

    Repeatedly take the best remaining AICc; every row within ``TIE_DELTA`` of it forms a
    tie group, ordered by CV-RMSE (``None`` last), then fewer parameters, then id; emit the
    group and repeat on the rest. Rows without an AICc come after all rows that have one
    and are peeled the same way on BIC; rows with neither (for example exact fits) come
    last in tie-group order. The input is not modified and its order does not matter.
    """
    with_aicc = [r for r in rows if r.metrics.aicc is not None]
    bic_only = [r for r in rows if r.metrics.aicc is None and r.metrics.bic is not None]
    neither = [r for r in rows if r.metrics.aicc is None and r.metrics.bic is None]
    return [
        *_peel(with_aicc, lambda r: float(r.metrics.aicc or 0.0)),
        *_peel(bic_only, lambda r: float(r.metrics.bic or 0.0)),
        *sorted(neither, key=_tie_order),
    ]


def _metric_order(rows: Sequence[RankedFit], rank_by: RankBy) -> list[RankedFit]:
    if rank_by in _DESCENDING:
        attr, sign = _DESCENDING[rank_by], -1.0
    else:
        attr, sign = _ASCENDING[rank_by], 1.0

    def key(row: RankedFit) -> tuple[bool, float, int, str]:
        value = getattr(row.metrics, attr)
        return (value is None, 0.0 if value is None else sign * value, row.metrics.k, row.model_id)

    return sorted(rows, key=key)


def _group_same_curves(ordered: list[RankedFit]) -> list[tuple[RankedFit, int]]:
    """Make rows of coinciding curves adjacent; pair each row with its group's first position."""
    position = {row.model_id: i for i, row in enumerate(ordered)}
    parent = {row.model_id: row.model_id for row in ordered}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in ordered:
        for other in row.same_curve_as:
            if other in parent:
                a, b = find(row.model_id), find(other)
                if a != b:
                    first, second = (a, b) if position[a] < position[b] else (b, a)
                    parent[second] = first
    members: dict[str, list[RankedFit]] = {}
    for row in ordered:
        members.setdefault(find(row.model_id), []).append(row)
    result: list[tuple[RankedFit, int]] = []
    for root in sorted(members, key=lambda r: position[r]):
        start = len(result)
        result.extend((row, start) for row in members[root])
    return result


def rank(rows: Sequence[RankedFit], rank_by: RankBy) -> tuple[RankedFit, ...]:
    """Sort ``rows`` by ``rank_by`` and number them from 1.

    ``R2``/``ADJ_R2`` rank high to low, ``RMSE``/``AICC``/``BIC``/``CV_RMSE`` low to high,
    each with ``None`` last and fewer parameters then id breaking ties; ``BEST_OVERALL`` uses
    :func:`best_overall_order`. Rows that list each other in ``same_curve_as`` stay adjacent
    and share the rank of the best of them (the next row's rank then skips the shared
    positions). Returns new rows; the inputs are not modified and the result does not depend
    on their order.
    """
    ordered = (
        best_overall_order(rows) if rank_by is RankBy.BEST_OVERALL else _metric_order(rows, rank_by)
    )
    return tuple(
        row.model_copy(update={"rank": start + 1}) for row, start in _group_same_curves(ordered)
    )


def overfitting_note(ranked: Sequence[RankedFit], rank_by: RankBy) -> str | None:
    """Warn when an R2-ranked top row buys little fit with many more parameters.

    Only for ``R2`` and ``ADJ_R2``: if the top row has at least 3 more parameters than
    another row whose ranking metric (R2 or adjusted R2) is within 0.01 of the top row's,
    return text naming both, choosing the simplest such row. Otherwise ``None``.
    """
    attr = _DESCENDING.get(rank_by)
    if attr is None or not ranked:
        return None
    top = ranked[0]
    top_value = getattr(top.metrics, attr)
    if top_value is None:
        return None
    simpler = [
        row
        for row in ranked[1:]
        if getattr(row.metrics, attr) is not None
        and top.metrics.k - row.metrics.k >= OVERFIT_PARAMETER_GAP
        and abs(top_value - getattr(row.metrics, attr)) <= OVERFIT_R2_GAP + 1e-12
    ]
    if not simpler:
        return None
    other = min(simpler, key=lambda r: (r.metrics.k, r.rank, r.model_id))
    label = "R2" if attr == "r2" else "adjusted R2"
    return (
        f"{top.name} ({top.model_id}) has the best {label} ({top_value:.4f}) with "
        f"{top.metrics.k} parameters, but {other.name} ({other.model_id}) reaches "
        f"{getattr(other.metrics, attr):.4f} with only {other.metrics.k}; the extra parameters "
        "may be fitting noise, so compare AICc or cross-validation before preferring it."
    )
