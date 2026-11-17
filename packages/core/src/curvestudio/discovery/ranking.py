"""Order fitted candidates; pure functions over metrics, no fitting."""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from curvestudio.discovery.types import RankBy, RankedFit
from curvestudio.models.starter import STARTER_MODELS

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


RELIABILITY_PENALTY = {
    "MULTISTART_DISAGREE": 6.0,
    "ILL_CONDITIONED": 4.0,
    "PARAMETER_AT_BOUND": 2.0,
}
"""AICc units added, for ordering only, for each reliability warning a row carries.

``MULTISTART_DISAGREE`` (several starts end at different optima: the likelihood is
multimodal, as for a free frequency or exponent) costs most: maximising over such a shape
parameter fits noise, and a free frequency is worth about three effective parameters, which
is 6 AICc units. A badly conditioned fit costs 4, a parameter at its bound 2 (common and
often legitimate, so only enough to break a tie). The reported AICc is never changed."""

_NON_GENERIC = ("engineering", "physics", "kinetics")
_CORE_IDS = frozenset(m.id for m in STARTER_MODELS)


SHAPE_PARAMETER_PENALTY = 2.0
"""AICc units per free-frequency parameter: it counts as one more effective parameter, which
AIC charges 2 for."""


def reliability_penalty(row: RankedFit) -> float:
    """The ordering-only AICc penalty of the reliability warnings the row's fit carries."""
    codes = {w.split(":", 1)[0].strip() for w in row.warnings}
    return sum(cost for code, cost in RELIABILITY_PENALTY.items() if code in codes)


def ranking_penalty(row: RankedFit) -> float:
    """Everything added to a row's AICc (or BIC) for ordering only: the reliability warnings
    of its fit plus :data:`SHAPE_PARAMETER_PENALTY` per free-frequency parameter.
    The reported AICc, ``delta_aicc`` and Akaike weights never include it."""
    return reliability_penalty(row) + SHAPE_PARAMETER_PENALTY * row.shape_parameters


def canonical_preference(row: RankedFit) -> int:
    """Lower is preferred as the headline of a group of same-curve rows: generic categories
    before engineering, physics and kinetics ones, hand-written before generated, and the
    core starter models (Michaelis-Menten, the exponentials, the Gaussian, ...: the names a
    user looks for) before the rest of the catalogue."""
    first = row.category.split(".", 1)[0]
    return 4 * (first in _NON_GENERIC) + 2 * int(row.generated) + (row.model_id not in _CORE_IDS)


def _cv_key(row: RankedFit) -> tuple[bool, float]:
    cv = row.metrics.cv_rmse
    return (cv is None, 0.0 if cv is None else cv)


def _tie_order(row: RankedFit) -> tuple[int, bool, float, int, str]:
    """Inside a tie group: fewer effective parameters (``k`` plus shape parameters), then
    CV-RMSE (``None`` last), then the canonical preference, then id.

    The group is by construction statistically indistinguishable on AICc, so the simplest
    model leads (Burnham and Anderson's rule for models within 2 units); cross-validation,
    which is noisy at the sample sizes where Auto-fit is used, only separates equally simple
    models. On seeded known-truth data (60 points, 2 % noise, five models, 12 noise draws
    each, the full library) this ordering put the generating model in the top three in 56
    of 60 runs against 50 of 60 for cross-validation first.
    """
    return (
        row.metrics.k + row.shape_parameters,
        *_cv_key(row),
        canonical_preference(row),
        row.model_id,
    )


def _peel(rows: list[RankedFit], criterion: Callable[[RankedFit], float]) -> list[RankedFit]:
    """Repeatedly emit the tie group around the best remaining ``criterion`` (lower is better).

    ``criterion`` is the AICc (or BIC) plus :func:`ranking_penalty`.
    """

    def adjusted(row: RankedFit) -> float:
        return criterion(row) + ranking_penalty(row)

    remaining = sorted(rows, key=lambda r: (adjusted(r), r.model_id))
    ordered: list[RankedFit] = []
    while remaining:
        best = adjusted(remaining[0])
        group = [r for r in remaining if adjusted(r) - best <= TIE_DELTA]
        ordered.extend(sorted(group, key=_tie_order))
        taken = {r.model_id for r in group}
        remaining = [r for r in remaining if r.model_id not in taken]
    return ordered


def best_overall_order(rows: Sequence[RankedFit]) -> list[RankedFit]:
    """Order ``rows`` best first by the AICc-led composite.

    Repeatedly take the best remaining AICc; every row within ``TIE_DELTA`` of it forms a
    tie group, ordered by fewer parameters, then CV-RMSE (``None`` last), then the canonical
    preference, then id; emit the group and repeat on the rest. A row with a reliability flag
    (``MULTISTART_DISAGREE``, ``ILL_CONDITIONED``, ``PARAMETER_AT_BOUND``) is charged
    :func:`reliability_penalty` in this comparison only, so it falls behind clean rows of
    about the same AICc; each free-frequency parameter is charged like an extra
    parameter (:data:`SHAPE_PARAMETER_PENALTY`), because maximising over it fits noise. Rows
    without an AICc come after all rows that have one and are peeled the same way on BIC; rows
    with neither come last in tie-group order. The input is not modified and its order does
    not matter.
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


def _exact_order(row: RankedFit) -> tuple[bool, int, bool, float, int, str]:
    """Order of exact fits: fewest parameters, then CV-RMSE, then canonical preference, id."""
    return (row.degenerate, row.metrics.k, *_cv_key(row), canonical_preference(row), row.model_id)


def _components(rows: Sequence[RankedFit]) -> list[list[RankedFit]]:
    """Connected groups of rows that list each other in ``same_curve_as``."""
    by_id = {r.model_id: r for r in rows}
    parent = {i: i for i in by_id}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in rows:
        for other in row.same_curve_as:
            if other in parent:
                parent[find(row.model_id)] = find(other)
    groups: dict[str, list[RankedFit]] = {}
    for row in rows:
        groups.setdefault(find(row.model_id), []).append(row)
    return list(groups.values())


def _arrange_same_curves(ordered: list[RankedFit]) -> list[tuple[RankedFit, int]]:
    """Tie the same-curve rows that have equally few parameters; flag the others redundant.

    In each group of coinciding curves the canonical row has the fewest parameters, then the
    best canonical preference, then the smallest id. Rows with that same parameter count tie
    with it: they are made adjacent, canonical first, at the position of the earliest of them,
    and share its rank. Rows with more parameters draw the same curve with needless extra
    freedom: they are not tied but keep their own position and carry ``redundant_with``.
    Returns each row with the position (0-based) whose ``+ 1`` is its rank.
    """
    position = {row.model_id: i for i, row in enumerate(ordered)}
    redundant: dict[str, str] = {}
    blocks: dict[str, list[RankedFit]] = {}
    for group in _components(ordered):
        if len(group) == 1:
            continue
        canonical = min(group, key=lambda r: (r.metrics.k, canonical_preference(r), r.model_id))
        tied = [r for r in group if r.metrics.k == canonical.metrics.k]
        tied.sort(
            key=lambda r: (r.model_id != canonical.model_id, canonical_preference(r), r.model_id)
        )
        anchor = min(position[r.model_id] for r in tied)
        blocks[ordered[anchor].model_id] = tied
        for r in group:
            if r.metrics.k > canonical.metrics.k:
                redundant[r.model_id] = canonical.model_id
    in_block = {r.model_id for tied in blocks.values() for r in tied}
    result: list[tuple[RankedFit, int]] = []
    for row in ordered:
        if row.model_id in blocks:
            start = len(result)
            result.extend((r, start) for r in blocks[row.model_id])
        elif row.model_id not in in_block:
            result.append((row, len(result)))
    return [
        (r.model_copy(update={"redundant_with": redundant[r.model_id]}), pos)
        if r.model_id in redundant
        else (r, pos)
        for r, pos in result
    ]


def rank(rows: Sequence[RankedFit], rank_by: RankBy) -> tuple[RankedFit, ...]:
    """Sort ``rows`` by ``rank_by`` and number them from 1.

    Exact fits (status ``exact_fit``) form a block at the top, ordered by fewer parameters,
    then CV-RMSE, then canonical preference, then id; they have no AICc or BIC, so callers
    leave them out of ``AICC``/``BIC`` rankings. After them come the non-degenerate rows
    and then the degenerate ones. ``R2``/``ADJ_R2`` rank high to low, ``RMSE``/``AICC``/
    ``BIC``/``CV_RMSE`` low to high, each with ``None`` last and fewer parameters then id
    breaking ties; ``BEST_OVERALL`` uses :func:`best_overall_order`. Rows that list each
    other in ``same_curve_as`` and have equally few parameters stay adjacent, canonical
    first, and share the best rank (the next row's rank skips the shared positions); rows
    with more parameters on the same curve keep their place and are marked ``redundant_with``
    the canonical row. Returns new rows; the inputs are not modified and the result does not
    depend on their order.
    """

    def ordered_block(block: list[RankedFit]) -> list[RankedFit]:
        if rank_by is RankBy.BEST_OVERALL:
            return best_overall_order(block)
        return _metric_order(block, rank_by)

    exact = sorted((r for r in rows if r.status == "exact_fit"), key=_exact_order)
    normal = [r for r in rows if r.status != "exact_fit"]
    ordered = [
        *exact,
        *ordered_block([r for r in normal if not r.degenerate]),
        *ordered_block([r for r in normal if r.degenerate]),
    ]
    return tuple(
        row.model_copy(update={"rank": start + 1}) for row, start in _arrange_same_curves(ordered)
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
