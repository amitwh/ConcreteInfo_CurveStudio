"""The Auto-fit result: one outcome per candidate, ranked rows, JSON, re-ranking.

Every enumerated candidate ends up in :attr:`AutoFitResult.outcomes` with a kind and a reason,
so nothing that failed, timed out, was skipped or fitted exactly is ever hidden or ranked
under a made-up number. The ranked rows are *derived* from the outcomes by
:func:`rank_outcomes`, which is also what :meth:`AutoFitResult.ranked_by` calls: changing the
ranking criterion never refits anything and never needs the per-point fit vectors.
"""

import json
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from curvestudio.discovery.curves import RowCurve
from curvestudio.discovery.ranking import (
    OVERFIT_PARAMETER_GAP,
    OVERFIT_R2_GAP,
    SAME_CURVE_RTOL,
    find_same_curves,
    rank,
)
from curvestudio.discovery.types import (
    AutoFitConfig,
    CandidateMetrics,
    RankBy,
    RankedFit,
)
from curvestudio.fitting import FitResult
from curvestudio.fitting.metrics import FiniteFloat

OutcomeKind = Literal["converged", "not_converged", "failed", "timed_out", "skipped", "exact_fit"]

_FITTED = ("converged", "exact_fit")
_RANKS_EXACT_FITS = frozenset({RankBy.R2, RankBy.ADJ_R2, RankBy.RMSE, RankBy.CV_RMSE})
"""Orderings that need no information criterion, so an exact fit can take part in them."""


class CandidateOutcome(BaseModel):
    """What happened to one candidate.

    ``reason`` explains every kind except a plain ``converged`` one (where it is empty unless
    cross-validation was incomplete). ``fit`` is kept for every candidate that was fitted,
    including the ones that did not converge, so the parameters it stopped at can be
    inspected. ``same_curve_as`` lists the fitted candidates whose curve coincides with
    this one. ``stage`` is ``"screen"`` for a candidate fitted from only a few starts that
    was not promoted to the full multistart (a later outcome of the same candidate with
    stage ``"final"`` replaces it); ``on_subsample`` marks a candidate fitted on the
    deterministic subsample of a very large dataset only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    name: str = ""
    category: str = ""
    outcome: OutcomeKind
    reason: str = ""
    fit: FitResult | None = None
    metrics: CandidateMetrics | None = None
    elapsed_seconds: FiniteFloat = 0.0
    same_curve_as: tuple[str, ...] = ()
    stage: Literal["screen", "final"] = "final"
    generated: bool = False
    shape_parameters: int = 0
    on_subsample: bool = False


EXACT_RELATIVE_TOLERANCE = 1e-6
"""Data are exact when the best fit's RMSE is below this fraction of the y range."""

EXACT_SAME_CURVE_RTOL = 1e-6
"""In the exact regime two curves coincide within this fraction of the data range (the
optimizer stops at a different point of the flat valley for each model)."""


def _y_range(outcomes: Sequence[CandidateOutcome]) -> float | None:
    """Peak-to-peak of the response, recovered from the fitted values plus residuals."""
    for o in outcomes:
        if o.fit is not None and o.fit.fitted and len(o.fit.fitted) == len(o.fit.residuals):
            y = np.asarray(o.fit.fitted, dtype=np.float64) + np.asarray(
                o.fit.residuals, dtype=np.float64
            )
            return float(np.ptp(y))
    return None


def _regime_exact(outcomes: Sequence[CandidateOutcome], y_range: float) -> tuple[bool, float]:
    """Whether the data are reproduced exactly, and the RMSE at or below which a fit counts."""
    rmses = [
        o.metrics.rmse
        for o in outcomes
        if o.outcome in _FITTED and o.metrics is not None and o.metrics.rmse is not None
    ]
    if not rmses:
        return False, 0.0
    best = min(rmses)
    if best > EXACT_RELATIVE_TOLERANCE * y_range:
        return False, 0.0
    return True, max(10.0 * best, EXACT_RELATIVE_TOLERANCE * y_range)


def resolve_exactness(
    outcomes: Sequence[CandidateOutcome], y_range: float | None = None
) -> list[CandidateOutcome]:
    """Decide, for the whole dataset, whether it is reproduced exactly, and mark exact fits.

    The data are *exact* when some fitted candidate has ``SSE = 0`` or an RMSE of at most
    ``1e-6`` of the y range (noiseless data; rounding and optimizer tolerance leave residuals
    of 1e-8 to 1e-14, never exactly zero). In that regime every converged candidate whose RMSE
    is at most ``max(10 * best RMSE, 1e-6 * range)`` becomes an ``exact_fit`` with no AICc
    or BIC: with the noise gone, information criteria only measure rounding dust and would
    rank a degenerate near-copy ahead of the model that generated the data. Outside it the
    outcomes are returned unchanged. Subsample outcomes and full-data outcomes of a very large
    run are decided separately. ``y_range`` defaults to the one implied by the fits.
    """
    result: list[CandidateOutcome] = []
    for on_subsample in (False, True):
        group = [o for o in outcomes if o.on_subsample == on_subsample]
        if not group:
            continue
        spread = y_range if y_range is not None else _y_range(group)
        exact, limit = (False, 0.0) if spread is None else _regime_exact(group, spread)
        for o in group:
            if (
                exact
                and o.outcome == "converged"
                and o.metrics is not None
                and o.metrics.rmse is not None
                and o.metrics.rmse <= limit
            ):
                note = (
                    "exact fit (residuals at rounding level, RMSE "
                    f"{o.metrics.rmse:.2g}): information criteria are not defined"
                )
                o = o.model_copy(
                    update={
                        "outcome": "exact_fit",
                        "metrics": o.metrics.model_copy(update={"aicc": None, "bic": None}),
                        "reason": "; ".join(filter(None, (note, o.reason))),
                    }
                )
            result.append(o)
    by_id = {o.candidate_id: o for o in result}
    return [by_id[o.candidate_id] for o in outcomes]


def annotate_same_curves(outcomes: Sequence[CandidateOutcome]) -> list[CandidateOutcome]:
    """Fill ``same_curve_as`` from the fitted values of the outcomes that carry them.

    Curves count as coinciding when their fitted values agree within ``1e-8`` of the data
    range at every point (``1e-6`` when any outcome is an exact fit: in the noiseless regime
    the optimizer stops anywhere in a flat valley). Outcomes without per-point values (a
    slimmed result) and every other outcome keep what they had. Candidates are first grouped
    by the sum of their fitted values, so the pairwise comparison only runs inside groups that
    could match; only vectors of the same length are compared.
    """
    exact = any(o.outcome == "exact_fit" for o in outcomes)
    rtol = EXACT_SAME_CURVE_RTOL if exact else SAME_CURVE_RTOL
    by_size: dict[int, dict[str, np.ndarray[Any, np.dtype[np.float64]]]] = {}
    scale = 0.0
    for o in outcomes:
        if o.outcome in _FITTED and o.fit is not None and o.fit.fitted:
            values = np.asarray(o.fit.fitted, dtype=np.float64)
            by_size.setdefault(values.size, {})[o.candidate_id] = values
            if scale == 0.0:
                scale = float(np.ptp(values + np.asarray(o.fit.residuals, dtype=np.float64)))
    same: dict[str, tuple[str, ...]] = {}
    for fitted in by_size.values():
        if len(fitted) < 2:
            continue
        same.update(dict.fromkeys(fitted, ()))
        window = rtol * scale * next(iter(fitted.values())).size
        order = sorted(fitted, key=lambda i: float(fitted[i].sum()))
        run: list[str] = [order[0]]
        for previous, current in pairwise(order):
            if float(fitted[current].sum()) - float(fitted[previous].sum()) > window:
                _compare(run, fitted, scale, rtol, same)
                run = []
            run.append(current)
        _compare(run, fitted, scale, rtol, same)
    return [
        o.model_copy(update={"same_curve_as": same[o.candidate_id]})
        if o.candidate_id in same
        else o
        for o in outcomes
    ]


def _compare(
    ids: list[str],
    fitted: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    scale: float,
    rtol: float,
    same: dict[str, tuple[str, ...]],
) -> None:
    if len(ids) > 1:
        same.update(find_same_curves({i: fitted[i] for i in ids}, scale, rtol))


def _r2_definition(o: CandidateOutcome) -> str:
    stats = None if o.fit is None else o.fit.stats
    return "centered" if stats is None else stats.r2_definition


def _overfitting_note(
    row: RankedFit, others: Sequence[tuple[RankedFit, str]], kind: str
) -> str | None:
    """Name the simplest candidate with far fewer parameters and almost the same R2."""
    if row.metrics.r2 is None:
        return None
    close = [
        other
        for other, other_kind in others
        if other_kind == kind
        and other.model_id != row.model_id
        and other.model_id not in row.same_curve_as
        and other.status != "exact_fit"
        and other.metrics.r2 is not None
        and row.metrics.k - other.metrics.k >= OVERFIT_PARAMETER_GAP
        and abs(row.metrics.r2 - other.metrics.r2) <= OVERFIT_R2_GAP + 1e-12
    ]
    if not close:
        return None
    simplest = min(close, key=lambda r: (r.metrics.k, -(r.metrics.r2 or 0.0), r.model_id))
    head = (
        f"{row.name} has the lowest AICc, which already charges it for its {row.metrics.k} "
        f"parameters, but {simplest.name} ({simplest.model_id}) reaches R2 "
        f"{simplest.metrics.r2:.4f} with only {simplest.metrics.k} (against "
        f"{row.metrics.r2:.4f}); treat the preference as weak and check cross-validation."
        if row.delta_aicc == 0.0
        else f"{row.name} uses {row.metrics.k} parameters (R2 {row.metrics.r2:.4f}), but "
        f"{simplest.name} ({simplest.model_id}) reaches R2 {simplest.metrics.r2:.4f} with only "
        f"{simplest.metrics.k}; the extra parameters may be fitting noise, so compare AICc or "
        "cross-validation before preferring it."
    )
    return head


def _rows(outcomes: Sequence[CandidateOutcome]) -> list[RankedFit]:
    """One unranked row per fitted outcome, with Akaike deltas and weights and notes."""
    base: list[tuple[RankedFit, str]] = []
    for o in outcomes:
        if o.outcome not in _FITTED or o.metrics is None or o.fit is None:
            continue
        row = RankedFit(
            rank=0,
            model_id=o.candidate_id,
            name=o.name or o.candidate_id,
            category=o.category,
            equation_fitted=o.fit.equation_fitted,
            parameters=o.fit.parameters,
            metrics=o.metrics,
            same_curve_as=o.same_curve_as,
            status="exact_fit" if o.outcome == "exact_fit" else o.fit.status.value,
            warnings=o.fit.warnings,
            generated=o.generated,
            shape_parameters=o.shape_parameters,
            degenerate=any(w.startswith("DEGENERATE_FIT") for w in o.fit.warnings),
        )
        base.append((row, _r2_definition(o)))
    aiccs = [r.metrics.aicc for r, _ in base if r.metrics.aicc is not None]
    best = min(aiccs) if aiccs else None
    if best is not None:
        raw = {
            r.model_id: math.exp(-(r.metrics.aicc - best) / 2.0)
            for r, _ in base
            if r.metrics.aicc is not None
        }
        total = sum(raw.values())
        base = [
            (
                r.model_copy(
                    update={
                        "delta_aicc": r.metrics.aicc - best,
                        "akaike_weight": raw[r.model_id] / total,
                    }
                )
                if r.metrics.aicc is not None
                else r,
                kind,
            )
            for r, kind in base
        ]
    return [
        row.model_copy(update={"overfitting_note": _overfitting_note(row, base, kind)})
        for row, kind in base
    ]


def rank_outcomes(outcomes: Sequence[CandidateOutcome], rank_by: RankBy) -> tuple[RankedFit, ...]:
    """The ranked rows of the fitted ``outcomes`` under ``rank_by``.

    Only fitted candidates (``converged`` and ``exact_fit``) become rows. An exact fit has no
    AICc or BIC, so it takes part only in the orderings that do not need one (best overall,
    R2, adjusted R2, RMSE, CV-RMSE), where the exact fits form a block above the rest; under
    ``AICC`` and ``BIC`` it stays out of the list and is reported through
    :attr:`AutoFitResult.exact_fits`. ``same_curve_as`` is read from the outcomes; pass them
    through :func:`resolve_exactness` and :func:`annotate_same_curves` first when they come
    straight from the engine. Candidates fitted on the full data of a very large run are
    ranked above those fitted on its subsample only (their criteria are not comparable).
    """
    keep_exact = rank_by is not RankBy.AICC and rank_by is not RankBy.BIC
    ranked: list[RankedFit] = []
    for on_subsample in (False, True):
        rows = _rows([o for o in outcomes if o.on_subsample == on_subsample])
        block = rank([r for r in rows if keep_exact or r.status != "exact_fit"], rank_by)
        offset = len(ranked)
        ranked.extend(
            r.model_copy(update={"rank": r.rank + offset}) if offset else r for r in block
        )
    return tuple(ranked)


def partial_ranking(outcomes: Sequence[CandidateOutcome], rank_by: RankBy) -> tuple[RankedFit, ...]:
    """Rank the outcomes seen so far (for example while streaming from ``iter_autofit``).

    A candidate that appears twice (a screening outcome, later replaced by its full multistart
    outcome) counts once, with its latest outcome.
    """
    latest = {o.candidate_id: o for o in outcomes}
    return rank_outcomes(annotate_same_curves(resolve_exactness(list(latest.values()))), rank_by)


class AutoFitResult(BaseModel):
    """Everything one Auto-fit run found.

    ``ranked`` holds the converged candidates ordered by ``config.rank_by``; ``outcomes``
    holds every candidate (ordered by id) with its kind and reason; ``skipped`` maps a
    candidate the data cannot support (or that was never fitted because the run was
    cancelled) to its reasons; ``unconverged`` maps every candidate that was fitted but did
    not converge (not converged, failed, timed out) to the reason; ``exact_fits`` lists the
    candidates that reproduce the data exactly (no information criteria exist for them).
    ``curves`` holds the sampled curve and bands of the top ``config.curve_top_n`` candidates
    under every ranking criterion, so a UI can plot any ordering without refitting.
    ``cancelled`` is true for a partial result. ``subsample_n`` is the size of the subsample
    a very large dataset (over 5000 active points) was screened on, else ``None``; the per-point
    vectors of such a result are kept only for the rows that have a curve.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: AutoFitConfig
    dataset_fingerprint: str
    n_active: int
    ranked: tuple[RankedFit, ...]
    outcomes: tuple[CandidateOutcome, ...]
    skipped: dict[str, tuple[str, ...]]
    unconverged: dict[str, str]
    exact_fits: tuple[str, ...]
    warnings: tuple[str, ...]
    cancelled: bool
    curves: dict[str, RowCurve] = Field(default_factory=dict)
    overfitting_note: str | None = None
    elapsed_seconds: FiniteFloat = 0.0
    subsample_n: int | None = None

    @property
    def failed(self) -> dict[str, str]:
        """Candidates that raised or whose optimizer failed, with the reason."""
        return {o.candidate_id: o.reason for o in self.outcomes if o.outcome == "failed"}

    @property
    def timed_out(self) -> dict[str, str]:
        """Candidates stopped by the time limit before converging, with the reason."""
        return {o.candidate_id: o.reason for o in self.outcomes if o.outcome == "timed_out"}

    def ranked_by(self, rank_by: RankBy) -> tuple[RankedFit, ...]:
        """The same candidates ordered by ``rank_by``; no refitting happens."""
        return rank_outcomes(self.outcomes, rank_by)

    def to_json(self, *, include_fits: bool = True) -> str:
        """Standard JSON (non-finite numbers are rejected, floats keep full precision).

        ``include_fits=False`` drops each fit's per-point vectors (``fitted``, ``residuals``)
        and covariance, which dominate the size for large data; rows, metrics, parameters,
        reasons and curves are kept, and :meth:`ranked_by` still works on the restored result.

        Raises:
            ValueError: a non-finite number is present.
        """
        result = self if include_fits else self._slim()
        return json.dumps(result.model_dump(mode="json"), allow_nan=False)

    def _slim(self) -> Self:
        def slim(o: CandidateOutcome) -> CandidateOutcome:
            if o.fit is None:
                return o
            bare = o.fit.model_copy(
                update={"fitted": [], "residuals": [], "covariance": None, "correlation": None}
            )
            return o.model_copy(update={"fit": bare})

        return self.model_copy(update={"outcomes": tuple(slim(o) for o in self.outcomes)})

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse :meth:`to_json` output; invalid JSON or non-finite values raise ``ValueError``."""

        def reject_constant(token: str) -> Any:
            raise ValueError(f"non-finite JSON constant {token!r} is not allowed")

        return cls.model_validate(json.loads(text, parse_constant=reject_constant))
