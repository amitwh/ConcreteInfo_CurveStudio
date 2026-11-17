"""The Auto-fit result: one outcome per candidate, ranked rows, JSON, re-ranking.

Every enumerated candidate ends up in :attr:`AutoFitResult.outcomes` with a kind and a reason,
so nothing that failed, timed out, was skipped or fitted exactly is ever hidden or ranked
under a made-up number. The ranked rows are *derived* from the outcomes by
:func:`rank_outcomes`, which is also what :meth:`AutoFitResult.ranked_by` calls: changing the
ranking criterion never refits anything and never needs the per-point fit vectors.
"""

import json
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
    this one.
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


def annotate_same_curves(outcomes: Sequence[CandidateOutcome]) -> list[CandidateOutcome]:
    """Fill ``same_curve_as`` from the fitted values of the outcomes that carry them.

    Curves count as coinciding when their fitted values agree within ``1e-8`` of the data
    range at every point. Outcomes without per-point values (a slimmed result) and every
    other outcome keep what they had. Candidates are first grouped by the sum of their
    fitted values, so the pairwise comparison only runs inside groups that could match.
    """
    fitted: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    scale = 0.0
    for o in outcomes:
        if o.outcome in _FITTED and o.fit is not None and o.fit.fitted:
            values = np.asarray(o.fit.fitted, dtype=np.float64)
            fitted[o.candidate_id] = values
            if scale == 0.0:
                scale = float(np.ptp(values + np.asarray(o.fit.residuals, dtype=np.float64)))
    if len(fitted) < 2:
        return list(outcomes)
    same: dict[str, tuple[str, ...]] = dict.fromkeys(fitted, ())
    window = SAME_CURVE_RTOL * scale * next(iter(fitted.values())).size
    order = sorted(fitted, key=lambda i: float(fitted[i].sum()))
    run: list[str] = [order[0]]
    for previous, current in pairwise(order):
        if float(fitted[current].sum()) - float(fitted[previous].sum()) > window:
            _compare(run, fitted, scale, same)
            run = []
        run.append(current)
    _compare(run, fitted, scale, same)
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
    same: dict[str, tuple[str, ...]],
) -> None:
    if len(ids) > 1:
        same.update(find_same_curves({i: fitted[i] for i in ids}, scale))


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
        and other.metrics.r2 is not None
        and row.metrics.k - other.metrics.k >= OVERFIT_PARAMETER_GAP
        and abs(row.metrics.r2 - other.metrics.r2) <= OVERFIT_R2_GAP + 1e-12
    ]
    if not close:
        return None
    simplest = min(close, key=lambda r: (r.metrics.k, -(r.metrics.r2 or 0.0), r.model_id))
    return (
        f"{row.name} uses {row.metrics.k} parameters (R2 {row.metrics.r2:.4f}), but "
        f"{simplest.name} ({simplest.model_id}) reaches R2 {simplest.metrics.r2:.4f} with only "
        f"{simplest.metrics.k}; the extra parameters may be fitting noise, so compare AICc or "
        "cross-validation before preferring it."
    )


def _rows(outcomes: Sequence[CandidateOutcome]) -> list[tuple[RankedFit, bool]]:
    """One unranked row per fitted outcome, with whether it is an exact fit."""
    base: list[tuple[RankedFit, bool, str]] = []
    for o in outcomes:
        if o.outcome not in _FITTED or o.metrics is None or o.fit is None:
            continue
        exact = o.outcome == "exact_fit"
        row = RankedFit(
            rank=0,
            model_id=o.candidate_id,
            name=o.name or o.candidate_id,
            category=o.category,
            equation_fitted=o.fit.equation_fitted,
            parameters=o.fit.parameters,
            metrics=o.metrics,
            same_curve_as=o.same_curve_as,
            status="exact_fit" if exact else o.fit.status.value,
            warnings=o.fit.warnings,
        )
        base.append((row, exact, _r2_definition(o)))
    pool = [(row, kind) for row, _, kind in base]
    return [
        (row.model_copy(update={"overfitting_note": _overfitting_note(row, pool, kind)}), exact)
        for row, exact, kind in base
    ]


def rank_outcomes(outcomes: Sequence[CandidateOutcome], rank_by: RankBy) -> tuple[RankedFit, ...]:
    """The ranked rows of the fitted ``outcomes`` under ``rank_by``.

    Only fitted candidates (``converged`` and ``exact_fit``) become rows. An exact fit has no
    AICc or BIC, so it takes part only in the orderings that need none (R2, adjusted R2, RMSE,
    CV-RMSE); under the others it stays out of the list and is reported through
    :attr:`AutoFitResult.exact_fits`. ``same_curve_as`` is read from the outcomes; pass them
    through :func:`annotate_same_curves` first when they come straight from the engine.
    """
    keep_exact = rank_by in _RANKS_EXACT_FITS
    rows = [row for row, exact in _rows(outcomes) if keep_exact or not exact]
    return rank(rows, rank_by)


def partial_ranking(outcomes: Sequence[CandidateOutcome], rank_by: RankBy) -> tuple[RankedFit, ...]:
    """Rank the outcomes seen so far (for example while streaming from ``iter_autofit``)."""
    return rank_outcomes(annotate_same_curves(outcomes), rank_by)


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
    ``cancelled`` is true for a partial result.
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
