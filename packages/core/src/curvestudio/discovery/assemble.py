"""Turn the outcomes of a run into its :class:`AutoFitResult`: skipped lists, ranking, curves,
warnings. Pure bookkeeping on the outcomes and the data; nothing is fitted here except the
sampled curves of the top rows."""

from typing import TYPE_CHECKING

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery.candidates import resolve_spec
from curvestudio.discovery.curves import RowCurve, sample_curve
from curvestudio.discovery.ranking import overfitting_note
from curvestudio.discovery.result import (
    AutoFitResult,
    CandidateOutcome,
    annotate_same_curves,
    rank_outcomes,
    resolve_exactness,
)
from curvestudio.discovery.types import CandidateRef, RankBy, RankedFit
from curvestudio.discovery.worker import describe, is_generated
from curvestudio.statistics import StatisticsError, build_view

if TYPE_CHECKING:
    from curvestudio.discovery.engine import Plan

_UNCONVERGED = ("not_converged", "failed", "timed_out")

MAX_LISTED_CAPPED = 50
"""More capped-out candidates than this are reported as one summary entry."""

CAPPED_SUMMARY_KEY = "(parameter cap)"

OUTLIER_SIGMA = 5.0
"""A studentised residual of the best curve above this raises an outlier warning."""

EXTREME_SCALE = 1e6
"""A data span above this, or below its reciprocal, raises the scale warning."""


def _unfitted(ref: CandidateRef, reason: str) -> CandidateOutcome:
    spec = resolve_spec(ref)
    name, category = describe(ref, spec)
    return CandidateOutcome(
        candidate_id=ref.id,
        name=name,
        category=category,
        outcome="skipped",
        reason=reason,
        generated=is_generated(ref),
    )


def _capped_skipped(plan: Plan) -> dict[str, tuple[str, ...]]:
    """``skipped`` entries for the candidates the parameter cap left out (never fitted, so not
    outcomes): one per model, or a single summary when there are many."""
    if len(plan.capped) <= MAX_LISTED_CAPPED:
        return {
            cid: (f"needs {k} parameters; cap is {plan.cap} for {plan.n_active} points",)
            for cid, k in sorted(plan.capped.items())
        }
    sample = ", ".join(sorted(plan.capped)[:8])
    return {
        CAPPED_SUMMARY_KEY: (
            f"{len(plan.capped)} models need more than {plan.cap} parameters (cap is {plan.cap} "
            f"for {plan.n_active} points); raise max_parameters or choose families to try them. "
            f"Examples: {sample}",
        )
    }


def _y_span(data: XY) -> float:
    y = data.y if data.weights is None else data.y[data.weights > 0]
    return float(np.ptp(y))


def assemble(
    plan: Plan, latest: dict[str, CandidateOutcome], cancelled: bool, elapsed: float
) -> AutoFitResult:
    """Build the result from the latest outcome of every attempted candidate."""
    config = plan.config
    by_id = dict(latest)
    for cid, reasons in plan.skipped.items():
        by_id[cid] = _unfitted(plan.refs[cid], "; ".join(reasons))
    never_fitted = [job.ref.id for job in plan.jobs if job.ref.id not in by_id]
    for cid in never_fitted:
        by_id[cid] = _unfitted(plan.refs[cid], "the run was cancelled before this was fitted")

    resolved = resolve_exactness(list(by_id.values()), _y_span(plan.data))
    ordered = sorted(annotate_same_curves(resolved), key=lambda o: o.candidate_id)
    ranked = rank_outcomes(ordered, config.rank_by)

    skipped = {
        o.candidate_id: plan.skipped.get(o.candidate_id, (o.reason,))
        for o in ordered
        if o.outcome == "skipped"
    }
    skipped.update(_capped_skipped(plan))
    unconverged = {o.candidate_id: o.reason for o in ordered if o.outcome in _UNCONVERGED}
    exact_fits = tuple(o.candidate_id for o in ordered if o.outcome == "exact_fit")
    curves, notes = _sample_curves(plan, ordered)
    notes.extend(_outlier_notes(plan, ordered, ranked))
    if plan.subsampled:
        ordered = _slim_vectors(ordered, keep=set(curves))

    return AutoFitResult(
        config=config,
        dataset_fingerprint=plan.data.fingerprint,
        n_active=plan.n_active,
        ranked=ranked,
        outcomes=tuple(ordered),
        skipped=skipped,
        unconverged=unconverged,
        exact_fits=exact_fits,
        warnings=_warnings(plan, ordered, cancelled, never_fitted, notes),
        cancelled=cancelled,
        curves=curves,
        overfitting_note=overfitting_note(ranked, config.rank_by),
        elapsed_seconds=elapsed,
        subsample_n=int(plan.screen_data.x.size) if plan.subsampled else None,
    )


def _slim_vectors(outcomes: list[CandidateOutcome], keep: set[str]) -> list[CandidateOutcome]:
    """Drop the per-point vectors of every fit not in ``keep``: at 100k points they are what
    fills memory and JSON, and only the top rows are drawn."""
    result: list[CandidateOutcome] = []
    for o in outcomes:
        if o.fit is None or o.candidate_id in keep or not o.fit.fitted:
            result.append(o)
            continue
        bare = o.fit.model_copy(
            update={"fitted": [], "residuals": [], "covariance": None, "correlation": None}
        )
        result.append(o.model_copy(update={"fit": bare}))
    return result


def _sample_curves(
    plan: Plan, ordered: list[CandidateOutcome]
) -> tuple[dict[str, RowCurve], list[str]]:
    """Curves and bands of the top rows under every ranking criterion (one copy each)."""
    top_n = plan.config.curve_top_n
    if top_n == 0:
        return {}, []
    wanted = sorted({r.model_id for rb in RankBy for r in rank_outcomes(ordered, rb)[:top_n]})
    by_id = {o.candidate_id: o for o in ordered}
    curves: dict[str, RowCurve] = {}
    notes: list[str] = []
    for cid in wanted:
        outcome = by_id[cid]
        fit = outcome.fit
        assert fit is not None  # only fitted outcomes become rows
        data = plan.screen_data if outcome.on_subsample else plan.data
        try:
            curve, reason = sample_curve(data, fit, resolve_spec(plan.refs[cid]))
        except Exception as exc:  # one undrawable curve must not lose the whole result
            curve, reason = None, f"{type(exc).__name__}: {exc}"
        if curve is None:
            notes.append(f"no curve was sampled for {cid}: {reason}")
        else:
            curves[cid] = curve
    return curves, notes


def _outlier_notes(
    plan: Plan, ordered: list[CandidateOutcome], ranked: tuple[RankedFit, ...]
) -> list[str]:
    """Warn about points more than 5 studentised residuals from the best-ranked curve."""
    top = next((r for r in ranked if r.status != "exact_fit"), None)
    if top is None:
        return []
    outcome = next(o for o in ordered if o.candidate_id == top.model_id)
    fit = outcome.fit
    data = plan.screen_data if outcome.on_subsample else plan.data
    if fit is None or fit.stats is None or fit.stats.residual_se in (None, 0.0) or not fit.fitted:
        return []
    try:
        view = build_view(data, fit, resolve_spec(plan.refs[top.model_id]))
        jac = view.weighted_jacobian
        gram = np.linalg.pinv(jac.T @ jac)
        leverage = np.einsum("ij,jk,ik->i", jac, gram, jac)
    except StatisticsError, np.linalg.LinAlgError, ValueError:
        return []
    scale = float(fit.stats.residual_se or 0.0)
    with np.errstate(all="ignore"):
        student = (
            np.sqrt(view.w) * view.residuals / (scale * np.sqrt(np.maximum(1.0 - leverage, 1e-12)))
        )
    worst = int(np.argmax(np.abs(student)))
    if not np.isfinite(student[worst]) or abs(student[worst]) <= OUTLIER_SIGMA:
        return []
    return [
        f"possible outlier: the point at x = {view.x[worst]:.6g}, y = {view.y[worst]:.6g} lies "
        f"{abs(student[worst]):.1f} studentised residuals from the top-ranked curve "
        f"({top.model_id}); check it before trusting the ranking"
    ]


def _scale_note(data: XY) -> str | None:
    x = data.x if data.weights is None else data.x[data.weights > 0]
    spans = {"x": float(np.ptp(x)), "y": _y_span(data)}
    extreme = {
        axis: span
        for axis, span in spans.items()
        if span > 0 and (span > EXTREME_SCALE or span < 1.0 / EXTREME_SCALE)
    }
    if not extreme:
        return None
    listed = ", ".join(f"{axis} spans {span:.3g}" for axis, span in extreme.items())
    return (
        f"data scale is extreme ({listed}): fits can lose precision and many models may fail to "
        "converge; consider rescaling the data (for example to order one) and refitting"
    )


def _warnings(
    plan: Plan,
    ordered: list[CandidateOutcome],
    cancelled: bool,
    never_fitted: list[str],
    curve_notes: list[str],
) -> tuple[str, ...]:
    config, data = plan.config, plan.data
    warnings: list[str] = []
    if cancelled:
        warnings.append(
            f"the run was cancelled: {len(never_fitted)} of {len(plan.jobs)} candidates were "
            "never fitted, so the ranking is partial"
        )
    if plan.subsampled:
        warnings.append(
            f"{plan.n_active} active points: candidates were screened on a deterministic "
            f"subsample of {plan.screen_data.x.size} points and the best of each ranking "
            "criterion were refitted on all points; rows fitted on all points rank above the "
            "rest, whose numbers come from the subsample only"
        )
    if plan.n_active < 10:
        warnings.append(
            f"only {plan.n_active} active points: rankings from so few points are unreliable"
        )
    note = _scale_note(data)
    if note is not None:
        warnings.append(note)
    if data.weights is not None:
        zero = int((data.weights <= 0.0).sum())
        warnings.append(
            "the data carries weights: every candidate was fitted weighted, so information "
            "criteria and RMSE are comparable only with each other"
            + (f"; {zero} zero-weight point(s) were left out" if zero else "")
        )
    timed_out = [o.candidate_id for o in ordered if o.outcome == "timed_out"]
    if timed_out:
        warnings.append(
            f"{len(timed_out)} candidate(s) reached the time limit of {config.time_limit:g} s "
            "before converging (or were stopped by the watchdog) and are not ranked"
        )
    late = [
        o.candidate_id
        for o in ordered
        if o.outcome in ("converged", "exact_fit")
        and o.fit is not None
        and any(w.startswith("time limit reached") for w in o.fit.warnings)
    ]
    if late:
        warnings.append(
            f"{len(late)} candidate(s) converged but hit the time limit afterwards, so their "
            "multistart search is incomplete (see each row's warnings)"
        )
    exact = [o.candidate_id for o in ordered if o.outcome == "exact_fit"]
    if exact:
        warnings.append(
            f"the data are reproduced exactly (noiseless): {len(exact)} candidate(s) fit them to "
            "rounding level. No AICc or BIC exists for them; they are listed in exact_fits and "
            "ranked first by number of parameters, then cross-validation, not by information "
            "criteria"
        )
    degenerate = [
        o.candidate_id
        for o in ordered
        if o.fit is not None and any(w.startswith("DEGENERATE_FIT") for w in o.fit.warnings)
    ]
    if degenerate:
        warnings.append(
            f"{len(degenerate)} candidate(s) collapsed onto a simpler model at their optimum "
            "(a parameter at its bound with no effect on the curve) and rank below every "
            "other row"
        )
    others = sum(1 for o in ordered if o.outcome in ("not_converged", "failed"))
    if others:
        warnings.append(f"{others} candidate(s) did not converge or failed and are not ranked")
    screened = sum(1 for o in ordered if o.stage == "screen" and o.outcome != "skipped")
    if screened:
        warnings.append(
            f"{screened} candidate(s) were fitted from 2 starts only: they were too far behind the "
            "best (AICc more than 10, or no convergence and an SSE over three times the best) to "
            "reach the top; turn off staged_starts to give every candidate the full multistart"
        )
    if not any(o.outcome in ("converged", "exact_fit") for o in ordered) and not cancelled:
        warnings.append("no candidate converged")
    warnings.extend(curve_notes)
    return tuple(warnings)
