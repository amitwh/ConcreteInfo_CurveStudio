"""The model discernibility report: can these data tell the leading models apart?

Auto-fit ranks models by information criteria, and "equally supported but predicting
differently" is exactly what a person needs to see. :func:`discernibility` reads a ranking
and the data it was made on, takes every model within 2 AICc of the best (the shipped
:data:`SIMPLE_DELTA_AICC`), refits each one deterministically through the fit-tool kit, and
compares the fitted curves on the data's own x range:

* two models are **predictively indistinguishable** when their curves differ by no more
  than the simpler model's prediction-interval half-width at every grid point;
* models are union-found into predictive groups;
* for two models in different groups, the report gives the contiguous x ranges where the
  difference exceeds the interval and the number of extra points a power approximation
  says would separate them;
* profile likelihood names a parameter the data cannot pin down, where the core supports
  it (a non-polynomial model with covariance on at most 2,000 points and at most 8
  parameters).

The report never invents a model or a number. A row it cannot refit is listed with the
reason; a number it cannot compute is an empty cell with a warning beside it. Every
function here is pure: it fits only the rows it is asked about, reads nothing else, and
touches no file, network or global state.

The approximations are named where they are used. The 2 AICc cut-off is a convention, not
a test. The extra-points figure is a two-sample power approximation for one extra parameter
on the grid, not a guarantee; it assumes the simpler model's residual standard deviation is
the noise and that the curve difference is the effect to detect.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import stats as scipy_stats

from curvestudio.data import XY
from curvestudio.discovery import AutoFitResult, RankedFit, candidate_ref, resolve_spec
from curvestudio.discovery.types import CandidateRef
from curvestudio.discovery.types import CustomRef as CandidateCustomRef
from curvestudio.discovery.types import PolynomialRef as CandidatePolynomialRef
from curvestudio.discovery.types import RegisteredRef as CandidateRegisteredRef
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit_polynomial
from curvestudio.statistics import FitView, build_view, profile_likelihood
from curvestudio.statistics.errors import StatisticsError
from curvestudio.tools._checks_selection import charged, simplest
from curvestudio.tools._errors import ToolError
from curvestudio.tools._fit_tools import (
    CancelledFitTool,
    FitTarget,
    cancel_checker,
    curve_of,
    evaluation_grid,
    prediction_half_width,
    resolve_target,
)
from curvestudio.tools._thresholds import SIMPLE_DELTA_AICC
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import joined, num
from curvestudio.tools.stats._base import (
    Counts,
    ToolResult,
    assumption,
    check_level,
    list_excluded,
)
from curvestudio.tools.stats._display import Cell, MultiLinePlot, XYSeries
from curvestudio.tools.stats._tables import table
from curvestudio.workflow import CustomRef, FitSpec, PolynomialRef, RegisteredRef, WorkflowError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

PROFILE_MAX_POINTS = 2_000
"""The most data points a profile-likelihood interval is computed on."""

PROFILE_MAX_PARAMETERS = 8
"""The most parameters a model may have for its profile intervals to be computed."""

MAX_PLOT_SERIES = 6
"""The most model curves the plot draws; a longer leading group says how many were left out."""

SEPARATION_ALPHA = 0.05
"""The significance level of the power approximation behind the extra-points figure."""

SEPARATION_POWER = 0.8
"""The target power of that approximation."""

_NOT_ASSESSED_CAP = "Not assessed: the model cap was reached"
_NOT_ASSESSED_CANCELLED = "Not assessed: the report was cancelled"

_MODEL_METHODS = (
    "Models are compared by their AICc distance, with a difference of 2 or less taken as no "
    "material difference. The leading group's fitted curves are compared on a "
    "{points}-point grid over the data's own x range: two models are indistinguishable when "
    "their curves differ by no more than the simpler model's {level:g} prediction-interval "
    "half-width at every grid point. A profile-likelihood interval is computed for each "
    "non-polynomial model with at most 2,000 points and at most 8 parameters, and a side "
    "that does not close names the parameter as not identifiable. The number of extra "
    "points that would separate two models is a power approximation for one extra "
    "parameter, n = (z_(1-alpha/2) + z_power)^2 sigma^2 / D^2, with alpha = 0.05, power = "
    "0.8, sigma the simpler model's residual standard deviation and D the root-mean-square "
    "curve difference on the grid."
)


class DiscernibilityRequest(BaseModel):
    """What the report is asked for.

    ``level`` is the confidence level of the prediction intervals and the profile intervals.
    ``grid_points`` is how many x values the curves are compared on. ``max_models`` caps
    how many models of the leading group are refitted and compared; the rest are listed
    and counted. ``profile`` turns the profile-likelihood check on or off.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: float = 0.95
    grid_points: int = 200
    max_models: int = 12
    profile: bool = True


class _Assessment(NamedTuple):
    """One leading-group row and what the report could make of it."""

    row: RankedFit
    delta: float
    target: FitTarget | None
    curve: FloatArray | None
    half_width: FloatArray | None
    residual_sd: float | None
    status: str


class _Pair(NamedTuple):
    """Two assessed models on the grid, and whether the interval can tell them apart."""

    first: int
    second: int
    difference: FloatArray
    half_width: FloatArray
    indistinguishable: bool


def _label(row: RankedFit) -> str:
    """A model as every table and sentence names it: its name and its id."""
    return f"{row.name} ({row.model_id})"


def _reason(error: BaseException) -> str:
    """The message of a core error, without its code."""
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def _delta_of(row: RankedFit, best_aicc: float) -> float | None:
    """The row's AICc distance from the best, stored when the ranking has it."""
    if row.delta_aicc is not None:
        return float(row.delta_aicc)
    aicc = row.metrics.aicc
    if aicc is None:
        return None
    return float(aicc - best_aicc)


def _same_curve_text(row: RankedFit) -> str | None:
    """What the ranking says about curves that coincide with this row."""
    if row.same_curve_as:
        return ", ".join(row.same_curve_as)
    if row.redundant_with is not None:
        return f"redundant with {row.redundant_with}"
    return None


def _stored_fit(ranking: AutoFitResult, row: RankedFit, data: XY) -> FitResult | None:
    """The row's own fit when the result kept it complete and matching ``data``.

    An API result stores each fit without its per-point vectors, and a large run's row may
    have been fitted on a subsample; either way the fit cannot be used on these data, and
    ``None`` sends the kit to a deterministic refit.
    """
    outcome = next((o for o in ranking.outcomes if o.candidate_id == row.model_id), None)
    if outcome is None:
        return None
    fit = outcome.fit
    if fit is None or fit.status is not FitStatus.CONVERGED:
        return None
    if len(fit.fitted) != data.x.size or len(fit.residuals) != data.x.size:
        return None
    if fit.provenance.dataset_fingerprint != data.fingerprint:
        return None
    return fit


def _workflow_spec(ref: CandidateRef) -> FitSpec:
    """The workflow's specification for a discovery candidate reference."""
    if isinstance(ref, CandidatePolynomialRef):
        return FitSpec(model=PolynomialRef(degree=ref.degree))
    if isinstance(ref, CandidateRegisteredRef):
        return FitSpec(model=RegisteredRef(id=ref.model_id))
    definition = ref.definition
    model = resolve_spec(ref)
    assert isinstance(ref, CandidateCustomRef) and model is not None
    return FitSpec(
        model=CustomRef(
            equation=definition.expression,
            parameters=[parameter.name for parameter in model.parameters],
            variable=definition.variables[0] if definition.variables else "x",
        )
    )


def _target_for(data: XY, ranking: AutoFitResult, row: RankedFit) -> tuple[FitTarget, bool]:
    """Resolve one ranked row into a target, refitting it when no complete fit is stored.

    Returns the target and whether the report refitted the data itself.

    A through-origin polynomial has no workflow specification, so a row without its stored
    fit is refitted through the core's linear fitter, which takes the flag, and handed to
    the kit as the result; :func:`~curvestudio.workflow.check_spec` accepts it for the
    polynomial of its degree.
    """
    ref = candidate_ref(ranking.config, row.model_id)
    spec = _workflow_spec(ref)
    stored = _stored_fit(ranking, row, data)
    refitted_here = False
    if stored is None and isinstance(ref, CandidatePolynomialRef) and ref.through_origin:
        stored = fit_polynomial(data, ref.degree, through_origin=True)
        refitted_here = True
    target = resolve_target(data, spec, stored)
    return target, refitted_here or target.refitted


def _curve_and_width(
    target: FitTarget, grid: FloatArray, level: float
) -> tuple[FloatArray, FloatArray, float | None]:
    """The target's curve, prediction half-width and residual SD on the grid.

    Raises:
        StatisticsError: the fit is not converged, does not match the data, or gives no
            covariance.
    """
    if target.result.status is not FitStatus.CONVERGED:
        raise StatisticsError(
            "FIT_NOT_CONVERGED",
            "the fit did not converge "
            f"({target.result.status.value}: {target.result.termination_reason})",
        )
    view: FitView = build_view(target.data, target.result, target.model)
    curve: FloatArray = np.asarray(curve_of(target, grid), dtype=np.float64)
    half_width: FloatArray = np.asarray(prediction_half_width(view, grid, level), dtype=np.float64)
    if not bool(np.all(np.isfinite(curve))):
        raise StatisticsError(
            "INVALID_X", "the fitted curve is not finite at every point of the grid"
        )
    if not bool(np.all(np.isfinite(half_width))):
        raise StatisticsError(
            "NO_COVARIANCE", "the prediction interval is not finite at every point of the grid"
        )
    residual_sd = None if view.s2 is None else math.sqrt(max(0.0, view.s2))
    return curve, half_width, residual_sd


def _profile_config(target: FitTarget, level: float) -> FitConfig:
    """The fit configuration the profile intervals must respect: the bounds that were fitted."""
    bounds = {
        name: (
            -math.inf if lower is None else lower,
            math.inf if upper is None else upper,
        )
        for name, (lower, upper) in target.bounds.items()
    }
    return FitConfig(
        method=target.fit_spec.method,
        loss=target.fit_spec.loss,
        bounds=bounds or None,
        seed=target.fit_spec.seed if target.fit_spec.seed is not None else 0,
        confidence_level=level,
    )


def _extra_points(curve_difference: float, residual_sd: float | None) -> int | None:
    """The extra points a power approximation needs to separate two curves, or ``None``.

    ``n = ceil((z_{1-alpha/2} + z_power)^2 * sigma^2 / D^2)`` with ``D`` the root-mean-square
    curve difference and ``sigma`` the simpler model's residual standard deviation.
    """
    if residual_sd is None or not math.isfinite(residual_sd) or residual_sd <= 0.0:
        return None
    if not math.isfinite(curve_difference) or curve_difference <= 0.0:
        return None
    quantile = float(
        scipy_stats.norm.ppf(1.0 - SEPARATION_ALPHA / 2.0) + scipy_stats.norm.ppf(SEPARATION_POWER)
    )
    needed = (quantile * residual_sd / curve_difference) ** 2
    return max(1, math.ceil(needed))


def _span_text(grid: FloatArray, start: int, end: int) -> str:
    """One contiguous x span, by its endpoints."""
    if start == end:
        return num(float(grid[start]))
    return f"{num(float(grid[start]))} to {num(float(grid[end]))}"


def _ranges_text(grid: FloatArray, mask: BoolArray) -> str:
    """The contiguous x ranges ``mask`` marks, each by its endpoints."""
    spans: list[str] = []
    start: int | None = None
    for index, flagged in enumerate(mask):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            spans.append(_span_text(grid, start, index - 1))
            start = None
    if start is not None:
        spans.append(_span_text(grid, start, mask.size - 1))
    return "; ".join(spans) if spans else "none"


def _simpler_key(row: RankedFit) -> tuple[int, bool, float, str]:
    """How a pair names its simpler model: fewer parameters, then AICc, then id."""
    aicc = row.metrics.aicc
    return (charged(row), aicc is None, 0.0 if aicc is None else aicc, row.model_id)


def _defensible(
    ranking: AutoFitResult, leading: list[tuple[RankedFit, float]]
) -> tuple[RankedFit | None, float | None, str | None]:
    """The leading group's simplest model that can be defended, with its AICc distance."""
    if not leading:
        return None, None, "no model lies within the AICc difference of the best"
    best_id = min(leading, key=lambda item: item[1])[0].model_id
    chosen, reason = simplest(ranking, best_id)
    leading_ids = {row.model_id for row, _delta in leading}
    if chosen is not None and chosen.model_id in leading_ids:
        delta = _delta_of(chosen, min(_delta for _row, _delta in leading))
        return chosen, delta, None
    usable = [
        (row, delta) for row, delta in leading if not row.degenerate and row.redundant_with is None
    ]
    if not usable:
        return (
            None,
            None,
            reason
            or (
                f"every model within {num(SIMPLE_DELTA_AICC)} AICc of the best is degenerate or "
                "redundant"
            ),
        )
    row, delta = min(
        usable,
        key=lambda item: (
            charged(item[0]),
            item[0].metrics.cv_rmse is None,
            item[0].metrics.cv_rmse or 0.0,
            item[0].rank,
        ),
    )
    return row, delta, None


def _assumptions(
    data: XY, grid: FloatArray, leading_count: int, ranked_count: int, refitted: int
) -> tuple[AuditCheck, ...]:
    """The report's three stated conventions."""
    low, high = float(grid[0]), float(grid[-1])
    refit_evidence = (
        "No row needed refitting: every assessed model came from the fit the result stored."
        if refitted == 0
        else f"{refitted} row(s) were refitted from the data with a single deterministic start."
    )
    return (
        assumption(
            "leading_group",
            f"The leading group is every model within {num(SIMPLE_DELTA_AICC)} AICc of the best",
            Verdict.PASS,
            f"{leading_count} of the {ranked_count} ranked models lie within "
            f"{num(SIMPLE_DELTA_AICC)} AICc of the best.",
            statistic=f"delta AICc <= {num(SIMPLE_DELTA_AICC)}",
            threshold=(
                "a difference of 2 AICc is the conventional cut-off for no material difference"
            ),
        ),
        assumption(
            "inside_data_range",
            "Predictions were compared only inside the data's own x range",
            Verdict.NOT_APPLICABLE,
            f"The grid runs from {num(low)} to {num(high)}, the range of "
            f"{data.x_name}; predictions outside it were not assessed.",
            threshold="the data's own x range",
        ),
        assumption(
            "deterministic_refits",
            "Every refit is deterministic",
            Verdict.PASS,
            refit_evidence,
            statistic=None,
            threshold="the same data and specification give the same refit",
        ),
    )


def discernibility(
    data: XY,
    ranking: AutoFitResult,
    *,
    request: DiscernibilityRequest = DiscernibilityRequest(),  # noqa: B008
    should_cancel: Callable[[], bool] | None = None,
) -> ToolResult:
    """Report whether these data can tell the leading group's models apart.

    Args:
        data: the data the ranking was made on, every row.
        ranking: the Auto-fit result to read.
        request: the level, the grid and the caps.
        should_cancel: a token read between models; when it turns true the report returns
            the part it finished with a ``cancelled`` warning.

    Returns:
        One :class:`~curvestudio.tools.stats.ToolResult`.

    Raises:
        ToolError: ``INVALID_LEVEL`` (the level is not in (0, 1)); ``INVALID_OPTION``
            (fewer than 2 grid points or fewer than 1 model); ``NO_DATA`` (the ranking holds
            no ranked model or no AICc to compare by).
    """
    check_level(request.level)
    if isinstance(request.grid_points, bool) or request.grid_points < 2:
        raise ToolError(
            "INVALID_OPTION",
            f"the grid needs at least 2 points, not {request.grid_points}",
        )
    if isinstance(request.max_models, bool) or request.max_models < 1:
        raise ToolError(
            "INVALID_OPTION", f"max_models must be at least 1, not {request.max_models}"
        )
    ranked = ranking.ranked
    if not ranked:
        raise ToolError("NO_DATA", "the Auto-fit ranking holds no converged model to compare")
    aiccs = [row.metrics.aicc for row in ranked if row.metrics.aicc is not None]
    if not aiccs:
        raise ToolError(
            "NO_DATA",
            "the Auto-fit ranking holds no AICc, so its models cannot be compared",
        )
    best_aicc = min(aiccs)

    leading: list[tuple[RankedFit, float]] = []
    for row in ranked:
        delta = _delta_of(row, best_aicc)
        if delta is not None and delta <= SIMPLE_DELTA_AICC:
            leading.append((row, delta))

    grid = evaluation_grid(data, points=request.grid_points)
    check = cancel_checker(should_cancel)
    warnings: list[str] = []
    assessments: list[_Assessment] = []
    not_identifiable: list[tuple[str, str, str]] = []
    cancelled = False
    refitted = 0

    for index, (row, delta) in enumerate(leading):
        if index >= request.max_models:
            assessments.append(_Assessment(row, delta, None, None, None, None, _NOT_ASSESSED_CAP))
            continue
        try:
            check()
        except CancelledFitTool:
            cancelled = True
            for later_row, later_delta in leading[index:]:
                assessments.append(
                    _Assessment(
                        later_row, later_delta, None, None, None, None, _NOT_ASSESSED_CANCELLED
                    )
                )
            break
        try:
            target, refitted_here = _target_for(data, ranking, row)
            curve, half_width, residual_sd = _curve_and_width(target, grid, request.level)
        except (WorkflowError, KeyError, ValueError, StatisticsError) as error:
            assessments.append(
                _Assessment(
                    row,
                    delta,
                    None,
                    None,
                    None,
                    None,
                    f"Not assessed: {_reason(error)}",
                )
            )
            warnings.append(f"{_label(row)} could not be assessed: {_reason(error)}")
            continue
        if refitted_here:
            refitted += 1
        status = "Assessed (refitted)" if refitted_here else "Assessed (the stored fit)"
        assessment = _Assessment(row, delta, target, curve, half_width, residual_sd, status)
        assessments.append(assessment)
        _profile(assessment, data, request, not_identifiable, warnings)

    left_out = len(leading) - request.max_models
    if left_out > 0:
        warnings.append(
            f"{left_out} model(s) in the leading group were not assessed: the report takes "
            f"at most {request.max_models}."
        )
    if len(leading) < 2:
        warnings.append(
            f"No other model lies within {num(SIMPLE_DELTA_AICC)} AICc of the best, so the "
            "leading group has no peer to compare."
        )
    if cancelled:
        done = sum(1 for assessment in assessments if assessment.target is not None)
        warnings.append(
            f"The report was cancelled after {done} of the leading group's models were "
            "assessed; the rest were not compared."
        )

    assessable = [
        (index, assessment)
        for index, assessment in enumerate(assessments)
        if assessment.curve is not None and assessment.half_width is not None
    ]
    if not assessable:
        warnings.append("No model of the leading group could be assessed, so no pair was compared.")
    elif len(assessable) < 2:
        warnings.append(
            "Only one model of the leading group could be assessed, so no pair was compared."
        )

    parent = list(range(len(assessable)))

    def root(here: int) -> int:
        while parent[here] != here:
            parent[here] = parent[parent[here]]
            here = parent[here]
        return here

    pairs: list[_Pair] = []
    for first in range(len(assessable)):
        for second in range(first + 1, len(assessable)):
            _index_a, model_a = assessable[first]
            _index_b, model_b = assessable[second]
            assert model_a.curve is not None and model_a.half_width is not None
            assert model_b.curve is not None and model_b.half_width is not None
            simpler = model_a if _simpler_key(model_a.row) <= _simpler_key(model_b.row) else model_b
            assert simpler.half_width is not None
            difference = np.abs(model_a.curve - model_b.curve)
            same = bool(np.all(difference <= simpler.half_width))
            pairs.append(_Pair(first, second, difference, simpler.half_width, same))
            if same:
                one, other = root(first), root(second)
                if one != other:
                    parent[other] = one

    groups_by_root: dict[int, list[int]] = {}
    group_order: list[int] = []
    for position in range(len(assessable)):
        found = root(position)
        if found not in groups_by_root:
            groups_by_root[found] = []
            group_order.append(found)
        groups_by_root[found].append(position)
    groups = [groups_by_root[found] for found in group_order]
    group_of: dict[int, int] = {}
    for number, members in enumerate(groups, start=1):
        for member in members:
            group_of[member] = number

    model_rows: list[tuple[Cell, ...]] = []
    for assessment in assessments:
        row = assessment.row
        model_rows.append(
            (
                row.rank,
                _label(row),
                row.metrics.aicc,
                assessment.delta,
                charged(row),
                assessment.status,
                _same_curve_text(row),
            )
        )

    predictive_rows: list[tuple[Cell, ...]] = []
    for number, members in enumerate(groups, start=1):
        labels = joined([_label(assessable[position][1].row) for position in members])
        intra = [pair for pair in pairs if pair.first in members and pair.second in members]
        largest: float | None = None
        interval: float | None = None
        if intra:
            at_max = max(intra, key=lambda pair: float(pair.difference.max()))
            largest = float(at_max.difference.max())
            interval = float(at_max.half_width[int(np.argmax(at_max.difference))])
        predictive_rows.append((number, labels, largest, interval))

    separation_rows: list[tuple[Cell, ...]] = []
    separated = 0
    for pair in pairs:
        if pair.indistinguishable:
            continue
        separated += 1
        simpler = (
            assessable[pair.first][1]
            if _simpler_key(assessable[pair.first][1].row)
            <= _simpler_key(assessable[pair.second][1].row)
            else assessable[pair.second][1]
        )
        rms = float(np.sqrt(np.mean(np.square(pair.difference))))
        extra = _extra_points(rms, simpler.residual_sd)
        basis = (
            "Power approximation for one extra parameter, alpha = "
            f"{num(SEPARATION_ALPHA)} and power = {num(SEPARATION_POWER)}; RMS curve "
            f"difference {num(rms)}, residual SD "
            f"{'unknown' if simpler.residual_sd is None else num(simpler.residual_sd)}"
        )
        separation_rows.append(
            (
                f"{group_of[pair.first]} vs {group_of[pair.second]}",
                _ranges_text(grid, pair.difference > pair.half_width),
                extra,
                basis,
            )
        )

    defensible, defensible_delta, defensible_reason = _defensible(ranking, leading)
    defensible_rows: list[tuple[Cell, ...]] = []
    if defensible is None:
        warnings.append(
            "The simplest model that can be defended could not be named: "
            f"{defensible_reason or 'no usable row in the leading group'}."
        )
    else:
        defensible_rows.append(
            (
                _label(defensible),
                charged(defensible),
                defensible.metrics.aicc,
                defensible_delta,
            )
        )

    for label, parameter, reason in not_identifiable:
        warnings.append(f"{label}: the profile for {parameter} did not close: {reason}.")

    plot_series: list[XYSeries] = []
    for _index, assessment in assessable[:MAX_PLOT_SERIES]:
        assert assessment.curve is not None
        plot_series.append(
            XYSeries(
                label=assessment.row.name,
                x=tuple(float(value) for value in grid),
                y=tuple(float(value) for value in assessment.curve),
            )
        )
    plot_note = None
    if len(assessable) > MAX_PLOT_SERIES:
        plot_note = (
            f"{MAX_PLOT_SERIES} of {len(assessable)} assessed models shown; the rest are in "
            "the tables."
        )

    interpretation = _interpretation(
        assessable,
        groups,
        separated,
        separation_rows,
        defensible,
        cancelled,
        len(leading),
    )

    n_offered = int(data.x.size)
    n_used = int(ranking.n_active)
    n_excluded = max(0, n_offered - n_used)
    excluded: list[ExcludedRow] = []
    if data.weights is not None:
        for at in np.flatnonzero(np.asarray(data.weights) <= 0.0):
            excluded.append(
                ExcludedRow(
                    index=int(at),
                    reason=("zero weight: the fit and the ranking used the active points only"),
                )
            )
    listed, unlisted = list_excluded(excluded)

    return ToolResult(
        tool="discernibility",
        title="Model discernibility",
        level=request.level,
        assumptions=_assumptions(data, grid, len(leading), len(ranked), refitted),
        warnings=tuple(warnings),
        interpretation=interpretation,
        methods_fragment=_MODEL_METHODS.format(points=request.grid_points, level=request.level),
        counts=Counts(n_offered=n_offered, n_used=n_used, n_excluded=n_excluded),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Models",
                (
                    ("rank", "Rank", "int"),
                    ("model", "Model", "text"),
                    ("aicc", "AICc", "number"),
                    ("delta_aicc", "Delta AICc", "number"),
                    ("parameters", "Parameters", "int"),
                    ("status", "Status", "text"),
                    ("same_curve_as", "Same curve as", "text"),
                ),
                model_rows,
                note=(
                    f"Every model within {num(SIMPLE_DELTA_AICC)} AICc of the best, in ranked "
                    "order. Parameters counts the fitted parameters plus any free-frequency "
                    "shape parameters the ranking charges."
                ),
            ),
            table(
                "Predictive groups",
                (
                    ("group", "Group", "int"),
                    ("models", "Models", "text"),
                    ("largest_difference", "Largest difference", "number"),
                    ("prediction_interval", "Prediction interval", "number"),
                ),
                predictive_rows,
                note=(
                    "Models in one group differ by no more than the simpler model's "
                    "prediction-interval half-width at every grid point. Largest difference "
                    "is the largest absolute curve difference among the group's pairs, and "
                    "Prediction interval is that pair's half-width where the difference is "
                    "largest."
                ),
            ),
            table(
                "Parameters not identifiable",
                (
                    ("model", "Model", "text"),
                    ("parameter", "Parameter", "text"),
                    ("reason", "Reason", "text"),
                ),
                not_identifiable,
                note=(
                    "A parameter whose profile-likelihood interval does not close on one or "
                    "both sides: the data do not pin it down at this level."
                ),
            ),
            table(
                "What would separate the groups",
                (
                    ("groups", "Groups", "text"),
                    ("x_range", "x range", "text"),
                    ("extra_points", "Extra points", "int"),
                    ("basis", "Basis", "text"),
                ),
                separation_rows,
                note=(
                    "Each row is one pair of models in different predictive groups. The x "
                    "range is where their curves differ by more than the simpler model's "
                    "prediction-interval half-width; Extra points is how many more "
                    "observations the stated power approximation says would separate them."
                ),
            ),
            table(
                "The simplest model that can be defended",
                (
                    ("model", "Model", "text"),
                    ("parameters", "Parameters", "int"),
                    ("aicc", "AICc", "number"),
                    ("delta_aicc", "Delta AICc", "number"),
                ),
                defensible_rows,
                note=(
                    "The leading group's row with the fewest parameters that is neither "
                    "degenerate nor redundant with a simpler model."
                ),
            ),
        ),
        plots=(
            MultiLinePlot(
                title="Models in the leading group",
                x_label=data.x_name,
                y_label=data.y_name,
                series=tuple(plot_series),
                note=plot_note,
            ),
        ),
    )


def _profile(
    assessment: _Assessment,
    data: XY,
    request: DiscernibilityRequest,
    not_identifiable: list[tuple[str, str, str]],
    warnings: list[str],
) -> None:
    """Profile each parameter of one assessed model where the core supports it."""
    if not request.profile:
        return
    target = assessment.target
    if target is None:
        return
    label = _label(assessment.row)
    if target.model is None:
        warnings.append(
            f"{label}: profile likelihood is not supported for polynomial fits, whose "
            "Wald interval is already exact."
        )
        return
    if data.x.size > PROFILE_MAX_POINTS:
        warnings.append(
            f"{label}: profile likelihood was not run on {data.x.size:,} points (the "
            f"report profiles at most {PROFILE_MAX_POINTS:,})."
        )
        return
    if len(target.model.parameters) > PROFILE_MAX_PARAMETERS:
        warnings.append(
            f"{label}: profile likelihood was not run for {len(target.model.parameters)} "
            f"parameters (the report profiles at most {PROFILE_MAX_PARAMETERS})."
        )
        return
    try:
        result = profile_likelihood(
            data,
            target.model,
            target.result,
            _profile_config(target, request.level),
            request.level,
        )
    except (StatisticsError, ValueError) as error:
        warnings.append(f"{label}: profile likelihood is not supported here: {_reason(error)}")
        return
    for interval in result.intervals:
        sides = [
            side
            for side, status in (
                ("lower", interval.low_status),
                ("upper", interval.high_status),
            )
            if status == "not_found"
        ]
        if not sides:
            continue
        where = " and ".join(sides)
        reason = (
            f"the profile never crossed the threshold on the {where} side within 50 standard errors"
        )
        not_identifiable.append((label, interval.name, reason))


def _interpretation(
    assessable: list[tuple[int, _Assessment]],
    groups: list[list[int]],
    separated: int,
    separation_rows: list[tuple[Cell, ...]],
    defensible: RankedFit | None,
    cancelled: bool,
    leading_count: int,
) -> str:
    """One to three plain sentences naming the groups, the simplest model and the gap."""
    if not assessable:
        return (
            f"The leading group holds {leading_count} model(s), and none of them could be "
            "assessed, so the data cannot be compared with them."
        )
    names = [
        f"group {number}: {joined([_label(assessable[position][1].row) for position in members])}"
        for number, members in enumerate(groups, start=1)
    ]
    if len(assessable) > 1:
        opening = (
            f"The {len(assessable)} assessed models form {len(groups)} predictive "
            f"{'group' if len(groups) == 1 else 'groups'} — {'; '.join(names)}."
        )
    else:
        opening = (
            f"Only {_label(assessable[0][1].row)} could be assessed, so there is no pair to "
            "compare."
        )
    if defensible is None:
        simplest_sentence = "No model in the leading group could be named as the simplest."
    else:
        simplest_sentence = (
            f"The simplest model that can be defended is {_label(defensible)} with "
            f"{charged(defensible)} parameters."
        )
    if separated:
        first = next((cells for cells in separation_rows if isinstance(cells[2], int)), None)
        if first is None:
            separation = (
                f"{separated} pair(s) of models predict differently by more than the simpler "
                "model's prediction interval; the table gives where."
            )
        else:
            more = "" if separated == 1 else " The other pairs are in the table."
            separation = (
                f"{separated} pair(s) of models predict differently by more than the simpler "
                f"model's prediction interval; the first ({first[0]}) differs over x = "
                f"{first[1]}, and about {first[2]} more points at this spread would separate "
                f"them (a power approximation for one extra parameter).{more}"
            )
    elif len(assessable) > 1:
        separation = (
            "No pair of assessed models predicts differently by more than the prediction "
            "interval anywhere in the data's x range, so these data cannot tell them apart."
        )
    else:
        separation = ""
    closing = " The report was cancelled, so the comparison is partial." if cancelled else ""
    return (
        " ".join(sentence for sentence in (opening, simplest_sentence, separation) if sentence)
        + closing
    )
