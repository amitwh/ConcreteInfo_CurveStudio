"""The next-measurement advisor: where to measure next, and what it buys.

Given a fitted model, the advisor reads the fit's information matrix and asks where a few
further observations would sharpen the estimates most. It knows nothing about the new data:
it assumes the model form holds and the fitted covariance is the true one, and it says so in
its assumptions, its warnings and its methods text.

The information matrix is ``M = J' W J`` from the shipped
:class:`~curvestudio.statistics.FitView` — the model's Jacobian at the fitted parameters over
the active points, weighted by the fit's weights (all ones for an unweighted fit) — and the
covariance of the estimates is ``sigma^2 pinv(M)``, with ``sigma^2`` the fit's residual
variance (SSE / df). Add a new observation at ``x`` with Jacobian row ``j`` and the matrix
becomes ``M + j j'``; ``replicates`` copies of it add ``replicates * j j'``.

Candidates are ``candidates`` evenly spaced values over the requested range: the data's own
range unless the request extends it, and the result says when it does. The exchange is greedy
and deterministic: at each step every candidate still eligible is tried with the rows chosen
so far, and the one is taken that, with its ``replicates`` rows added, maximises the
determinant of the augmented information matrix in logs (D-optimality) or minimises the
covariance diagonal entry of the named parameter. A location carries at most ``replicates``
rows, all added in its one step, so there are ``points`` steps and one recommended location
per step.

No random number is drawn and no data are invented, so the same fit and request always give
byte-identical JSON.
"""

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from curvestudio.data import XY
from curvestudio.fitting import FitResult
from curvestudio.statistics import FitView, build_view
from curvestudio.tools._errors import ToolError
from curvestudio.tools._fit_tools import (
    CancelledFitTool,
    cancel_checker,
    evaluation_grid,
    prediction_half_width,
    resolve_target,
)
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    ToolResult,
    assumption,
    check_level,
    list_excluded,
)
from curvestudio.tools.stats._display import FloatArray, MultiLinePlot, XYSeries
from curvestudio.tools.stats._tables import table
from curvestudio.workflow import CustomRef, FitSpec, PolynomialRef, RegisteredRef

MIN_POINTS = 1
"""The fewest measurements one request may recommend."""

MAX_POINTS = 20
"""The most measurements one request may recommend; the exchange is quadratic in it."""

MIN_CANDIDATES = 10
"""The fewest candidate locations the exchange chooses from."""

MAX_CANDIDATES = 1_000
"""The most candidate locations one step may be evaluated over."""

MIN_REPLICATES = 1
"""The fewest rows one recommended location carries."""

MAX_REPLICATES = 20
"""The most rows one recommended location carries."""

FEW_DATA_POINTS = 10
"""Below this many active points the extension is said to rest on few data."""

END_SHARE = 0.02
"""A point within this share of the range from an end counts as lying at that end."""


class AdvisorRequest(BaseModel):
    """What to ask the advisor: how many points, over which range, for which parameter.

    ``points`` is 1 to 20 and ``candidates`` 10 to 1,000 (the exchange tries every candidate
    at every step); ``replicates`` is 1 to 20 rows at one location. ``low`` and ``high``
    replace the data's own range on the sides they are given. ``objective`` is ``d_optimal``
    (maximise the determinant of the information matrix) or ``parameter`` (minimise the named
    ``parameter``'s variance, which is then required); ``parameter`` is ignored under
    ``d_optimal``. ``level`` is the confidence level of the prediction band. The values are
    checked by :func:`advise`, which raises ``INVALID_OPTION`` for anything out of range.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    points: int = 3
    candidates: int = 400
    low: float | None = None
    high: float | None = None
    replicates: int = 1
    objective: Literal["d_optimal", "parameter"] = "d_optimal"
    parameter: str | None = None
    level: float = 0.95


def _whole(name: str, value: int, low: int, high: int) -> None:
    """Refuse a count that is not a whole number in ``[low, high]``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError("INVALID_OPTION", f"{name} must be a whole number, not {value!r}")
    if not low <= value <= high:
        raise ToolError("INVALID_OPTION", f"{name} must be from {low} to {high}, not {value}")


def _check_request(request: AdvisorRequest) -> None:
    """Refuse a request of an unknown type, an out-of-range value or a bad level."""
    if not isinstance(request, AdvisorRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"advise takes an AdvisorRequest, not a request of {type(request).__name__}",
        )
    check_level(request.level)
    _whole("points", request.points, MIN_POINTS, MAX_POINTS)
    _whole("candidates", request.candidates, MIN_CANDIDATES, MAX_CANDIDATES)
    _whole("replicates", request.replicates, MIN_REPLICATES, MAX_REPLICATES)


def _check_range(low: float, high: float) -> None:
    """Refuse a range that is not finite or does not rise."""
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ToolError(
            "INVALID_OPTION", f"the candidate range must be finite, not {low!r} to {high!r}"
        )
    if low >= high:
        raise ToolError(
            "INVALID_OPTION", f"the candidate range needs low below high, not {low!r} and {high!r}"
        )


def _check_parameter(request: AdvisorRequest, names: tuple[str, ...]) -> int | None:
    """The index of the parameter the ``parameter`` objective minimises, or ``None``.

    The name is required under that objective (``INVALID_OPTION`` when missing) and must be
    one of the model's parameters (``UNKNOWN_COLUMN`` when not). Under ``d_optimal`` the
    field plays no part and is not read.
    """
    if request.objective != "parameter":
        return None
    if request.parameter is None:
        raise ToolError(
            "INVALID_OPTION",
            "the parameter objective needs the name of the parameter to narrow, e.g. "
            f"parameter='{names[0]}'",
        )
    if request.parameter not in names:
        raise ToolError(
            "UNKNOWN_COLUMN",
            f"parameter names {request.parameter!r}, which is not a parameter of the model "
            f"({', '.join(names)})",
        )
    return names.index(request.parameter)


def _information_matrix(view: FitView, weighted: bool) -> FloatArray:
    """``M = J' W J``: the weighted Jacobian, squared up, on the active points."""
    jacobian = view.weighted_jacobian if weighted else view.jacobian_at(view.x)
    return np.asarray(jacobian.T @ jacobian, dtype=np.float64)


def _covariance(information: FloatArray, sigma2: float) -> FloatArray:
    """The estimates' covariance ``sigma^2 pinv(M)``."""
    return np.asarray(sigma2 * np.linalg.pinv(information, hermitian=True), dtype=np.float64)


def _standard_errors(covariance: FloatArray) -> tuple[float, ...]:
    """The square root of the covariance diagonal, one per parameter."""
    return tuple(float(value) for value in np.sqrt(np.maximum(np.diag(covariance), 0.0)))


def _d_optimal_score(information: FloatArray) -> float:
    """``log det(M)``, or ``-inf`` when the matrix is not positive definite."""
    sign, log_determinant = np.linalg.slogdet(information)
    return float(log_determinant) if float(sign) > 0.0 else -math.inf


def _parameter_score(information: FloatArray, index: int) -> float:
    """Minus one covariance diagonal entry, so that larger is better."""
    variance = float(np.linalg.pinv(information, hermitian=True)[index, index])
    return -variance


def _narrowing(before: float, after: float) -> float | None:
    """How far an SE fell, in percent of where it started; ``None`` when it started at 0."""
    if before <= 0.0:
        return None
    return 100.0 * (before - after) / before


def _model_label(spec: FitSpec) -> str:
    """The model in a sentence: the polynomial's degree, the registered id or the equation."""
    ref = spec.model
    if isinstance(ref, PolynomialRef):
        return f"a degree {ref.degree} polynomial"
    if isinstance(ref, RegisteredRef):
        return ref.id
    assert isinstance(ref, CustomRef)  # the third and last reference kind
    return "the custom equation"


def _design_sentence(steps: list[tuple[int, float, int, tuple[float, ...]]]) -> str:
    """The recommended locations in words, with their replicate counts."""
    parts = []
    for _step, x, replicates, _se in steps:
        parts.append(f"x = {num(x)}" + (f" ({replicates} rows)" if replicates != 1 else ""))
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def advise(
    data: XY,
    spec: FitSpec,
    result: FitResult,
    *,
    request: AdvisorRequest = AdvisorRequest(),  # noqa: B008
    should_cancel: Callable[[], bool] | None = None,
) -> ToolResult:
    """Recommend the next measurement locations for a fitted model.

    The information matrix comes from the shipped :class:`~curvestudio.statistics.FitView`
    built from ``data`` and ``result``; the candidates are evenly spaced over the requested
    range, and each step adds the candidate that most sharpens the information matrix (or
    the named parameter's variance). Every standard error after a step is read from
    ``sigma^2 pinv(M)``, and the prediction band's narrowing is the shipped pointwise
    prediction interval on the fit's residual degrees of freedom.

    Args:
        data: the data the fit was computed from, every row. Zero-weight rows are listed as
            excluded and take no part.
        spec: the specification the fit ran with; :func:`~curvestudio.tools.resolve_target`
            checks that the result is the fit it describes.
        result: the fit whose Jacobian and residual variance the design is built from.
        request: the number of points and candidates, the range, the replicates, the
            objective and the level.
        should_cancel: an optional token checked between candidate evaluations and between
            steps. Once it turns true the result is built from the steps already taken and
            carries the warning ``cancelled after k of N points``.

    Returns:
        One :class:`~curvestudio.tools.stats.ToolResult`: the recommended points, the
        narrowing of every parameter's SE, the after-SEs and band half-widths, the
        assumptions, the band-narrowing plot, the warnings and the methods text.

    Raises:
        ToolError: ``INVALID_OPTION`` for a request outside the tool's range, a missing
            parameter under the parameter objective, a range that does not rise, or a model
            that is not finite at a candidate; ``UNKNOWN_COLUMN`` for a parameter name the
            model does not have.
        StatisticsError: from the view or the band when the fit gives no covariance or the
            data are not the fit's.
        WorkflowError: from the target resolution when the specification is not the fit's.
    """
    _check_request(request)
    target = resolve_target(data, spec, result)
    view = build_view(data, target.result, target.model)
    focus_index = _check_parameter(request, view.names)
    before_covariance = view.require_covariance()
    assert view.s2 is not None  # require_covariance ensured df > 0
    sigma2 = float(view.s2)
    weighted = data.weights is not None

    data_low, data_high = _data_range(data)
    low = data_low if request.low is None else float(request.low)
    high = data_high if request.high is None else float(request.high)
    _check_range(low, high)
    try:
        candidates = evaluation_grid(
            data, points=request.candidates, low=request.low, high=request.high
        )
    except ValueError as exc:
        raise ToolError("INVALID_OPTION", f"the candidate range cannot be used: {exc}") from exc

    candidate_rows = np.asarray(view.jacobian_at(candidates), dtype=np.float64)
    finite = np.all(np.isfinite(candidate_rows), axis=1)
    if not np.all(finite):
        first = float(candidates[int(np.flatnonzero(~finite)[0])])
        raise ToolError(
            "INVALID_OPTION",
            f"the model is not finite at x = {num(first)}; the candidate range must lie inside "
            "the model's domain",
        )

    information = _information_matrix(view, weighted)
    before_se = _standard_errors(before_covariance)
    check = cancel_checker(should_cancel)
    counts = np.zeros(candidates.size, dtype=np.intp)
    augmented = information.copy()
    steps: list[tuple[int, float, int, tuple[float, ...]]] = []
    cancelled = False
    exhausted = False
    for step in range(1, request.points + 1):
        try:
            check()
        except CancelledFitTool:
            cancelled = True
            break
        best = -1
        best_score = -math.inf
        for index in range(candidates.size):
            try:
                check()
            except CancelledFitTool:
                cancelled = True
                break
            if counts[index] > 0:
                continue
            row = candidate_rows[index]
            trial = augmented + request.replicates * np.outer(row, row)
            if focus_index is None:
                score = _d_optimal_score(trial)
            else:
                score = _parameter_score(trial, focus_index)
            if score > best_score:
                best = index
                best_score = score
        if cancelled:
            break
        if best < 0:
            exhausted = True
            break
        row = candidate_rows[best]
        augmented = augmented + request.replicates * np.outer(row, row)
        counts[best] = request.replicates
        after = _standard_errors(_covariance(augmented, sigma2))
        steps.append((step, float(candidates[best]), request.replicates, after))
    if steps:
        after_covariance = _covariance(augmented, sigma2)
        after_se = steps[-1][3]
    else:
        after_covariance = before_covariance
        after_se = before_se

    try:
        band_grid = evaluation_grid(data, low=request.low, high=request.high)
    except ValueError as exc:
        raise ToolError("INVALID_OPTION", f"the grid cannot be used: {exc}") from exc
    before_half = prediction_half_width(view, band_grid, request.level)
    after_view = replace(view, cov=after_covariance) if steps else view
    after_half = prediction_half_width(after_view, band_grid, request.level)
    before_band = float(np.mean(before_half))
    after_band = float(np.mean(after_half))

    extends = low < data_low or high > data_high
    warnings = _warnings(
        steps=steps,
        cancelled=cancelled,
        exhausted=exhausted,
        request=request,
        low=low,
        high=high,
        data_low=data_low,
        data_high=data_high,
        extends=extends,
        n_active=int(view.x.size),
    )
    focus = _focus(focus_index, view.names, before_se, after_se)
    excluded, excluded_unlisted = _excluded(data)
    return ToolResult(
        tool="advisor",
        title=f"Where to measure next for {_model_label(spec)}",
        level=request.level,
        estimates=(
            *(
                Estimate(label=f"SE of {name}", value=float(after_se[index]))
                for index, name in enumerate(view.names)
            ),
            Estimate(
                label="Prediction band half-width at the grid, mean (before)", value=before_band
            ),
            Estimate(
                label="Prediction band half-width at the grid, mean (after)", value=after_band
            ),
        ),
        assumptions=_assumptions(
            sigma2=sigma2,
            weighted=weighted,
            extends=extends,
            low=low,
            high=high,
            data_low=data_low,
            data_high=data_high,
        ),
        warnings=warnings,
        interpretation=_interpretation(
            steps=steps,
            cancelled=cancelled,
            request=request,
            focus=focus,
            before_se=before_se,
            after_se=after_se,
            names=view.names,
            before_band=before_band,
            after_band=after_band,
            focus_index=focus_index,
        ),
        methods_fragment=_methods(
            spec=spec,
            request=request,
            names=view.names,
            n_active=int(view.x.size),
            weighted=weighted,
            sigma2=sigma2,
            candidates=int(candidates.size),
            low=low,
            high=high,
            focus_index=focus_index,
        ),
        counts=_counts(data, view),
        excluded=excluded,
        excluded_unlisted=excluded_unlisted,
        tables=(
            table(
                "Recommended points",
                (
                    ("step", "Step", "int"),
                    ("x", "x", "number"),
                    ("replicates", "Replicates", "int"),
                    *(
                        (f"se_{index}", f"SE of {name}", "number")
                        for index, name in enumerate(view.names)
                    ),
                ),
                [(step, x, replicates, *se) for step, x, replicates, se in steps],
                note=(
                    "One row per step of the greedy exchange, with every parameter's standard "
                    "error read from sigma^2 pinv(M) after that step, sigma^2 being the fit's "
                    "residual variance and M the information matrix J' W J of the active points "
                    "augmented with the points chosen so far."
                ),
            ),
            table(
                "Narrowing",
                (
                    ("parameter", "Parameter", "text"),
                    ("before", "SE before", "number"),
                    ("after", "SE after", "number"),
                    ("change", "Change %", "number"),
                ),
                [
                    (
                        name,
                        before_se[index],
                        after_se[index],
                        _narrowing(before_se[index], after_se[index]),
                    )
                    for index, name in enumerate(view.names)
                ],
                note=(
                    "Before is the fit's own covariance; after is sigma^2 pinv(M) with every "
                    "recommended point added. A positive Change % means the SE fell by that "
                    "share of where it started; an empty cell means it started at zero."
                ),
            ),
        ),
        plots=(
            MultiLinePlot(
                title="Prediction band narrowing",
                x_label="x",
                y_label="Prediction half-width",
                series=(
                    XYSeries(
                        label="Before",
                        x=tuple(float(value) for value in band_grid),
                        y=tuple(float(value) for value in before_half),
                    ),
                    XYSeries(
                        label="After",
                        x=tuple(float(value) for value in band_grid),
                        y=tuple(float(value) for value in after_half),
                    ),
                ),
                note=(
                    "The shipped pointwise prediction interval's half-width over the candidate "
                    f"range at the {num(request.level * 100)} % level, on the fit's residual "
                    "degrees of freedom, before and after the recommended points."
                ),
            ),
        ),
    )


def _data_range(data: XY) -> tuple[float, float]:
    """The data's own x range, from every finite row."""
    xs = np.asarray(data.x, dtype=np.float64)
    finite = xs[np.isfinite(xs)]
    return float(finite.min()), float(finite.max())


def _counts(data: XY, view: FitView) -> Counts:
    """The rows offered, the active rows used, and the rest."""
    offered = int(data.x.size)
    used = int(view.x.size)
    return Counts(n_offered=offered, n_used=used, n_excluded=offered - used)


def _excluded(data: XY) -> tuple[tuple[ExcludedRow, ...], int]:
    """The zero-weight rows, listed with the reason, and how many were not listed."""
    if data.weights is None:
        return (), 0
    rows = tuple(
        ExcludedRow(index=index, reason="weight is zero")
        for index, weight in enumerate(data.weights)
        if weight == 0.0
    )
    return list_excluded(rows)


def _assumptions(
    *,
    sigma2: float,
    weighted: bool,
    extends: bool,
    low: float,
    high: float,
    data_low: float,
    data_high: float,
) -> tuple[AuditCheck, ...]:
    """The two standing assumptions, and the extrapolation caution when it applies."""
    weight_note = "the fit's weights" if weighted else "all ones, because the fit is unweighted"
    checks = [
        assumption(
            "covariance_as_truth",
            "The fitted covariance is taken as the true one",
            Verdict.PASS,
            f"The predicted standard errors are sigma^2 pinv(M), with sigma^2 = SSE/df = "
            f"{num(sigma2)} the fit's residual variance and M = J' W J carrying {weight_note}; "
            "the design is only as good as that covariance.",
        ),
        assumption(
            "model_as_given",
            "The model form is taken as correct",
            Verdict.NOT_APPLICABLE,
            "A design says where to measure if the model is right; it cannot test the model's "
            "form, so the model's adequacy has to be judged from the fit's own audit.",
        ),
    ]
    if extends:
        checks.append(
            assumption(
                "extrapolation",
                "The candidate range extends beyond the data",
                Verdict.CAUTION,
                f"The data run from {num(data_low)} to {num(data_high)} and the candidate range "
                f"is {num(low)} to {num(high)}, so the curve — and the predicted standard "
                "error — is extrapolated outside the data.",
                next_step=(
                    "Measure within the data's range unless the model is known to hold there."
                ),
            )
        )
    return tuple(checks)


def _warnings(
    *,
    steps: list[tuple[int, float, int, tuple[float, ...]]],
    cancelled: bool,
    exhausted: bool,
    request: AdvisorRequest,
    low: float,
    high: float,
    data_low: float,
    data_high: float,
    extends: bool,
    n_active: int,
) -> tuple[str, ...]:
    """Everything the request or the run does that a person should be told about."""
    warnings: list[str] = []
    if extends:
        warnings.append(
            f"The candidate range {num(low)} to {num(high)} extends beyond the data's range "
            f"{num(data_low)} to {num(data_high)}; the curve is extrapolated there, and so is "
            "the predicted standard error."
        )
    if cancelled:
        warnings.append(f"cancelled after {len(steps)} of {request.points} points")
    if exhausted:
        warnings.append(
            f"only {len(steps)} distinct candidate locations could be placed; the requested "
            f"{request.points} points were not available over this range"
        )
    span = high - low
    at_end = all(
        abs(x - low) <= END_SHARE * span or abs(x - high) <= END_SHARE * span
        for _step, x, _replicates, _se in steps
    )
    if at_end and steps and n_active < FEW_DATA_POINTS:
        warnings.append(
            f"Every recommended point lies at an end of the range, and the fit rests on only "
            f"{n_active} active points; with so few data the ends carry the most information, "
            "but a point inside the range would say more about the model's shape."
        )
    return tuple(warnings)


def _focus(
    focus_index: int | None,
    names: tuple[str, ...],
    before_se: tuple[float, ...],
    after_se: tuple[float, ...],
) -> int:
    """The parameter the interpretation reports: the named one, or the most narrowed."""
    if focus_index is not None:
        return focus_index
    reductions = [
        (before_se[index] - after_se[index]) / before_se[index] if before_se[index] > 0.0 else 0.0
        for index in range(len(names))
    ]
    return max(range(len(names)), key=lambda index: reductions[index])


def _interpretation(
    *,
    steps: list[tuple[int, float, int, tuple[float, ...]]],
    cancelled: bool,
    request: AdvisorRequest,
    focus: int,
    before_se: tuple[float, ...],
    after_se: tuple[float, ...],
    names: tuple[str, ...],
    before_band: float,
    after_band: float,
    focus_index: int | None,
) -> str:
    """Where to measure, what it buys, and what the design assumes."""
    if not steps:
        return (
            "The run stopped before the first point was chosen, so no measurement is "
            "recommended. A design assumes the model form and the fitted covariance are right; "
            "if either is wrong, its points need not be the best ones."
        )
    sentences = [f"Measure next at {_design_sentence(steps)}."]
    name = names[focus]
    before = before_se[focus]
    after = after_se[focus]
    change = _narrowing(before, after)
    if change is None:
        sentences.append(f"The standard error of {name!r} stays at zero.")
    else:
        lead = (
            "Under the parameter objective" if focus_index is not None else "With the most to gain"
        )
        sentences.append(
            f"{lead}, the standard error of {name!r} falls from {num(before)} to "
            f"{num(after)} ({num(change)} % narrower)."
        )
    sentences.append(
        f"Over the grid the mean prediction half-width falls from {num(before_band)} to "
        f"{num(after_band)}."
    )
    if cancelled:
        sentences.append(
            f"The run was cancelled after {len(steps)} of {request.points} points, so these "
            "are the steps it finished."
        )
    sentences.append(
        "The design assumes the model form and the fitted covariance are right; if either is "
        "wrong, these points need not be the best ones."
    )
    return " ".join(sentences)


def _methods(
    *,
    spec: FitSpec,
    request: AdvisorRequest,
    names: tuple[str, ...],
    n_active: int,
    weighted: bool,
    sigma2: float,
    candidates: int,
    low: float,
    high: float,
    focus_index: int | None,
) -> str:
    """The methods paragraph: objective, exchange, information matrix and sigma^2."""
    weight_note = "the fit's weights" if weighted else "all ones (an unweighted fit)"
    if focus_index is None:
        criterion = (
            "D-optimality: each step added the candidate maximising the determinant, in logs, "
            "of the information matrix augmented by its replicate Jacobian rows"
        )
    else:
        criterion = (
            f"minimising the variance of {names[focus_index]!r}, the corresponding diagonal "
            "entry of the pseudo-inverse of the information matrix augmented by its replicate "
            "Jacobian rows"
        )
    return (
        f"Next-measurement design. The information matrix was M = J' W J, with J the model's "
        f"Jacobian at the fitted parameters on the {n_active} active points and W holding "
        f"{weight_note}; "
        f"sigma^2 = SSE/df = {num(sigma2)} was the fit's residual variance, and the covariance "
        "of the estimates was taken as sigma^2 pinv(M). The points were chosen by a greedy "
        f"exchange over {candidates} evenly spaced candidates from {num(low)} to {num(high)}, "
        f"adding {request.replicates} row{'s' if request.replicates != 1 else ''} per location "
        f"(a new observation at weight 1), by {criterion}. The prediction half-width is the "
        f"shipped pointwise prediction interval at the {num(request.level * 100)} % level on "
        "the fit's residual degrees of freedom. The model form and the fitted covariance are "
        "assumed correct."
    )


__all__ = [
    "AdvisorRequest",
    "advise",
]
