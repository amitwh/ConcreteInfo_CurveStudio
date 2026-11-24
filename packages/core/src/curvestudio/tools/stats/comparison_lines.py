"""Deming and Passing-Bablok regression between two methods that measure one quantity.

Both tools fit the straight line that relates method B to method A, on the rows complete in
both methods, and report what the line says about whether the two methods agree.

Deming regression fits the line that allows for measurement error in both methods, weighted by
the ratio of their error variances. The slope is the closed-form solution of the Deming
equations, and its standard error, with the intercept's, is the delete-one jackknife estimate
over the pairs. The error-variance ratio is an assumption the fit makes, not a quantity these
data estimate, and the result says so.

Passing-Bablok regression makes no assumption about the distribution of the errors. It
estimates the slope as the shifted median of the slopes between every pair of points, and the
intercept as the median of the intercepts through that slope. Its confidence intervals are the
rank-based intervals of Passing and Bablok (1983); a straight-line relation between the two
methods is still assumed, and the plot is where a departure from it shows.

A line through the pairs is drawn against the line of identity, so a slope away from one or an
intercept away from zero is read at a glance. Every row left out is counted and listed, never
dropped without a word. A number the tool cannot give is ``None`` with the reason, never NaN.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    ToolResult,
    assumption,
    check_level,
    list_excluded,
    pairs,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    Line,
    ScatterPlot,
    Table,
    stride,
)
from curvestudio.tools.stats._tables import confidence_interval, table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_PAIRS = 3
"""The fewest complete pairs either method takes: a slope needs more than a line through two."""

MAX_PAIRS = 1000
"""The most complete pairs Passing-Bablok takes: every pair of them is a slope, a quadratic
number, and the tool is not built to hold a million of them silently."""

DEMING_MAX_PAIRS = 10_000
"""The most complete pairs Deming takes. Its standard errors come from a delete-one jackknife,
which refits the line once per pair: the cost is quadratic, about 0.8 s at this many pairs and
75 s at ten times it, so the tool refuses above it instead of tying a request up for a minute."""

TIE_EPS = 1e-12
"""Two readings count as equal when they differ by less than this fraction of their size, the
same rule the mcr package uses, so a pair of near-equal x values is not given a huge slope."""

DEMING = "Deming"

PASSING_BABLOK = "Passing\u2013Bablok"
"""The method's name as the result and the plot title show it, with its en dash."""

PARAMETER_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("parameter", "Parameter", "text"),
    ("estimate", "Estimate", "number"),
    ("se", "SE", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
)
"""The columns of the parameters table, shared by both tools."""


class DemingRequest(BaseModel):
    """The two methods, the ratio of their error variances, and the confidence level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_a: Column = Field(
        description=(
            "The readings of the first method, one per row. The fitted line predicts method B "
            "from this method."
        )
    )
    method_b: Column = Field(
        description="The readings of the second method, one per row, of the same quantity."
    )
    error_ratio: float = Field(
        default=1.0,
        description=(
            "The ratio of the error variance of method B to that of method A, above zero. One "
            "means both methods carry the same error variance; a larger value means method B "
            "is the noisier one."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class PassingBablokRequest(BaseModel):
    """The two methods and the confidence level; at most 1,000 complete pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_a: Column = Field(
        description=(
            "The readings of the first method, one per row. The fitted line predicts method B "
            "from this method."
        )
    )
    method_b: Column = Field(
        description="The readings of the second method, one per row, of the same quantity."
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Line(NamedTuple):
    """A fitted line: the slope and the intercept, with a bound for each where there is one."""

    slope: float
    intercept: float
    slope_lower: float | None
    slope_upper: float | None
    intercept_lower: float | None
    intercept_upper: float | None


def _require_varying(name: str, values: FloatArray) -> None:
    """Refuse a method that holds one value repeated: a slope against it is undefined.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` when ``values`` does not vary.
    """
    if float(np.min(values)) == float(np.max(values)):
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the method {name!r} holds one value repeated, so a line through it is undefined",
        )


def _require_positive_ratio(error_ratio: float) -> None:
    """Refuse an error ratio that is not a finite number above zero.

    Raises:
        ToolError: ``INVALID_OPTION`` when ``error_ratio`` is not finite and above zero.
    """
    if not math.isfinite(error_ratio) or error_ratio <= 0.0:
        raise ToolError(
            "INVALID_OPTION",
            f"the error ratio must be a number above zero, not {error_ratio!r}",
        )


def _moments(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """The sums of squares ``Sxx``, ``Syy`` and ``Sxy`` about the two means."""
    mx = float(np.mean(x))
    my = float(np.mean(y))
    xc = x - mx
    yc = y - my
    return float(xc @ xc), float(yc @ yc), float(xc @ yc)


def _deming_fit(x: FloatArray, y: FloatArray, error_ratio: float) -> tuple[float, float] | None:
    """The Deming slope and intercept of ``y`` on ``x`` for a given error-variance ratio.

    The slope solves ``Sxy b**2 + (error_ratio * Sxx - Syy) b - error_ratio * Sxy = 0``; the
    root whose sign follows the covariance is taken. When the covariance is exactly zero the
    fitted line is horizontal, so the slope is zero.

    Returns ``None`` when either column holds one value repeated, so no line is defined.
    """
    sxx, syy, sxy = _moments(x, y)
    if sxx == 0.0 or syy == 0.0:
        return None
    if sxy == 0.0:
        slope = 0.0
    else:
        difference = syy - error_ratio * sxx
        slope = (difference + math.sqrt(difference**2 + 4.0 * error_ratio * sxy**2)) / (2.0 * sxy)
    return slope, float(np.mean(y)) - slope * float(np.mean(x))


def _jackknife_se(values: Sequence[float]) -> float | None:
    """The delete-one jackknife standard error of ``values``, or ``None`` below two of them."""
    size = len(values)
    if size < 2:
        return None
    mean = sum(values) / size
    return math.sqrt((size - 1) / size * sum((value - mean) ** 2 for value in values))


def _jackknife(
    x: FloatArray, y: FloatArray, error_ratio: float
) -> tuple[float | None, float | None]:
    """The jackknife standard errors of the slope and the intercept, over the pairs.

    Each leave-one-out sample that still holds a line contributes one slope and one intercept;
    a sample in which a method has become constant is skipped, and the jackknife is taken over
    the samples that remain.
    """
    size = x.size
    slopes: list[float] = []
    intercepts: list[float] = []
    for leave_out in range(size):
        keep = np.ones(size, dtype=bool)
        keep[leave_out] = False
        fit = _deming_fit(x[keep], y[keep], error_ratio)
        if fit is not None:
            slopes.append(fit[0])
            intercepts.append(fit[1])
    return _jackknife_se(slopes), _jackknife_se(intercepts)


def _error_ratio_assumption(error_ratio: float) -> AuditCheck:
    """The error ratio the fit assumes, which the data cannot check."""
    return assumption(
        "error_ratio",
        "The ratio of the methods' error variances is an assumption of the fit",
        Verdict.NOT_APPLICABLE,
        "Deming regression weights each method by its measured error. The fit takes the ratio "
        f"of method B's error variance to method A's to be {num(error_ratio)}; the pairs "
        "themselves do not say what that ratio is.",
        next_step=(
            "If replicate measurements give the error of each method, set the ratio from them; "
            "otherwise the ratio of one assumes both methods carry the same error."
        ),
    )


def _linearity_assumption() -> AuditCheck:
    """The straight line Passing-Bablok assumes, which only the plot can show."""
    return assumption(
        "linearity",
        "A straight line between the two methods is what the method assumes",
        Verdict.NOT_APPLICABLE,
        "Passing-Bablok regression makes no assumption about how the errors are distributed, "
        "but it does assume that the relation between the methods is a straight line.",
        next_step="Look at the plot for a curve in the points or a change in their spread.",
    )


def _parameters_table(estimates: Sequence[Estimate], note: str) -> Table | None:
    """The parameters table when some estimate carries an interval, and ``None`` otherwise."""
    if not any(estimate.lower is not None or estimate.upper is not None for estimate in estimates):
        return None
    rows: list[tuple[Cell, ...]] = [
        (estimate.label, estimate.value, estimate.se, estimate.lower, estimate.upper)
        for estimate in estimates
    ]
    return table("Parameters", PARAMETER_COLUMNS, rows, note=note)


def _scatter(
    method_a: FloatArray,
    method_b: FloatArray,
    line: _Line,
    a_name: str,
    b_name: str,
    title: str,
) -> ScatterPlot:
    """Method B against method A, with the fitted line and the line of identity."""
    shown_x, note = stride(method_a)
    shown_y, _ = stride(method_b)
    low = float(min(method_a.min(), method_b.min()))
    high = float(max(method_a.max(), method_b.max()))
    return ScatterPlot(
        title=title,
        x_label=a_name,
        y_label=b_name,
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        lines=(
            Line(
                x0=low,
                y0=line.intercept + line.slope * low,
                x1=high,
                y1=line.intercept + line.slope * high,
                label="Fitted line",
            ),
            Line(x0=low, y0=low, x1=high, y1=high, label="Identity"),
        ),
        note=note,
    )


def _agreement(line: _Line, level: float) -> str:
    """What the intervals say about the methods agreeing on average, as a clause."""
    slope_lower, slope_upper = line.slope_lower, line.slope_upper
    intercept_lower, intercept_upper = line.intercept_lower, line.intercept_upper
    if (
        slope_lower is None
        or slope_upper is None
        or intercept_lower is None
        or intercept_upper is None
    ):
        return "The intervals could not be formed from these pairs, so they are not shown."
    percent = num(level * 100)
    interval = (
        f"At the {percent} % level the slope lies between {num(slope_lower)} and "
        f"{num(slope_upper)} and the intercept between {num(intercept_lower)} and "
        f"{num(intercept_upper)}"
    )
    if slope_lower <= 1.0 <= slope_upper and intercept_lower <= 0.0 <= intercept_upper:
        return f"{interval}, so the data are consistent with the two methods agreeing on average."
    return f"{interval}, so the data are not consistent with the two methods agreeing on average."


def _deming_interpretation(
    line: _Line,
    se_slope: float | None,
    se_intercept: float | None,
    error_ratio: float,
    a_name: str,
    b_name: str,
    level: float,
) -> str:
    """One to three sentences: the fit, what its interval says, and the ratio it assumes."""
    if se_slope is None or se_intercept is None:
        first = (
            f"A Deming fit of {b_name} on {a_name} gives a slope of {num(line.slope)} and an "
            f"intercept of {num(line.intercept)}; the jackknife standard error could not be "
            "computed from these pairs."
        )
    else:
        first = (
            f"A Deming fit of {b_name} on {a_name} gives a slope of {num(line.slope)} with a "
            f"jackknife standard error of {num(se_slope)} and an intercept of "
            f"{num(line.intercept)} with a standard error of {num(se_intercept)}."
        )
    sentences = [
        first,
        _agreement(line, level),
        f"The fit takes the ratio of the methods' error variances to be {num(error_ratio)}, "
        "which the data do not estimate.",
    ]
    return " ".join(sentences)


def _paba_interpretation(line: _Line, a_name: str, b_name: str, level: float) -> str:
    """One to three sentences: the fit, what its interval says, and the assumption it makes."""
    sentences = [
        f"{PASSING_BABLOK} regression of {b_name} on {a_name} gives a slope of "
        f"{num(line.slope)} and an intercept of {num(line.intercept)}, from the shifted median "
        "of the pairwise slopes.",
        _agreement(line, level),
        "The method assumes a straight line between the methods but no particular distribution "
        "of the errors.",
    ]
    return " ".join(sentences)


def _deming_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Deming regression fits a straight line that allows for measurement error in both "
        "methods, weighted by the ratio of their error variances. The slope is the closed-form "
        "solution of the Deming equations and the intercept follows from it. The standard "
        "errors of the slope and the intercept are the delete-one jackknife estimates over the "
        "pairs, and each interval is the estimate plus or minus t times its jackknife standard "
        "error on n - 1 degrees of freedom. The error-variance ratio is an assumption of the "
        "fit, not a quantity these data estimate."
    )


def _paba_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        f"{PASSING_BABLOK} regression estimates the slope as the shifted median of the slopes "
        "between every pair of points, and the intercept as the median of the intercepts "
        "through that slope. Its confidence intervals are the rank-based intervals of Passing "
        "and Bablok (1983). The method needs no assumption about the distribution of the "
        "errors, but it does assume that the relation between the methods is a straight line."
    )


def _paba(x: FloatArray, y: FloatArray, level: float) -> _Line:
    """The Passing-Bablok line and its rank-based intervals, as mcr and the 1983 paper give it.

    Every pair of points contributes the angle of its slope, with a pair of equal x values
    reading as a right angle when the methods rise together. The slope is the shifted median
    of those angles, the shift removing the slopes at or below minus one from the rank. The
    interval for the slope is the pair of angles at the rank the normal quantile and the
    sample size set, and the interval for the intercept is the median of the intercepts through
    those two slope bounds.
    """
    size = x.size
    positive = float(scipy_stats.kendalltau(x, y).statistic) >= 0.0
    left, right = np.triu_indices(size, k=1)
    dx = x[right] - x[left]
    dy = y[right] - y[left]
    same_x = dx == 0.0
    same_x |= np.abs(dx) < TIE_EPS * (np.abs(x[left]) + np.abs(x[right])) / 2.0
    same_y = dy == 0.0
    same_y |= np.abs(dy) < TIE_EPS * (np.abs(y[left]) + np.abs(y[right])) / 2.0
    regular = ~same_x
    angles = [np.arctan(dy[regular] / dx[regular])]
    vertical = same_x & ~same_y
    if bool(vertical.any()):
        right_angle = math.pi / 2.0 if positive else -math.pi / 2.0
        angles.append(np.full(int(np.count_nonzero(vertical)), right_angle))
    ordered = np.sort(np.concatenate(angles))
    total = ordered.size

    def order(index: int) -> float:
        return float(ordered[index])

    below = int(np.count_nonzero(ordered <= -math.pi / 4.0))
    below_strict = int(np.count_nonzero(ordered < -math.pi / 4.0))
    above = int(np.count_nonzero(ordered >= math.pi / 4.0))
    above_strict = int(np.count_nonzero(ordered > math.pi / 4.0))
    offset = below + below_strict if positive else -(above + above_strict)

    ranked = total + offset
    half = (ranked + 1) // 2
    if ranked % 2 == 0:
        slope = math.tan((order(half - 1) + order(half)) / 2.0)
    else:
        slope = math.tan(order(half - 1))

    quantile = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    span = math.floor(quantile * math.sqrt(size * (size - 1.0) * (2.0 * size + 5.0) / 18.0) + 0.5)
    lowest = 2 * (below - below_strict) + 1 if positive else 2 * (above - above_strict) + 1

    def slope_bound(rank: int, required: int, *, upper: bool) -> float | None:
        index = (rank + 1) // 2
        if upper:
            if index > total:
                return None
        elif index < required:
            return None
        if rank % 2 == 0:
            if index >= total:
                return None
            return math.tan((order(index - 1) + order(index)) / 2.0)
        return math.tan(order(index - 1))

    slope_lower = slope_bound(total - span + offset, lowest, upper=False)
    slope_upper = slope_bound(total + span + offset, total, upper=True)

    median = float(np.median(y - slope * x))
    intercept_lower = None if slope_upper is None else float(np.median(y - slope_upper * x))
    intercept_upper = None if slope_lower is None else float(np.median(y - slope_lower * x))
    return _Line(
        slope=slope,
        intercept=median,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
        intercept_lower=intercept_lower,
        intercept_upper=intercept_upper,
    )


def run_deming(request: BaseModel) -> ToolResult:
    """Run the Deming tool on its own request."""
    if not isinstance(request, DemingRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"deming takes a DemingRequest, not a request of {type(request).__name__}",
        )
    return _analyse_deming(request)


def _analyse_deming(request: DemingRequest) -> ToolResult:
    """The Deming fit, once the request is known to be this tool's own."""
    check_level(request.level)
    _require_positive_ratio(request.error_ratio)
    pairing = pairs(request.method_a, request.method_b)
    a, b = pairing.columns
    size = a.values.size
    if size < MIN_PAIRS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"Deming regression needs at least {MIN_PAIRS} complete pairs, and {size} "
            f"{'is' if size == 1 else 'are'} left",
        )
    if size > DEMING_MAX_PAIRS:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"Deming regression takes at most {DEMING_MAX_PAIRS:,} complete pairs, and "
            f"{size:,} are left: its jackknife standard errors cost a refit per pair",
        )
    _require_varying(a.name, a.values)
    _require_varying(b.name, b.values)
    fit = _deming_fit(a.values, b.values, request.error_ratio)
    assert fit is not None  # both methods vary, so a line exists
    slope, intercept = fit
    se_slope, se_intercept = _jackknife(a.values, b.values, request.error_ratio)
    df = float(size - 1)
    slope_lower, slope_upper = (
        (None, None)
        if se_slope is None
        else confidence_interval(slope, se_slope, df, request.level)
    )
    intercept_lower, intercept_upper = (
        (None, None)
        if se_intercept is None
        else confidence_interval(intercept, se_intercept, df, request.level)
    )
    line = _Line(
        slope=slope,
        intercept=intercept,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
        intercept_lower=intercept_lower,
        intercept_upper=intercept_upper,
    )
    estimates = (
        Estimate(
            label="Slope",
            value=slope,
            se=se_slope,
            lower=slope_lower,
            upper=slope_upper,
        ),
        Estimate(
            label="Intercept",
            value=intercept,
            se=se_intercept,
            lower=intercept_lower,
            upper=intercept_upper,
        ),
    )
    parameters = _parameters_table(
        estimates,
        note=(
            "Deming regression; jackknife standard errors. SE is the delete-one jackknife "
            "estimate over the pairs, and each interval is the estimate plus or minus t times "
            "its SE on n - 1 degrees of freedom at the stated level. The error ratio is the "
            "ratio of method B's error variance to method A's."
        ),
    )
    tables = () if parameters is None else (parameters,)
    excluded = len(pairing.excluded)
    listed, unlisted = list_excluded(pairing.excluded)
    return ToolResult(
        tool="deming",
        title=f"{DEMING} regression of {b.name} on {a.name}",
        level=request.level,
        estimates=estimates,
        assumptions=(_error_ratio_assumption(request.error_ratio),),
        interpretation=_deming_interpretation(
            line, se_slope, se_intercept, request.error_ratio, a.name, b.name, request.level
        ),
        methods_fragment=_deming_methods(),
        counts=Counts(n_offered=size + excluded, n_used=size, n_excluded=excluded),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tables,
        plots=(
            _scatter(
                a.values,
                b.values,
                line,
                a.name,
                b.name,
                f"{DEMING} regression",
            ),
        ),
    )


def run_passingbablok(request: BaseModel) -> ToolResult:
    """Run the Passing-Bablok tool on its own request."""
    if not isinstance(request, PassingBablokRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"passingbablok takes a PassingBablokRequest, not a request of "
            f"{type(request).__name__}",
        )
    return _analyse_passingbablok(request)


def _analyse_passingbablok(request: PassingBablokRequest) -> ToolResult:
    """The Passing-Bablok fit, once the request is known to be this tool's own."""
    check_level(request.level)
    pairing = pairs(request.method_a, request.method_b)
    a, b = pairing.columns
    size = a.values.size
    if size < MIN_PAIRS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"Passing-Bablok regression needs at least {MIN_PAIRS} complete pairs, and {size} "
            f"{'is' if size == 1 else 'are'} left",
        )
    if size > MAX_PAIRS:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"Passing-Bablok regression takes at most {MAX_PAIRS:,} complete pairs, and "
            f"{size:,} are left",
        )
    _require_varying(a.name, a.values)
    _require_varying(b.name, b.values)
    line = _paba(a.values, b.values, request.level)
    estimates = (
        Estimate(
            label="Slope",
            value=line.slope,
            lower=line.slope_lower,
            upper=line.slope_upper,
        ),
        Estimate(
            label="Intercept",
            value=line.intercept,
            lower=line.intercept_lower,
            upper=line.intercept_upper,
        ),
    )
    parameters = _parameters_table(
        estimates,
        note=(
            f"{PASSING_BABLOK} regression. The estimate is the shifted median of the pairwise "
            "slopes, and each interval is the rank-based interval of Passing and Bablok (1983) "
            "at the stated level. The method gives no standard error."
        ),
    )
    tables = () if parameters is None else (parameters,)
    excluded = len(pairing.excluded)
    listed, unlisted = list_excluded(pairing.excluded)
    return ToolResult(
        tool="passingbablok",
        title=f"{PASSING_BABLOK} regression of {b.name} on {a.name}",
        level=request.level,
        estimates=estimates,
        assumptions=(_linearity_assumption(),),
        interpretation=_paba_interpretation(line, a.name, b.name, request.level),
        methods_fragment=_paba_methods(),
        counts=Counts(n_offered=size + excluded, n_used=size, n_excluded=excluded),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tables,
        plots=(
            _scatter(
                a.values,
                b.values,
                line,
                a.name,
                b.name,
                f"{PASSING_BABLOK} regression",
            ),
        ),
    )


TOOLS.register(
    ToolSpec(
        name="deming",
        family="comparison",
        title=f"{DEMING} regression",
        request=DemingRequest,
        run=run_deming,
        max_rows=DEMING_MAX_PAIRS,
    )
)

TOOLS.register(
    ToolSpec(
        name="passingbablok",
        family="comparison",
        title=f"{PASSING_BABLOK} regression",
        request=PassingBablokRequest,
        run=run_passingbablok,
        max_rows=MAX_PAIRS,
    )
)
