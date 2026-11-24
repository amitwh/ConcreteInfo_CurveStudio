"""Bland-Altman agreement between two methods that measure the same quantity.

The tool aligns the two methods on the rows complete in both, takes the difference of each
pair (method A minus method B) and its mean, and reports what agreement they show: the bias,
the standard deviation of the differences and the limits of agreement at the bias plus or
minus 1.96 standard deviations. Each of the three carries an interval: the bias its t interval
on n - 1 degrees of freedom, and each limit the interval whose standard error is the one
Bland and Altman (1999) give, ``s * sqrt(1/n + 1.96**2 / (2 * (n - 1)))``, on the same
degrees of freedom. The points beyond the limits are listed, never removed.

Proportional bias is checked by the ordinary least squares regression of the difference on
the mean of the two methods: a slope away from zero means the difference changes with the
size of the reading. The assumptions the method makes are checked where the data can show
them: Shapiro-Wilk's W test for normality of the differences, and the studentized
Breusch-Pagan test for a variance that is constant across the range.

Two plots show the comparison: the difference against the mean, with the bias, both limits
and zero drawn, and method A against method B with the line of equality. A pair that the two
methods agree on exactly adds no evidence of disagreement, but every difference of zero
leaves the slope undefined; the result then says so rather than inventing a test.

A p-value is a statement about evidence, never a statement of proof. Every row left out is
counted and listed, never dropped without a word.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.data.dataset import IntArray
from curvestudio.diagnostics import breusch_pagan, shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import num, p_threshold
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    assumption,
    check_level,
    list_excluded,
    p_text,
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
from curvestudio.tools.stats._tables import confidence_interval, mean_ci, table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_PAIRS = 3
"""The fewest complete pairs the comparison takes: a spread and a slope both need one."""

LIMIT_MULTIPLIER = 1.96
"""How many standard deviations of the differences sit between the bias and each limit."""

LIMIT_PERCENT = 95
"""The share of differences the limits of agreement are expected to cover."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

EXACT_RATIO = 1e-20
"""The residuals count as exactly zero when their sum of squares is this small a fraction of
the differences' total: a perfect line leaves only the rounding of a float behind."""

BLAND_ALTMAN = "Bland\u2013Altman"
"""The method's name as the result and the plot title show it, with its en dash."""

OUTSIDE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("row", "Row", "int"),
    ("mean", "Mean", "number"),
    ("difference", "Difference", "number"),
)
"""The columns of the table of points beyond the limits of agreement."""


class BlandAltmanRequest(BaseModel):
    """The two methods and the confidence level of every interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_a: Column = Field(
        description=(
            "The readings of the first method, one per row. The difference of a pair is this "
            "method's reading minus the second's."
        )
    )
    method_b: Column = Field(
        description="The readings of the second method, one per row, of the same quantity."
    )
    level: float = Field(
        default=0.95,
        description=(
            "The confidence level of every interval, between 0 and 1. The limits of agreement "
            "themselves stay at 95 %."
        ),
    )


class _Fit(NamedTuple):
    """The ordinary least squares fit of the difference on the mean of the two methods.

    ``se``, ``statistic``, ``p_value``, ``lower`` and ``upper`` are ``None`` when the slope's
    spread cannot be estimated; ``reason`` then says why. ``intercept`` and ``slope`` are
    still the fit's own, which the property tests read.
    """

    intercept: float
    slope: float
    se: float | None
    statistic: float | None
    p_value: float | None
    lower: float | None
    upper: float | None
    reason: str | None


def _difference_fit(means: FloatArray, differences: FloatArray, level: float) -> _Fit:
    """Fit ``differences`` on ``means`` by ordinary least squares.

    The intercept is the difference at a mean of zero, and the slope is the change in the
    difference per unit of the mean: a slope away from zero is proportional bias.

    Raises:
        ToolError: ``INVALID_OPTION`` for a confidence level outside (0, 1).
    """
    check_level(level)
    n = differences.size
    mean_x = float(np.mean(means))
    mean_y = float(np.mean(differences))
    centred = means - mean_x
    sxx = float(centred @ centred)
    sst = float(((differences - mean_y) ** 2).sum())
    if sxx <= 0.0:
        return _Fit(
            intercept=mean_y,
            slope=0.0,
            se=None,
            statistic=None,
            p_value=None,
            lower=None,
            upper=None,
            reason=(
                "the mean of the two methods holds one value, so there is no range of means "
                "to regress the difference on"
            ),
        )
    slope = float(centred @ (differences - mean_y)) / sxx
    intercept = mean_y - slope * mean_x
    residual = differences - (intercept + slope * means)
    sse = float(residual @ residual)
    if sst <= 0.0:
        return _Fit(
            intercept=intercept,
            slope=slope,
            se=None,
            statistic=None,
            p_value=None,
            lower=None,
            upper=None,
            reason=(
                "every difference is the same, so the difference does not vary with the mean "
                "and its slope cannot be tested"
            ),
        )
    if sse <= EXACT_RATIO * sst:
        return _Fit(
            intercept=intercept,
            slope=slope,
            se=None,
            statistic=None,
            p_value=None,
            lower=None,
            upper=None,
            reason=(
                "the differences lie exactly on a straight line, so the slope's standard "
                "error is zero and its test is undefined"
            ),
        )
    df = n - 2
    se = math.sqrt((sse / df) / sxx)
    statistic = slope / se
    p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), df))
    lower, upper = confidence_interval(slope, se, float(df), level)
    return _Fit(
        intercept=intercept,
        slope=slope,
        se=se,
        statistic=statistic,
        p_value=p_value,
        lower=lower,
        upper=upper,
        reason=None,
    )


def _limit_se(sd: float, n: int) -> float:
    """The standard error of a limit of agreement, by Bland and Altman (1999).

    The limit is ``d - k s`` with ``k = 1.96``. The sample mean and the sample standard
    deviation of normal differences are independent, so the limit's variance is the mean's
    variance plus ``k**2`` times the standard deviation's: ``s**2 / n`` plus
    ``k**2 * s**2 / (2 (n - 1))``.
    """
    return sd * math.sqrt(1.0 / n + LIMIT_MULTIPLIER**2 / (2.0 * (n - 1)))


def _verdict(p_value: float | None) -> Verdict:
    """The verdict a p-value gives an assumption."""
    if p_value is None:
        return Verdict.NOT_APPLICABLE
    if p_value < FAIL_P:
        return Verdict.FAIL
    if p_value < CAUTION_P:
        return Verdict.CAUTION
    return Verdict.PASS


def _normality(differences: FloatArray) -> AuditCheck:
    """The Shapiro-Wilk check of normality of the differences."""
    found = shapiro_wilk(differences)
    if found.status != "ok" or found.p_value is None:
        return assumption(
            "normality",
            "The normality of the differences could not be checked",
            Verdict.NOT_APPLICABLE,
            found.reason or "Shapiro-Wilk's W test could not be run on these differences.",
        )
    assert found.statistic is not None
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: "The differences are consistent with a normal distribution",
        Verdict.CAUTION: "The differences leave the normal assumption in doubt",
        Verdict.FAIL: "The differences show evidence against a normal distribution",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the differences are far from normal, the limits and their intervals rely on "
        "an approximation; with many pairs the test also reacts to departures of no practical "
        "size."
    )
    return assumption(
        "normality",
        titles[verdict],
        verdict,
        f"Shapiro-Wilk's W test gives W = {num(found.statistic)}, {p_text(found.p_value)}.",
        statistic=f"W = {num(found.statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _constant_variance(differences: FloatArray, means: FloatArray, fit: _Fit) -> AuditCheck:
    """The studentized Breusch-Pagan check that the differences vary equally across the range."""
    fitted = fit.intercept + fit.slope * means
    residual = differences - fitted
    found = breusch_pagan(residual, fitted)
    if found.status != "ok" or found.p_value is None:
        return assumption(
            "constant_variance",
            "The constancy of the difference variance could not be checked",
            Verdict.NOT_APPLICABLE,
            found.reason or "The Breusch-Pagan test could not be run on these differences.",
        )
    assert found.statistic is not None
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: "The differences are equally variable across the range",
        Verdict.CAUTION: "The variance of the differences may change with the mean",
        Verdict.FAIL: "The variance of the differences changes with the mean",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the spread of the differences grows with the mean, one width for the limits "
        "does not fit the whole range; a transformation or a regression on the mean may help."
    )
    return assumption(
        "constant_variance",
        titles[verdict],
        verdict,
        f"The studentized Breusch-Pagan test gives LM = {num(found.statistic)}, "
        f"{p_text(found.p_value)}.",
        statistic=f"LM = {num(found.statistic)}",
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _limit(label: str, value: float, se: float, df: float, level: float) -> Estimate:
    """One limit of agreement with its interval at ``level`` on ``df`` degrees of freedom."""
    lower, upper = confidence_interval(value, se, df, level)
    return Estimate(label=label, value=value, se=se, lower=lower, upper=upper)


def _estimates(bias_estimate: Estimate, sd: float, n: int, level: float) -> tuple[Estimate, ...]:
    """The bias, the two limits and the standard deviation, in the order a person reads them."""
    se = _limit_se(sd, n)
    df = float(n - 1)
    lower_value = bias_estimate.value - LIMIT_MULTIPLIER * sd
    upper_value = bias_estimate.value + LIMIT_MULTIPLIER * sd
    return (
        bias_estimate,
        _limit("Lower limit of agreement", lower_value, se, df, level),
        _limit("Upper limit of agreement", upper_value, se, df, level),
        Estimate(label="Standard deviation of the differences", value=sd),
    )


def _slope_test(fit: _Fit, n: int, level: float) -> HypothesisTest:
    """The t test of the fit's slope, which is the check for proportional bias."""
    effect = (
        None
        if fit.se is None or fit.lower is None or fit.upper is None
        else Estimate(
            label="Slope",
            value=fit.slope,
            se=fit.se,
            lower=fit.lower,
            upper=fit.upper,
        )
    )
    return HypothesisTest(
        name="Ordinary least squares of the difference on the mean",
        method=(
            "Ordinary least squares regression of the difference on the mean of the two "
            "methods, t test of the slope"
        ),
        null_hypothesis=(
            "The slope of the difference on the mean of the two methods is zero, so the "
            "difference shows no proportional bias."
        ),
        statistic=fit.statistic,
        df=None if fit.statistic is None else float(n - 2),
        p_value=fit.p_value,
        p_value_text=p_text(fit.p_value),
        effect=effect,
        reason=fit.reason,
    )


def _outside(
    positions: IntArray,
    means: FloatArray,
    differences: FloatArray,
    lower: float,
    upper: float,
) -> Table:
    """The points whose difference falls beyond a limit, in position order."""
    rows: list[tuple[Cell, ...]] = [
        (int(position), float(mean), float(difference))
        for position, mean, difference in zip(positions, means, differences, strict=True)
        if float(difference) < lower or float(difference) > upper
    ]
    return table(
        "Points outside the limits of agreement",
        OUTSIDE_COLUMNS,
        rows,
        note=(
            "A point lies outside when its difference is below the lower limit or above the "
            "upper one. Row is the row's position in the data, counting from zero; the "
            "points are kept in every statistic."
        ),
    )


def _bland_altman_plot(
    means: FloatArray,
    differences: FloatArray,
    bias: float,
    lower: float,
    upper: float,
    method_a: str,
    method_b: str,
) -> ScatterPlot:
    """The difference against the mean, with the bias, both limits and zero drawn."""
    shown_x, note = stride(means)
    shown_y, _ = stride(differences)
    low, high = float(means.min()), float(means.max())
    return ScatterPlot(
        title=BLAND_ALTMAN,
        x_label=f"Mean of {method_a} and {method_b}",
        y_label=f"Difference ({method_a} minus {method_b})",
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        lines=(
            Line(x0=low, y0=bias, x1=high, y1=bias, label="Bias"),
            Line(x0=low, y0=lower, x1=high, y1=lower, label="Lower limit of agreement"),
            Line(x0=low, y0=upper, x1=high, y1=upper, label="Upper limit of agreement"),
            Line(x0=low, y0=0.0, x1=high, y1=0.0, label="Zero"),
        ),
        note=note,
    )


def _identity_plot(
    method_a: FloatArray,
    method_b: FloatArray,
    a_name: str,
    b_name: str,
) -> ScatterPlot:
    """Method A against method B, with the line of equality."""
    shown_x, note = stride(method_b)
    shown_y, _ = stride(method_a)
    low = float(min(method_a.min(), method_b.min()))
    high = float(max(method_a.max(), method_b.max()))
    return ScatterPlot(
        title=f"{a_name} against {b_name}",
        x_label=b_name,
        y_label=a_name,
        x=tuple(float(value) for value in shown_x),
        y=tuple(float(value) for value in shown_y),
        lines=(Line(x0=low, y0=low, x1=high, y1=high, label="Identity"),),
        note=note,
    )


def _evidence(fit: _Fit) -> str:
    """What the slope's p-value says about proportional bias, as a clause of evidence."""
    if fit.p_value is None:
        return f"the slope could not be tested because {fit.reason}"
    if fit.p_value >= CAUTION_P:
        return "the data are consistent with no proportional bias"
    if fit.p_value >= FAIL_P:
        return "the data show evidence of proportional bias"
    return "the data show strong evidence of proportional bias"


def _interpretation(
    bias_estimate: Estimate,
    estimates: Sequence[Estimate],
    fit: _Fit,
    level: float,
) -> str:
    """One to three sentences: the bias and the limits, then the check for proportional bias."""
    assert bias_estimate.lower is not None and bias_estimate.upper is not None
    sentences = [
        f"The mean difference (method A minus method B) is {num(bias_estimate.value)}, with a "
        f"{num(level * 100)} % interval from {num(bias_estimate.lower)} to "
        f"{num(bias_estimate.upper)}; {LIMIT_PERCENT} % of differences are expected to lie "
        f"between the limits of agreement, {num(estimates[1].value)} and "
        f"{num(estimates[2].value)}."
    ]
    if fit.p_value is None:
        sentences.append(
            f"The slope of the difference on the mean could not be tested because {fit.reason}."
        )
    else:
        sentences.append(
            f"The slope of the difference on the mean is {num(fit.slope)} "
            f"({p_text(fit.p_value)}), so {_evidence(fit)}."
        )
    return " ".join(sentences)


def _methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Bland-Altman agreement compares two methods that measure the same quantity. The bias "
        "is the mean difference, method A minus method B, with its interval from the t "
        "distribution on n - 1 degrees of freedom. The limits of agreement are the bias plus "
        "or minus 1.96 standard deviations of the differences; the standard error of a limit "
        "is the standard deviation times the square root of 1/n plus 1.96 squared over twice "
        "n - 1, from Bland and Altman (1999), and its interval uses the same t distribution. "
        "Proportional bias is the slope of the ordinary least squares regression of the "
        "difference on the mean of the two methods, tested with the t statistic on n - 2 "
        "degrees of freedom. Shapiro-Wilk's W test checks normality of the differences, and "
        "the studentized Breusch-Pagan test checks that their variance is constant across the "
        "range."
    )


def run_blandaltman(request: BaseModel) -> ToolResult:
    """Run the Bland-Altman tool on its own request."""
    if not isinstance(request, BlandAltmanRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"blandaltman takes a BlandAltmanRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: BlandAltmanRequest) -> ToolResult:
    """The comparison itself, once the request is known to be this tool's own."""
    check_level(request.level)
    pairing = pairs(request.method_a, request.method_b)
    a = pairing.columns[0]
    b = pairing.columns[1]
    n = a.values.size
    if n < MIN_PAIRS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the comparison needs at least {MIN_PAIRS} complete pairs, and {n} "
            f"{'is' if n == 1 else 'are'} left",
        )
    differences = a.values - b.values
    means = (a.values + b.values) / 2.0
    bias_estimate = mean_ci(differences, "Bias", request.level)
    sd = float(np.std(differences, ddof=1))
    estimates = _estimates(bias_estimate, sd, n, request.level)
    fit = _difference_fit(means, differences, request.level)
    lower_limit = float(estimates[1].value)
    upper_limit = float(estimates[2].value)

    tables: list[Table] = []
    outside = _outside(a.rows, means, differences, lower_limit, upper_limit)
    if outside.rows:
        tables.append(outside)

    warnings: list[str] = []
    if bool(np.all(differences == 0.0)):
        warnings.append(
            "Every difference is zero: the two methods gave the same reading on every row, "
            "so the limits of agreement are zero and the slope's test cannot run."
        )

    excluded = len(pairing.excluded)
    listed, unlisted = list_excluded(pairing.excluded)
    plots = (
        _bland_altman_plot(
            means,
            differences,
            bias_estimate.value,
            lower_limit,
            upper_limit,
            a.name,
            b.name,
        ),
        _identity_plot(a.values, b.values, a.name, b.name),
    )
    return ToolResult(
        tool="blandaltman",
        title=f"{BLAND_ALTMAN} agreement between {a.name} and {b.name}",
        level=request.level,
        estimates=estimates,
        tests=(_slope_test(fit, n, request.level),),
        assumptions=(
            _normality(differences),
            _constant_variance(differences, means, fit),
        ),
        warnings=tuple(warnings),
        interpretation=_interpretation(bias_estimate, estimates, fit, request.level),
        methods_fragment=_methods(),
        counts=Counts(
            n_offered=n + excluded,
            n_used=n,
            n_excluded=excluded,
        ),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=plots,
    )


TOOLS.register(
    ToolSpec(
        name="blandaltman",
        family="comparison",
        title=f"{BLAND_ALTMAN} agreement",
        request=BlandAltmanRequest,
        run=run_blandaltman,
    )
)
