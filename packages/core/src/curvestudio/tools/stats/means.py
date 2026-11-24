"""The t tests: one sample, two independent samples and paired samples.

Each test answers one question about a mean or a difference of means: is it the value the
null hypothesis names. The result carries the difference with its interval, the test with
its statistic, degrees of freedom and p-value, and the standardised effect size with its
interval: Hedges' g for the one-sample and two-sample tests, Cohen's dz for the paired
test. Every interval comes from the t distribution, and every effect-size interval comes
from inverting the non-central t distribution, so the interval and the p-value can never
tell different stories.

A p-value is a statement about evidence, never a statement of proof: a small one is
evidence against the null hypothesis, and a large one is not evidence that the null is
true. Every assumption the tests make is checked where the data can show it: Shapiro-Wilk
for normality, Levene's median-centred test for equal variances under Student's test, and
the completeness of the pairs under the paired test.
"""

import math
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import optimize
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.diagnostics import shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import num, p_threshold
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    aligned,
    assumption,
    check_level,
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import FloatArray, box_plot
from curvestudio.tools.stats._tables import (
    GroupSummary,
    confidence_interval,
    group_summary,
    group_table,
)
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

ONE_SAMPLE_MIN = 3
"""The fewest values a one-sample test takes: its Hedges' g correction needs two df."""

GROUP_MIN = 2
"""The fewest values each group of a two-sample test takes: a spread needs two."""

PAIRED_MIN = 2
"""The fewest complete pairs a paired test takes."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

_NULL_LABEL = "Hypothesised value"
"""What the hypothesised mean or difference is called on a form."""

_KIND_LABEL = "Kind of test"


class TTestRequest(BaseModel):
    """One t test: what is compared, against which value, and at which confidence level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["one_sample", "independent", "paired"] = Field(
        description=(
            "The test: one sample against a hypothesised mean, two independent groups "
            "against a hypothesised difference, or paired observations against a "
            "hypothesised mean difference."
        ),
        json_schema_extra={"label": _KIND_LABEL},
    )
    a: Column = Field(
        description=(
            "The first column: the values of the one-sample test, the values with a group "
            "label for the independent test, or the first column of a pair."
        )
    )
    b: Column | None = Field(
        default=None,
        description=(
            "The second column: the other group of an independent test when no group "
            "column is given, or the second column of a pair. Left empty for a one-sample "
            "test and for the paired test on the differences in the first column."
        ),
    )
    groups: GroupColumn | None = Field(
        default=None,
        description=(
            "For the independent test, a group column with exactly two levels; the test "
            "compares the two levels' means. Left empty when the two groups are two columns."
        ),
    )
    null: float = Field(
        default=0.0,
        description=(
            "The hypothesised mean of the one-sample test, or the hypothesised difference "
            "of the independent and paired tests."
        ),
        json_schema_extra={"label": _NULL_LABEL},
    )
    equal_variances: bool = Field(
        default=False,
        description=(
            "For the independent test, assume the two groups have equal variances and run "
            "Student's test; left false, Welch's test does not assume it."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Sample(NamedTuple):
    """One group's values and the few numbers every form of the test needs from them."""

    label: str
    values: FloatArray
    mean: float
    sd: float
    n: int


class _Statistic(NamedTuple):
    """The test itself: its statistic, degrees of freedom, p-value and standard error."""

    statistic: float
    df: float
    p_value: float
    se: float


def run_ttest(request: BaseModel) -> ToolResult:
    """Run the t test on its own request."""
    if not isinstance(request, TTestRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"ttest takes a TTestRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: TTestRequest) -> ToolResult:
    """The test itself, once the request is known to be this tool's own."""
    check_level(request.level)
    _check_fields(request)
    if not math.isfinite(request.null):
        raise ToolError(
            "INVALID_OPTION",
            f"the hypothesised value must be a finite number, not {request.null}",
        )
    if request.kind == "one_sample":
        return _one_sample(request)
    if request.kind == "independent":
        return _independent(request)
    return _paired(request)


def _check_fields(request: TTestRequest) -> None:
    """Refuse a field the chosen kind of test does not take.

    Raises:
        ToolError: ``INVALID_OPTION`` naming the field that does not fit the kind.
    """
    if request.kind == "one_sample":
        if request.b is not None:
            raise ToolError(
                "INVALID_OPTION",
                "the one-sample test takes one column, and a second column was given; the "
                "paired test takes the second column, or the differences in the first",
            )
        if request.groups is not None:
            raise ToolError(
                "INVALID_OPTION", "the one-sample test takes one column, not a group column"
            )
        if request.equal_variances:
            raise ToolError(
                "INVALID_OPTION",
                "the one-sample test has no second group whose variance could be compared",
            )
    elif request.kind == "independent":
        if request.b is not None and request.groups is not None:
            raise ToolError(
                "INVALID_OPTION",
                "the independent test takes a group column or a second column, not both",
            )
        if request.b is None and request.groups is None:
            raise ToolError(
                "INVALID_OPTION",
                "the independent test takes a group column with two levels, or a second "
                "column, and neither was given",
            )
    else:
        if request.groups is not None:
            raise ToolError(
                "INVALID_OPTION",
                "the paired test pairs two columns; it takes no group column",
            )
        if request.equal_variances:
            raise ToolError(
                "INVALID_OPTION", "the paired test does not compare two groups' variances"
            )


def _sample(label: str, values: FloatArray) -> _Sample:
    """One group reduced to its finite values, with its mean and sample SD.

    An empty group comes back with no values and zero mean and spread, so the caller's
    size check names the refusal.
    """
    flat = np.asarray(values, dtype=np.float64)
    flat = flat[np.isfinite(flat)]
    n = int(flat.size)
    if n == 0:
        return _Sample(label=str(label), values=flat, mean=0.0, sd=0.0, n=0)
    sd = float(np.std(flat, ddof=1)) if n >= 2 else 0.0
    return _Sample(label=str(label), values=flat, mean=float(np.mean(flat)), sd=sd, n=n)


def _finite_column(column: Column) -> tuple[FloatArray, tuple[ExcludedRow, ...]]:
    """One independent column's finite values, and the positions it leaves out and why."""
    values = np.asarray(column.values, dtype=np.float64)
    finite = np.isfinite(values)
    excluded: list[ExcludedRow] = []
    for at in np.flatnonzero(~finite):
        value = float(values[at])
        reason = (
            f"missing in {column.name}"
            if math.isnan(value)
            else f"not a finite number in {column.name}"
        )
        excluded.append(ExcludedRow(index=int(column.rows[at]), reason=reason))
    return values[finite], tuple(excluded)


def _require_group_size(sample: _Sample) -> None:
    """Refuse a group too small for its spread to be estimated.

    Raises:
        ToolError: ``GROUP_OF_ONE`` for a group of one, ``TOO_FEW_ROWS`` for an empty one.
    """
    if sample.n >= GROUP_MIN:
        return
    if sample.n == 1:
        raise ToolError(
            "GROUP_OF_ONE",
            f"the group {sample.label!r} holds 1 value, so its spread cannot be estimated",
        )
    raise ToolError(
        "TOO_FEW_ROWS",
        f"the group {sample.label!r} holds no usable value, and each group needs at least "
        f"{GROUP_MIN}",
    )


def _one_sample_statistic(sample: _Sample, null: float) -> _Statistic:
    """The one-sample t test on ``sample`` against ``null``.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` when every value is the same, so the test is undefined.
    """
    if sample.sd == 0.0:
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the column {sample.label!r} holds one value repeated, so the t statistic is "
            "undefined",
        )
    se = sample.sd / math.sqrt(sample.n)
    statistic = (sample.mean - null) / se
    df = float(sample.n - 1)
    return _Statistic(
        statistic=statistic,
        df=df,
        p_value=float(2.0 * scipy_stats.t.sf(abs(statistic), df)),
        se=se,
    )


def _independent_statistic(
    first: _Sample, second: _Sample, null: float, *, equal_variances: bool
) -> _Statistic:
    """Welch's or Student's two-sample t test of ``first`` less ``second`` against ``null``.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` when no group varies, so the pooled spread is zero.
    """
    if first.sd == 0.0 and second.sd == 0.0:
        raise _constant(first, second)
    if equal_variances:
        degrees: float = float(first.n + second.n - 2)
        pooled_variance = ((first.n - 1) * first.sd**2 + (second.n - 1) * second.sd**2) / degrees
        pooled_sd = math.sqrt(pooled_variance)
        se = pooled_sd * math.sqrt(1.0 / first.n + 1.0 / second.n)
    else:
        first_variance = first.sd**2 / first.n
        second_variance = second.sd**2 / second.n
        degrees = (first_variance + second_variance) ** 2 / (
            first_variance**2 / (first.n - 1) + second_variance**2 / (second.n - 1)
        )
        se = math.sqrt(first_variance + second_variance)
    statistic = (first.mean - second.mean - null) / se
    return _Statistic(
        statistic=statistic,
        df=float(degrees),
        p_value=float(2.0 * scipy_stats.t.sf(abs(statistic), degrees)),
        se=se,
    )


def _constant(first: _Sample, second: _Sample) -> ToolError:
    """The refusal for two groups whose combined spread is zero."""
    return ToolError(
        "CONSTANT_COLUMN",
        f"the groups {first.label!r} and {second.label!r} hold no variation between them, "
        "so the t statistic is undefined",
    )


def _paired_statistic(differences: _Sample, null: float) -> _Statistic:
    """The paired t test on the differences against ``null``.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` when every difference is the same.
    """
    if differences.sd == 0.0:
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the differences {differences.label!r} are all the same, so the t statistic is "
            "undefined",
        )
    se = differences.sd / math.sqrt(differences.n)
    statistic = (differences.mean - null) / se
    df = float(differences.n - 1)
    return _Statistic(
        statistic=statistic,
        df=df,
        p_value=float(2.0 * scipy_stats.t.sf(abs(statistic), df)),
        se=se,
    )


def _noncentral_interval(statistic: float, df: float, level: float) -> tuple[float, float]:
    """The interval for the non-centrality parameter behind ``statistic`` at ``df``.

    The interval is the set of non-centrality values whose non-central t distribution puts
    the observed statistic in its central ``level`` region. Its bounds are the roots of the
    non-central t CDF at the observed statistic, found by expanding a bracket and bisecting.
    """
    alpha = 1.0 - level

    def boundary(target: float) -> float:
        def excess(noncentrality: float) -> float:
            return float(scipy_stats.nct.cdf(statistic, df, noncentrality)) - target

        low = statistic - 1.0
        step = 1.0
        for _ in range(200):
            if excess(low) > 0.0:
                break
            low -= step
            step *= 2.0
        high = statistic + 1.0
        step = 1.0
        for _ in range(200):
            if excess(high) < 0.0:
                break
            high += step
            step *= 2.0
        return float(optimize.brentq(excess, low, high))

    return boundary(1.0 - alpha / 2.0), boundary(alpha / 2.0)


def _standardised_effect(
    statistic: float,
    df: float,
    scale: float,
    level: float,
    *,
    corrected: bool,
    label: str,
) -> Estimate:
    """A standardised effect size with its interval, from the non-central t inversion.

    ``scale`` turns the non-centrality parameter into the effect size: the standard error
    divided by the standardising deviation. ``corrected`` applies Hedges' small-sample
    factor ``1 - 3/(4 df - 1)``, which the one-sample and two-sample tests use; Cohen's dz
    for the paired test does not.
    """
    if corrected and df < 2.0:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"Hedges' g needs more than one degree of freedom, and this test has {num(df)}",
        )
    lower, upper = _noncentral_interval(statistic, df, level)
    correction = 1.0 - 3.0 / (4.0 * df - 1.0) if corrected else 1.0
    return Estimate(
        label=label,
        value=correction * statistic * scale,
        lower=correction * lower * scale,
        upper=correction * upper * scale,
    )


def _difference(label: str, value: float, se: float, df: float, level: float) -> Estimate:
    """The estimated difference with its t interval."""
    lower, upper = confidence_interval(value, se, df, level)
    return Estimate(label=label, value=value, se=se, lower=lower, upper=upper)


def _bounds(estimate: Estimate) -> tuple[float, float]:
    """An estimate's interval, which every estimate built here carries."""
    assert estimate.lower is not None and estimate.upper is not None
    return estimate.lower, estimate.upper


def _verdict(p_value: float | None) -> Verdict:
    """The verdict a p-value gives an assumption."""
    if p_value is None:
        return Verdict.NOT_APPLICABLE
    if p_value < FAIL_P:
        return Verdict.FAIL
    if p_value < CAUTION_P:
        return Verdict.CAUTION
    return Verdict.PASS


def _normality(
    check_id: str, label: str, values: FloatArray, *, plural: bool = False
) -> AuditCheck:
    """One Shapiro-Wilk check of normality for ``label``."""
    found = shapiro_wilk(values)
    statistic = None if found.statistic is None else f"W = {num(found.statistic)}"
    if found.status != "ok" or found.p_value is None:
        return assumption(
            check_id,
            f"The normality of {label} could not be checked",
            Verdict.NOT_APPLICABLE,
            found.reason or "Shapiro-Wilk's W test could not be run on these values.",
        )
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: f"{label} {'are' if plural else 'is'} consistent with a normal distribution",
        Verdict.CAUTION: f"{label} {'leave' if plural else 'leaves'} the normal assumption in "
        "doubt",
        Verdict.FAIL: f"{label} {'show' if plural else 'shows'} evidence against a normal "
        "distribution",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the values are far from normal, the p-value and the interval rely on an "
        "approximation; a rank-based comparison does not need normality."
    )
    return assumption(
        check_id,
        titles[verdict],
        verdict,
        f"Shapiro-Wilk's W test gives {statistic}, {p_text(found.p_value)}.",
        statistic=statistic,
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _levene(first: _Sample, second: _Sample) -> AuditCheck:
    """Levene's median-centred test of equal variances, as an assumption check."""
    result = scipy_stats.levene(first.values, second.values, center="median")
    p_value = float(result.pvalue)
    verdict = _verdict(p_value)
    titles = {
        Verdict.PASS: "The two groups have similar spread",
        Verdict.CAUTION: "The two groups' spreads may differ",
        Verdict.FAIL: "The two groups' spreads differ",
    }
    statistic = f"W = {num(float(result.statistic))}"
    next_step = (
        None
        if verdict is Verdict.PASS
        else "Use Welch's t test, which does not assume the two groups have equal variances."
    )
    return assumption(
        "equal_variances",
        titles[verdict],
        verdict,
        f"Levene's test, centred on the median, gives {statistic}, {p_text(p_value)}.",
        statistic=statistic,
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _completeness(excluded: int, used: int, comparable: bool) -> AuditCheck:
    """The check that every pair of the paired test is complete."""
    if not comparable:
        return assumption(
            "complete_pairs",
            "There are no pairs to complete",
            Verdict.NOT_APPLICABLE,
            "The differences were given in one column, so there is no second column to pair.",
        )
    if excluded == 0:
        return assumption(
            "complete_pairs",
            "Every pair is complete",
            Verdict.PASS,
            f"The two columns cover the same rows, and every one of the {used:,} pairs "
            "holds a finite value in both.",
        )
    title = "1 pair is incomplete" if excluded == 1 else f"{excluded:,} pairs are incomplete"
    evidence = (
        "1 row lacks a usable value in one column of its pair and was left out of the test; "
        "it is listed with the result."
        if excluded == 1
        else f"{excluded:,} rows lack a usable value in one column of their pair and were "
        "left out of the test; they are listed with the result."
    )
    return assumption(
        "complete_pairs",
        title,
        Verdict.CAUTION,
        evidence,
        next_step="Check why the pairs are incomplete: a row without a usable partner does "
        "not belong to a paired comparison.",
    )


def _row_count(used: int, excluded: int) -> Counts:
    """The counts a result carries: what was offered, used and left out."""
    return Counts(n_offered=used + excluded, n_used=used, n_excluded=excluded)


def _evidence(p_value: float | None) -> str:
    """What the p-value says about the null hypothesis, as a sentence of evidence."""
    if p_value is None:
        return "The test could not be run."
    if p_value >= CAUTION_P:
        return "The data are consistent with the null hypothesis."
    if p_value >= FAIL_P:
        return "The data show evidence against the null hypothesis."
    return "The data show strong evidence against the null hypothesis."


def _null_sentence(kind: str, subject: str, null: float) -> str:
    """The null hypothesis in words, naming the value it tests against."""
    if kind == "one_sample":
        return f"The mean of {subject} is {num(null)}."
    if kind == "independent":
        if null == 0.0:
            return "The two groups have the same mean."
        return f"The difference of the two means is {num(null)}."
    if null == 0.0:
        return "The mean of the differences is 0."
    return f"The mean of the differences is {num(null)}."


def _hypothesis(
    kind: str,
    subject: str,
    name: str,
    method: str,
    null: float,
    statistic: _Statistic,
    difference: Estimate,
) -> HypothesisTest:
    """One test as every surface shows it, with the difference beside its p-value."""
    return HypothesisTest(
        name=name,
        method=method,
        null_hypothesis=_null_sentence(kind, subject, null),
        statistic=statistic.statistic,
        df=statistic.df,
        p_value=statistic.p_value,
        p_value_text=p_text(statistic.p_value),
        effect=difference,
    )


def _summaries(samples: list[_Sample]) -> list[GroupSummary]:
    """The summary by group, one entry per compared group."""
    return [group_summary(sample.label, sample.values) for sample in samples]


def _one_sample(request: TTestRequest) -> ToolResult:
    """The one-sample t test."""
    pairing = pairs(request.a)
    column = pairing.columns[0]
    sample = _sample(column.name, column.values)
    if sample.n < ONE_SAMPLE_MIN:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the one-sample t test needs at least {ONE_SAMPLE_MIN} values, and the column "
            f"{column.name!r} holds {sample.n}",
        )
    statistic = _one_sample_statistic(sample, request.null)
    difference = _difference(
        f"Mean of {sample.label} less the hypothesised mean",
        sample.mean - request.null,
        statistic.se,
        statistic.df,
        request.level,
    )
    effect = _standardised_effect(
        statistic.statistic,
        statistic.df,
        statistic.se / sample.sd,
        request.level,
        corrected=True,
        label="Hedges' g",
    )
    test = _hypothesis(
        "one_sample",
        sample.label,
        "One-sample t test",
        "One-sample t test, two-sided",
        request.null,
        statistic,
        difference,
    )
    listed, unlisted = list_excluded(pairing.excluded)
    excluded = len(pairing.excluded)
    table = group_table(_summaries([sample]), level=request.level)
    plot = box_plot(f"Box plot of {sample.label}", sample.label, [(sample.label, sample.values)])
    return ToolResult(
        tool="ttest",
        title=f"One-sample t test of {sample.label}",
        level=request.level,
        estimates=(difference, effect),
        tests=(test,),
        assumptions=(_normality("normality", sample.label, sample.values),),
        interpretation=_one_sample_words(sample, request, statistic, difference, effect),
        methods_fragment=_one_sample_methods(sample),
        counts=_row_count(sample.n, excluded),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(table,),
        plots=(plot,),
    )


def _one_sample_words(
    sample: _Sample,
    request: TTestRequest,
    statistic: _Statistic,
    difference: Estimate,
    effect: Estimate,
) -> str:
    """One to three sentences: the estimates, the test and what its p-value says."""
    percent = num(request.level * 100)
    lower, upper = _bounds(difference)
    effect_lower, effect_upper = _bounds(effect)
    first = (
        f"The mean of {sample.label} is {num(sample.mean)}, so the departure from a "
        f"hypothesised mean of {num(request.null)} is {num(difference.value)} with a "
        f"{percent} % interval from {num(lower)} to {num(upper)}."
    )
    second = (
        f"The one-sample t test gives {p_text(statistic.p_value)} "
        f"(t = {num(statistic.statistic)}, df = {num(statistic.df)})."
    )
    third = (
        f"Hedges' g is {num(effect.value)} with a {percent} % interval from "
        f"{num(effect_lower)} to {num(effect_upper)}. {_evidence(statistic.p_value)}"
    )
    return f"{first} {second} {third}"


def _one_sample_methods(sample: _Sample) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        f"A two-sided one-sample t test compares the mean of {sample.label} with the "
        "hypothesised mean; the interval is a Student t interval on n - 1 degrees of "
        "freedom. Hedges' g divides the departure from the hypothesised mean by the sample "
        "standard deviation, with the small-sample correction 1 - 3/(4 df - 1), and its "
        "interval comes from inverting the non-central t distribution. Shapiro-Wilk's W test "
        "checks normality of the values."
    )


def _independent(request: TTestRequest) -> ToolResult:
    """Welch's or Student's two-sample t test."""
    if request.groups is not None:
        samples, excluded_rows, group_name = _group_samples(request)
        title = f"Two-sample t test of {request.a.name} by {group_name}"
    else:
        assert request.b is not None
        # Two independent columns are two samples, not a pairing: each keeps its own rows,
        # so a longer column is not cut down to the shorter one's positions.
        first_values, first_excluded = _finite_column(request.a)
        second_values, second_excluded = _finite_column(request.b)
        samples = [
            _sample(request.a.name, first_values),
            _sample(request.b.name, second_values),
        ]
        excluded_rows = tuple(
            sorted((*first_excluded, *second_excluded), key=lambda row: row.index)
        )
        title = f"Two-sample t test of {request.a.name} and {request.b.name}"
    first, second = samples
    _require_group_size(first)
    _require_group_size(second)
    statistic = _independent_statistic(
        first, second, request.null, equal_variances=request.equal_variances
    )
    pooled_sd = _pooled_sd(first, second)
    difference = _difference(
        f"Difference of means, {first.label} less {second.label}",
        first.mean - second.mean,
        statistic.se,
        statistic.df,
        request.level,
    )
    # The effect standardises the observed difference between the groups, not its distance
    # from a non-zero hypothesised difference: the null moves the test, not the effect size.
    effect = _standardised_effect(
        (first.mean - second.mean) / statistic.se,
        statistic.df,
        statistic.se / pooled_sd,
        request.level,
        corrected=True,
        label="Hedges' g",
    )
    if request.equal_variances:
        name = "Student's t test"
        method = "Student's t test, two-sided"
    else:
        name = "Welch's t test"
        method = "Welch's t test, two-sided"
    test = _hypothesis("independent", name, name, method, request.null, statistic, difference)
    assumptions: tuple[AuditCheck, ...] = (
        _normality("normality_first", first.label, first.values),
        _normality("normality_second", second.label, second.values),
    )
    if request.equal_variances:
        assumptions = (*assumptions, _levene(first, second))
    listed, unlisted = list_excluded(excluded_rows)
    return ToolResult(
        tool="ttest",
        title=title,
        level=request.level,
        estimates=(difference, effect),
        tests=(test,),
        assumptions=assumptions,
        interpretation=_independent_words(
            first, second, request, statistic, difference, effect, name
        ),
        methods_fragment=_independent_methods(first, second, request.equal_variances, name),
        counts=_row_count(first.n + second.n, len(excluded_rows)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(group_table(_summaries(samples), level=request.level),),
        plots=(
            box_plot(
                f"Box plot of {first.label} and {second.label}",
                request.a.name,
                [(sample.label, sample.values) for sample in samples],
            ),
        ),
    )


def _group_samples(
    request: TTestRequest,
) -> tuple[list[_Sample], tuple[ExcludedRow, ...], str]:
    """The two samples of an independent test given as a value column and a group column.

    Raises:
        ToolError: ``LEVELS_REQUIRED`` unless the group column holds exactly two levels.
    """
    assert request.groups is not None
    labels = _two_levels(request.groups)
    pairing = aligned(request.a, request.groups)
    values = pairing.values.values
    found = pairing.groups[0].labels
    samples = [_sample(label, values[found == label]) for label in labels]
    return samples, pairing.excluded, request.groups.name


def _two_levels(groups: GroupColumn) -> tuple[str, str]:
    """The group column's two labels, in order of first appearance.

    Raises:
        ToolError: ``LEVELS_REQUIRED`` unless there are exactly two.
    """
    found, first = np.unique(groups.labels, return_index=True)
    ordered = found[np.argsort(first)]
    if ordered.size != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the group column {groups.name!r} holds {ordered.size} "
            f"{'label' if ordered.size == 1 else 'labels'}, and this test compares exactly "
            "two",
        )
    return str(ordered[0]), str(ordered[1])


def _pooled_sd(first: _Sample, second: _Sample) -> float:
    """The pooled sample standard deviation of the two groups."""
    degrees = first.n + second.n - 2
    variance = ((first.n - 1) * first.sd**2 + (second.n - 1) * second.sd**2) / degrees
    return math.sqrt(variance)


def _independent_words(
    first: _Sample,
    second: _Sample,
    request: TTestRequest,
    statistic: _Statistic,
    difference: Estimate,
    effect: Estimate,
    name: str,
) -> str:
    """One to three sentences for the two-sample test."""
    percent = num(request.level * 100)
    lower, upper = _bounds(difference)
    effect_lower, effect_upper = _bounds(effect)
    first_sentence = (
        f"The mean of {first.label} is {num(first.mean)} and of {second.label} is "
        f"{num(second.mean)}, so the difference is {num(difference.value)} with a "
        f"{percent} % interval from {num(lower)} to {num(upper)}."
    )
    against = "against no difference" if request.null == 0.0 else f"against {num(request.null)}"
    second_sentence = (
        f"{name} {against} gives {p_text(statistic.p_value)} "
        f"(t = {num(statistic.statistic)}, df = {num(statistic.df)})."
    )
    third = (
        f"Hedges' g is {num(effect.value)} with a {percent} % interval from "
        f"{num(effect_lower)} to {num(effect_upper)}. {_evidence(statistic.p_value)}"
    )
    return f"{first_sentence} {second_sentence} {third}"


def _independent_methods(first: _Sample, second: _Sample, equal_variances: bool, name: str) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    if equal_variances:
        degrees = "n1 + n2 - 2"
        check = (
            "Shapiro-Wilk's W test checks normality of each group, and Levene's test "
            "centred on the median checks that the two groups have equal variances."
        )
    else:
        degrees = "Welch-Satterthwaite"
        check = "Shapiro-Wilk's W test checks normality of each group."
    return (
        f"A two-sided {name} compares the means of {first.label} and {second.label}; the "
        f"interval is a Student t interval on {degrees} degrees of freedom. Hedges' g "
        "divides the difference of the means by the pooled standard deviation, with the "
        "small-sample correction 1 - 3/(4 df - 1), and its interval comes from inverting the "
        f"non-central t distribution. {check}"
    )


def _paired(request: TTestRequest) -> ToolResult:
    """The paired t test, on two columns or on the differences in one."""
    if request.b is not None:
        pairing = pairs(request.a, request.b)
        left, right = pairing.columns
        excluded_rows = pairing.excluded
        differences = left.values - right.values
        label = f"{left.name} less {right.name}"
        difference_label = f"Mean of the differences, {left.name} less {right.name}"
        samples = [
            _sample(left.name, left.values),
            _sample(right.name, right.values),
        ]
        comparable = True
        title = f"Paired t test of {left.name} and {right.name}"
        y_label = left.name
    else:
        pairing = pairs(request.a)
        differences = pairing.columns[0].values
        excluded_rows = pairing.excluded
        label = f"Differences in {request.a.name}"
        difference_label = f"Mean of the differences in {request.a.name}"
        samples = [_sample(label, differences)]
        comparable = False
        title = f"Paired t test of the differences in {request.a.name}"
        y_label = label
    sample = _sample(label, differences)
    if sample.n < PAIRED_MIN:
        unit = "complete pairs" if comparable else "differences"
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the paired t test needs at least {PAIRED_MIN} {unit}, and it has {sample.n}",
        )
    statistic = _paired_statistic(sample, request.null)
    difference = _difference(
        difference_label, sample.mean, statistic.se, statistic.df, request.level
    )
    # Cohen's dz standardises the observed mean difference, not its distance from a
    # non-zero hypothesised difference.
    effect = _standardised_effect(
        sample.mean / statistic.se,
        statistic.df,
        statistic.se / sample.sd,
        request.level,
        corrected=False,
        label="Cohen's dz",
    )
    test = _hypothesis(
        "paired",
        "Paired t test",
        "Paired t test",
        "Paired t test, two-sided",
        request.null,
        statistic,
        difference,
    )
    listed, unlisted = list_excluded(excluded_rows)
    return ToolResult(
        tool="ttest",
        title=title,
        level=request.level,
        estimates=(difference, effect),
        tests=(test,),
        assumptions=(
            _normality("normality", "the differences", sample.values, plural=True),
            _completeness(len(excluded_rows), sample.n, comparable),
        ),
        interpretation=_paired_words(sample, request, statistic, difference, effect, comparable),
        methods_fragment=_paired_methods(comparable),
        counts=_row_count(sample.n, len(excluded_rows)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(group_table(_summaries(samples), level=request.level),),
        plots=(
            box_plot(
                title,
                y_label,
                [(entry.label, entry.values) for entry in samples]
                if comparable
                else [(sample.label, sample.values)],
            ),
        ),
    )


def _paired_words(
    sample: _Sample,
    request: TTestRequest,
    statistic: _Statistic,
    difference: Estimate,
    effect: Estimate,
    comparable: bool,
) -> str:
    """One to three sentences for the paired test."""
    percent = num(request.level * 100)
    subjects = "complete pairs" if comparable else "differences"
    against = "against no difference" if request.null == 0.0 else f"against {num(request.null)}"
    lower, upper = _bounds(difference)
    effect_lower, effect_upper = _bounds(effect)
    first = (
        f"The mean difference is {num(difference.value)} with a {percent} % interval from "
        f"{num(lower)} to {num(upper)}."
    )
    second = (
        f"The paired t test on {sample.n:,} {subjects} {against} gives "
        f"{p_text(statistic.p_value)} (t = {num(statistic.statistic)}, "
        f"df = {num(statistic.df)})."
    )
    third = (
        f"Cohen's dz is {num(effect.value)} with a {percent} % interval from "
        f"{num(effect_lower)} to {num(effect_upper)}. {_evidence(statistic.p_value)}"
    )
    return f"{first} {second} {third}"


def _paired_methods(comparable: bool) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    pairs_clause = "runs on the complete pairs" if comparable else "runs on the differences"
    return (
        f"A two-sided paired t test {pairs_clause}; the interval is a Student t interval on "
        "n - 1 degrees of freedom. Cohen's dz divides the mean difference by the standard "
        "deviation of the differences, and its interval comes from inverting the non-central "
        "t distribution. Shapiro-Wilk's W test checks normality of the differences."
    )


TOOLS.register(
    ToolSpec(
        name="ttest",
        family="means",
        title="t tests",
        request=TTestRequest,
        run=run_ttest,
    )
)
