"""One-way analysis of variance, its Welch counterpart, and the Kruskal-Wallis test.

The analysis of variance asks whether several groups share one mean. It splits the total
variation of the values into the part between the groups and the part within them, and its
F statistic is the ratio of the two mean squares; eta squared and omega squared say how much
of the variation the groups account for. Welch's analysis of variance does not assume the
groups have equal variances: it weights each group by its own variance, both in the statistic
and in its denominator degrees of freedom. Tukey's honestly significant difference test or
Dunnett's test against a control follows when there are more than two groups, and each of
their p-values is already adjusted for the family.

The Kruskal-Wallis test asks the same question of the groups' mean ranks, with tied values
taking their average rank and a correction for the ties; Dunn's test compares the pairs and
Holm's method adjusts its p-values. A rank test makes no assumption of normality, and whether
the groups share a shape is not something this tool tests: the box plot is what a person reads
for it.

A p-value is a statement about evidence, never a statement of proof: a small one is evidence
against the null hypothesis, and a large one is not evidence that the null is true. Every
assumption the tests make is checked where the data can show it: Shapiro-Wilk for the
normality of each group and Levene's median-centred test for equal variances.
"""

import math
import warnings
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.diagnostics import shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import num, p_threshold
from curvestudio.tools.stats._base import (
    Counts,
    Estimate,
    HypothesisTest,
    ToolResult,
    aligned,
    assumption,
    check_level,
    levels,
    list_excluded,
    p_text,
)
from curvestudio.tools.stats._display import Cell, ColumnKind, FloatArray, box_plot
from curvestudio.tools.stats._tables import (
    group_summary,
    group_table,
    holm,
    table,
)
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

GROUP_MIN = 2
"""The fewest values a group takes: its spread needs two, and so does its mean rank."""

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

_POST_HOC_LABEL = "Post-hoc test"

DUNNETT_SEED = 20261124
"""The fixed generator seed of Dunnett's p-value, so the tool stays a pure function.

SciPy integrates Dunnett's distribution by randomized quasi-Monte Carlo: an unfixed generator
would make the p-value move a little from one call to the next. The seed is fixed here and
the same seed reproduces the reference implementation's number exactly.
"""

ANOVA_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("source", "Source", "text"),
    ("ss", "SS", "number"),
    ("df", "df", "number"),
    ("ms", "MS", "number"),
    ("f", "F", "number"),
    ("p", "p", "p_value"),
)
"""The columns of the analysis-of-variance table, in the order a person reads them."""

RANK_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("group", "Group", "text"),
    ("n", "n", "int"),
    ("mean_rank", "Mean rank", "number"),
    ("median", "Median", "number"),
    ("q1", "Q1", "number"),
    ("q3", "Q3", "number"),
)
"""The columns of the ranks-by-group table, in the order a person reads them."""


class AnovaRequest(BaseModel):
    """One one-way analysis of variance: the values, their groups and the post-hoc test."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column = Field(
        description="The numeric column whose mean is compared across the groups."
    )
    groups: GroupColumn = Field(
        description=(
            "A group column; each of its levels is a group, and at least two levels are needed."
        )
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )
    post_hoc: Literal["tukey", "dunnett"] = Field(
        default="tukey",
        description=(
            "The comparison of the group means when there are more than two groups: Tukey's "
            "honestly significant difference test for every pair, or Dunnett's test of every "
            "group against a control."
        ),
        json_schema_extra={"label": _POST_HOC_LABEL},
    )
    control: str | None = Field(
        default=None,
        description=(
            "For Dunnett's test, the level of the group column that is the control; it must be "
            "one of the levels. Leave empty for Tukey's test."
        ),
    )


class KruskalRequest(BaseModel):
    """One Kruskal-Wallis test: the values, their groups and a confidence level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column = Field(
        description="The numeric column whose mean rank is compared across the groups."
    )
    groups: GroupColumn = Field(
        description=(
            "A group column; each of its levels is a group, and at least two levels are needed."
        )
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Group(NamedTuple):
    """One group's finite values and the few numbers every form of the test needs."""

    label: str
    values: FloatArray
    n: int
    mean: float
    sd: float


class _AnovaSums(NamedTuple):
    """The sums of squares of one-way analysis of variance and the F test built on them."""

    between: float
    within: float
    total: float
    df_between: float
    df_within: float
    df_total: float
    ms_between: float
    ms_within: float
    statistic: float
    p_value: float


class _FStatistic(NamedTuple):
    """An F test: its statistic, both degrees of freedom and its p-value."""

    statistic: float
    df: float
    df_denominator: float
    p_value: float


class _HStatistic(NamedTuple):
    """The Kruskal-Wallis statistic, its degrees of freedom and its p-value."""

    statistic: float
    df: float
    p_value: float


def run_anova(request: BaseModel) -> ToolResult:
    """Run the one-way analysis of variance on its own request."""
    if not isinstance(request, AnovaRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"anova takes an AnovaRequest, not a request of {type(request).__name__}",
        )
    return _analyse_anova(request)


def run_kruskal(request: BaseModel) -> ToolResult:
    """Run the Kruskal-Wallis test on its own request."""
    if not isinstance(request, KruskalRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"kruskal takes a KruskalRequest, not a request of {type(request).__name__}",
        )
    return _analyse_kruskal(request)


def _verdict(p_value: float | None) -> Verdict:
    """The verdict a p-value gives an assumption."""
    if p_value is None:
        return Verdict.NOT_APPLICABLE
    if p_value < FAIL_P:
        return Verdict.FAIL
    if p_value < CAUTION_P:
        return Verdict.CAUTION
    return Verdict.PASS


def _summarise(label: str, values: FloatArray) -> _Group:
    """One group reduced to its finite values, with its mean and sample standard deviation."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return _Group(label=str(label), values=finite, n=0, mean=0.0, sd=0.0)
    sd = float(np.std(finite, ddof=1)) if count >= 2 else 0.0
    return _Group(label=str(label), values=finite, n=count, mean=float(np.mean(finite)), sd=sd)


def _samples(values: FloatArray, found: np.ndarray, labels: Sequence[str]) -> tuple[_Group, ...]:
    """One sample per label, sliced from the aligned values and labels."""
    return tuple(_summarise(label, values[found == label]) for label in labels)


def _require_groups(groups: Sequence[_Group]) -> None:
    """Refuse a group too small for its spread to be estimated.

    Raises:
        ToolError: ``GROUP_OF_ONE`` for a group of one, ``TOO_FEW_ROWS`` for an empty one.
    """
    for group in groups:
        if group.n >= GROUP_MIN:
            continue
        if group.n == 1:
            raise ToolError(
                "GROUP_OF_ONE",
                f"the group {group.label!r} holds 1 value, so its spread cannot be estimated",
            )
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the group {group.label!r} holds no usable value, and each group needs at least "
            f"{GROUP_MIN}",
        )


def _constant_group(groups: Sequence[_Group]) -> _Group | None:
    """The first group whose values do not vary, or ``None`` when every group does."""
    return next((group for group in groups if group.sd == 0.0), None)


def _refuse_constant(groups: Sequence[_Group]) -> None:
    """Refuse a group whose values do not vary, so Welch's weights would be infinite.

    Raises:
        ToolError: ``CONSTANT_COLUMN`` naming the group.
    """
    constant = _constant_group(groups)
    if constant is not None:
        raise ToolError(
            "CONSTANT_COLUMN",
            f"the group {constant.label!r} holds one value repeated, so the analysis of "
            "variance and Welch's weights are undefined",
        )


def _require_control(request: AnovaRequest, labels: Sequence[str]) -> None:
    """Refuse Dunnett's test without a control that is one of the group column's levels.

    Raises:
        ToolError: ``LEVELS_REQUIRED`` naming the labels when the control is missing or is not
            one of them; ``INVALID_OPTION`` when a control is given for Tukey's test.
    """
    if request.post_hoc == "tukey":
        if request.control is not None:
            raise ToolError(
                "INVALID_OPTION",
                "Tukey's test compares every pair and takes no control group; choose "
                "Dunnett's test to compare the groups against a control",
            )
        return
    if request.control is None:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"Dunnett's test needs one of the levels {', '.join(repr(label) for label in labels)} "
            "as its control, and none was given",
        )
    if request.control not in labels:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the control {request.control!r} is not one of the group column's levels "
            f"{', '.join(repr(label) for label in labels)}",
        )


def _anova_sums(groups: Sequence[_Group]) -> _AnovaSums:
    """The sums of squares of one-way analysis of variance and the classical F test.

    Raises:
        ZeroDivisionError: the within-groups sum of squares is zero; the caller refuses a
            constant group first.
    """
    combined = np.concatenate([group.values for group in groups])
    grand = float(np.mean(combined))
    between = sum(group.n * (group.mean - grand) ** 2 for group in groups)
    within = sum(float(np.sum((group.values - group.mean) ** 2)) for group in groups)
    total = between + within
    count = int(combined.size)
    group_count = len(groups)
    df_between = float(group_count - 1)
    df_within = float(count - group_count)
    df_total = float(count - 1)
    ms_between = between / df_between
    ms_within = within / df_within
    statistic = ms_between / ms_within
    return _AnovaSums(
        between=between,
        within=within,
        total=total,
        df_between=df_between,
        df_within=df_within,
        df_total=df_total,
        ms_between=ms_between,
        ms_within=ms_within,
        statistic=statistic,
        p_value=float(scipy_stats.f.sf(statistic, df_between, df_within)),
    )


def _classical_f(groups: Sequence[_Group]) -> _FStatistic:
    """The classical one-way F statistic on the groups."""
    sums = _anova_sums(groups)
    return _FStatistic(
        statistic=sums.statistic,
        df=sums.df_between,
        df_denominator=sums.df_within,
        p_value=sums.p_value,
    )


def _welch_f(groups: Sequence[_Group]) -> _FStatistic:
    """Welch's analysis of variance: each group weighted by its own variance.

    The statistic divides a variance-weighted spread of the group means by the correction
    that carries Welch's denominator degrees of freedom.

    Raises:
        ZeroDivisionError: every group holds one value, so no weight is defined; the caller
            refuses such a group first.
    """
    group_count = len(groups)
    sizes = np.array([group.n for group in groups], dtype=np.float64)
    means = np.array([group.mean for group in groups], dtype=np.float64)
    variances = np.array([group.sd**2 for group in groups], dtype=np.float64)
    weights = sizes / variances
    total_weight = float(weights.sum())
    weighted_mean = float((weights * means).sum() / total_weight)
    spread = float((weights * (means - weighted_mean) ** 2).sum() / (group_count - 1))
    terms = float(((1.0 - weights / total_weight) ** 2 / (sizes - 1.0)).sum())
    correction = 1.0 + 2.0 * (group_count - 2) / (group_count**2 - 1) * terms
    statistic = spread / correction
    df_denominator = (group_count**2 - 1) / (3.0 * terms)
    return _FStatistic(
        statistic=statistic,
        df=float(group_count - 1),
        df_denominator=df_denominator,
        p_value=float(scipy_stats.f.sf(statistic, float(group_count - 1), df_denominator)),
    )


def _normality(check_id: str, label: str, values: FloatArray) -> AuditCheck:
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
        Verdict.PASS: f"{label} is consistent with a normal distribution",
        Verdict.CAUTION: f"{label} leaves the normal assumption in doubt",
        Verdict.FAIL: f"{label} shows evidence against a normal distribution",
    }
    next_step = (
        None
        if verdict is Verdict.PASS
        else "If the values are far from normal, the p-value relies on an approximation; a "
        "rank-based comparison does not need normality."
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


def _levene_test(groups: Sequence[_Group]) -> tuple[float, float]:
    """Levene's median-centred statistic and p-value for the groups.

    A group whose absolute deviations from its median are all equal leaves the statistic's
    denominator zero; SciPy then divides by zero, so the warning is kept out of the caller's
    way and the check reads the non-finite result as not applicable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = scipy_stats.levene(*(group.values for group in groups), center="median")
    return float(result.statistic), float(result.pvalue)


def _levene(groups: Sequence[_Group]) -> AuditCheck:
    """Levene's median-centred test of equal variances, as an assumption check."""
    statistic_value, p_value = _levene_test(groups)
    if not math.isfinite(statistic_value) or not math.isfinite(p_value):
        return assumption(
            "equal_variances",
            "The equality of the groups' variances could not be checked",
            Verdict.NOT_APPLICABLE,
            "Levene's test, centred on the median, gave no usable statistic on these groups.",
        )
    verdict = _verdict(p_value)
    titles = {
        Verdict.PASS: "The groups have similar spread",
        Verdict.CAUTION: "The groups' spreads may differ",
        Verdict.FAIL: "The groups' spreads differ",
    }
    statistic = f"W = {num(statistic_value)}"
    next_step = (
        None
        if verdict is Verdict.PASS
        else "Read Welch's analysis of variance, which does not assume the groups have equal "
        "variances."
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


def _tukey_scale(ms_within: float, left: _Group, right: _Group) -> float:
    """The studentized range's scale for the comparison of two groups."""
    return math.sqrt(ms_within / 2.0 * (1.0 / left.n + 1.0 / right.n))


def _tukey_p_values(
    groups: Sequence[_Group], ms_within: float, df_within: float
) -> tuple[float, ...]:
    """The studentized-range p-value of every Tukey comparison, in pair order."""
    group_count = len(groups)
    p_values: list[float] = []
    for first in range(group_count):
        for second in range(first + 1, group_count):
            left, right = groups[first], groups[second]
            statistic = abs(left.mean - right.mean) / _tukey_scale(ms_within, left, right)
            p_values.append(
                float(scipy_stats.studentized_range.sf(statistic, group_count, df_within))
            )
    return tuple(p_values)


def _tukey(
    groups: Sequence[_Group], ms_within: float, df_within: float, level: float
) -> tuple[HypothesisTest, ...]:
    """One Tukey HSD comparison per pair of groups, adjusted by the studentized range."""
    group_count = len(groups)
    critical = float(scipy_stats.studentized_range.ppf(level, group_count, df_within))
    tests: list[HypothesisTest] = []
    for first in range(group_count):
        for second in range(first + 1, group_count):
            left, right = groups[first], groups[second]
            difference = left.mean - right.mean
            scale = _tukey_scale(ms_within, left, right)
            statistic = abs(difference) / scale
            p_value = float(scipy_stats.studentized_range.sf(statistic, group_count, df_within))
            effect = Estimate(
                label=f"Difference of means, {left.label} less {right.label}",
                value=difference,
                lower=difference - critical * scale,
                upper=difference + critical * scale,
            )
            tests.append(
                HypothesisTest(
                    name=f"Tukey HSD, {left.label} less {right.label}",
                    method="Tukey's honestly significant difference test, family-adjusted",
                    null_hypothesis=f"The means of {left.label} and {right.label} are equal.",
                    statistic=statistic,
                    df=float(group_count),
                    df_denominator=df_within,
                    p_value=p_value,
                    p_value_text=p_text(p_value),
                    effect=effect,
                    adjusted_p=p_value,
                    adjustment="Tukey HSD",
                )
            )
    return tuple(tests)


def _dunnett(
    groups: Sequence[_Group],
    control: str,
    df_within: float,
    level: float,
) -> tuple[HypothesisTest, ...]:
    """One Dunnett comparison per treatment against the control, adjusted for the family."""
    control_group = next(group for group in groups if group.label == control)
    treatments = [group for group in groups if group.label != control]
    result = scipy_stats.dunnett(
        *(group.values for group in treatments),
        control=control_group.values,
        rng=np.random.default_rng(DUNNETT_SEED),
    )
    statistics = np.atleast_1d(result.statistic)
    p_values = np.atleast_1d(result.pvalue)
    lower, upper = result.confidence_interval(confidence_level=level)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    tests: list[HypothesisTest] = []
    for index, treatment in enumerate(treatments):
        difference = treatment.mean - control_group.mean
        p_value = float(p_values[index])
        effect = Estimate(
            label=f"Difference of means, {treatment.label} less {control}",
            value=difference,
            lower=float(lower[index]),
            upper=float(upper[index]),
        )
        tests.append(
            HypothesisTest(
                name=f"Dunnett's test, {treatment.label} against {control}",
                method="Dunnett's test against the control group, family-adjusted",
                null_hypothesis=(f"The mean of {treatment.label} equals the mean of {control}."),
                statistic=float(statistics[index]),
                df=df_within,
                p_value=p_value,
                p_value_text=p_text(p_value),
                effect=effect,
                adjusted_p=p_value,
                adjustment="Dunnett",
            )
        )
    return tuple(tests)


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


def _comparison_words(request: AnovaRequest, comparisons: Sequence[HypothesisTest]) -> str:
    """What the post-hoc comparisons found, as a sentence of evidence."""
    rejected = sum(
        1 for test in comparisons if test.adjusted_p is not None and test.adjusted_p < CAUTION_P
    )
    smallest = min(test.adjusted_p for test in comparisons if test.adjusted_p is not None)
    if request.post_hoc == "tukey":
        name = "Tukey HSD"
        subject = "comparisons"
    else:
        name = "Dunnett's test"
        subject = "comparisons against the control"
    verb = "has" if rejected == 1 else "have"
    return (
        f"{name} adjusts the {len(comparisons)} {subject} for the family: {rejected} of them "
        f"{verb} an adjusted p below {num(CAUTION_P)}, the smallest {p_text(smallest)}."
    )


def _anova_words(
    request: AnovaRequest,
    sums: _AnovaSums,
    welch: _FStatistic,
    eta: Estimate,
    omega: Estimate,
    comparisons: Sequence[HypothesisTest],
) -> str:
    """One to three sentences: the statistics, the effect sizes and what they say."""
    first = (
        f"The one-way analysis of variance gives F = {num(sums.statistic)} on "
        f"{num(sums.df_between)} and {num(sums.df_within)} degrees of freedom, "
        f"{p_text(sums.p_value)}; eta squared is {num(eta.value)} and omega squared is "
        f"{num(omega.value)}."
    )
    second = (
        f"Welch's analysis of variance gives F = {num(welch.statistic)} on "
        f"{num(welch.df)} and {num(welch.df_denominator)} degrees of freedom, "
        f"{p_text(welch.p_value)}."
    )
    third = _comparison_words(request, comparisons) if comparisons else _evidence(sums.p_value)
    return f"{first} {second} {third}"


def _anova_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "One-way analysis of variance compares the group means by the ratio of the "
        "between-groups mean square to the within-groups mean square, with k - 1 and N - k "
        "degrees of freedom. Eta squared is the between-groups sum of squares divided by the "
        "total sum of squares, and omega squared corrects it for the number of groups. Welch's "
        "analysis of variance weights each group by its own variance and computes its "
        "denominator degrees of freedom from those weights. Shapiro-Wilk's W test checks "
        "normality of each group, and Levene's test centred on the median checks equal "
        "variances. Tukey's honestly significant difference test or Dunnett's test against the "
        "control adjusts the pairwise p-values for the family."
    )


def _anova_table(sums: _AnovaSums) -> tuple[tuple[Cell, ...], ...]:
    """The analysis-of-variance table's rows, from the sums of squares."""
    return (
        (
            "Between groups",
            sums.between,
            sums.df_between,
            sums.ms_between,
            sums.statistic,
            sums.p_value,
        ),
        ("Within groups", sums.within, sums.df_within, sums.ms_within, None, None),
        ("Total", sums.total, sums.df_total, None, None, None),
    )


def _analyse_anova(request: AnovaRequest) -> ToolResult:
    """The analysis of variance itself, once the request is known to be this tool's own."""
    check_level(request.level)
    labels = levels(request.groups, minimum=2)
    _require_control(request, labels)
    pairing = aligned(request.values, request.groups)
    groups = _samples(pairing.values.values, pairing.groups[0].labels, labels)
    _require_groups(groups)
    _refuse_constant(groups)
    sums = _anova_sums(groups)
    welch = _welch_f(groups)
    eta = Estimate(label="Eta squared", value=sums.between / sums.total)
    omega = Estimate(
        label="Omega squared",
        value=(sums.between - sums.df_between * sums.ms_within) / (sums.total + sums.ms_within),
    )
    comparisons: tuple[HypothesisTest, ...] = ()
    if len(groups) > 2:
        if request.post_hoc == "dunnett":
            assert request.control is not None
            comparisons = _dunnett(groups, request.control, sums.df_within, request.level)
        else:
            comparisons = _tukey(groups, sums.ms_within, sums.df_within, request.level)
    tests = (
        HypothesisTest(
            name="One-way analysis of variance",
            method="One-way analysis of variance, F test",
            null_hypothesis="The means of every group are equal.",
            statistic=sums.statistic,
            df=sums.df_between,
            df_denominator=sums.df_within,
            p_value=sums.p_value,
            p_value_text=p_text(sums.p_value),
            effect=eta,
        ),
        HypothesisTest(
            name="Welch's analysis of variance",
            method="Welch's analysis of variance, F test",
            null_hypothesis="The means of every group are equal.",
            statistic=welch.statistic,
            df=welch.df,
            df_denominator=welch.df_denominator,
            p_value=welch.p_value,
            p_value_text=p_text(welch.p_value),
            effect=omega,
        ),
        *comparisons,
    )
    assumptions = (
        *(
            _normality(f"normality_{index + 1}", group.label, group.values)
            for index, group in enumerate(groups)
        ),
        _levene(groups),
    )
    listed, unlisted = list_excluded(pairing.excluded)
    used = sum(group.n for group in groups)
    summaries = [group_summary(group.label, group.values) for group in groups]
    return ToolResult(
        tool="anova",
        title=f"One-way analysis of variance of {request.values.name} by {request.groups.name}",
        level=request.level,
        estimates=(eta, omega),
        tests=tests,
        assumptions=assumptions,
        interpretation=_anova_words(request, sums, welch, eta, omega, comparisons),
        methods_fragment=_anova_methods(),
        counts=_row_count(used, len(pairing.excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Analysis of variance",
                ANOVA_COLUMNS,
                _anova_table(sums),
                note=(
                    "SS is the sum of squares, MS the mean square and df the degrees of "
                    "freedom. The F statistic is the between-groups mean square divided by "
                    "the within-groups mean square, on k - 1 and N - k degrees of freedom. "
                    "The total row is the corrected total."
                ),
            ),
            group_table(summaries, level=request.level),
        ),
        plots=(
            box_plot(
                f"Box plot of {request.values.name} by {request.groups.name}",
                request.values.name,
                [(group.label, group.values) for group in groups],
            ),
        ),
    )


def _kruskal_h(groups: Sequence[_Group]) -> _HStatistic:
    """The Kruskal-Wallis statistic with its tie correction, on the groups' mean ranks."""
    combined = np.concatenate([group.values for group in groups])
    ranks = scipy_stats.rankdata(combined)
    count = int(combined.size)
    group_count = len(groups)
    total = 0.0
    start = 0
    for group in groups:
        chunk = ranks[start : start + group.n]
        total += float(chunk.sum()) ** 2 / group.n
        start += group.n
    statistic = 12.0 / (count * (count + 1)) * total - 3.0 * (count + 1)
    _, tied = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(tied.astype(np.float64) ** 3 - tied))
    correction = 1.0 - tie_sum / (count**3 - count)
    if correction <= 0.0:
        return _HStatistic(statistic=0.0, df=float(group_count - 1), p_value=1.0)
    statistic = statistic / correction
    return _HStatistic(
        statistic=statistic,
        df=float(group_count - 1),
        p_value=float(scipy_stats.chi2.sf(statistic, group_count - 1)),
    )


def _mean_ranks(groups: Sequence[_Group]) -> tuple[float, ...]:
    """The mean rank of each group, over the values of all of them ranked together."""
    combined = np.concatenate([group.values for group in groups])
    ranks = scipy_stats.rankdata(combined)
    means: list[float] = []
    start = 0
    for group in groups:
        means.append(float(ranks[start : start + group.n].mean()))
        start += group.n
    return tuple(means)


def _rank_sigma(groups: Sequence[_Group]) -> float:
    """The variance of a difference of mean ranks, before the group sizes are applied."""
    combined = np.concatenate([group.values for group in groups])
    count = int(combined.size)
    _, tied = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(tied.astype(np.float64) ** 3 - tied))
    return count * (count + 1) / 12.0 - tie_sum / (12.0 * (count - 1))


def _dunn(groups: Sequence[_Group], level: float) -> tuple[HypothesisTest, ...]:
    """One Dunn comparison per pair of groups, with Holm's adjustment on the p-values."""
    means = _mean_ranks(groups)
    sigma = _rank_sigma(groups)
    critical = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    tests: list[HypothesisTest] = []
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            left, right = groups[first], groups[second]
            difference = means[first] - means[second]
            scale = math.sqrt(sigma * (1.0 / left.n + 1.0 / right.n))
            statistic = 0.0 if scale == 0.0 else difference / scale
            p_value = float(2.0 * scipy_stats.norm.sf(abs(statistic)))
            effect = Estimate(
                label=f"Difference of mean ranks, {left.label} less {right.label}",
                value=difference,
                se=scale,
                lower=difference - critical * scale,
                upper=difference + critical * scale,
            )
            tests.append(
                HypothesisTest(
                    name=f"Dunn's test, {left.label} less {right.label}",
                    method="Dunn's test, two-sided, with Holm's adjustment",
                    null_hypothesis=(
                        f"The mean ranks of {left.label} and {right.label} are equal."
                    ),
                    statistic=statistic,
                    p_value=p_value,
                    p_value_text=p_text(p_value),
                    effect=effect,
                )
            )
    return holm(tests)


def _kruskal_words(
    kruskal: _HStatistic,
    epsilon: Estimate,
    groups: Sequence[_Group],
    comparisons: Sequence[HypothesisTest],
) -> str:
    """One to three sentences for the Kruskal-Wallis test."""
    ranks = _mean_ranks(groups)
    lowest = min(range(len(groups)), key=lambda index: ranks[index])
    highest = max(range(len(groups)), key=lambda index: ranks[index])
    first = (
        f"The Kruskal–Wallis test gives H = {num(kruskal.statistic)} on "  # noqa: RUF001
        f"{num(kruskal.df)} degrees of freedom, {p_text(kruskal.p_value)}; epsilon squared is "
        f"{num(epsilon.value)}."
    )
    second = (
        f"The mean ranks run from {num(ranks[lowest])} for {groups[lowest].label} to "
        f"{num(ranks[highest])} for {groups[highest].label}."
    )
    if comparisons:
        rejected = sum(
            1 for test in comparisons if test.adjusted_p is not None and test.adjusted_p < CAUTION_P
        )
        smallest = min(test.adjusted_p for test in comparisons if test.adjusted_p is not None)
        verb = "has" if rejected == 1 else "have"
        third = (
            f"Dunn's tests adjust the {len(comparisons)} comparisons by Holm's method: "
            f"{rejected} of them {verb} an adjusted p below {num(CAUTION_P)}, the smallest "
            f"{p_text(smallest)}."
        )
    else:
        third = _evidence(kruskal.p_value)
    return f"{first} {second} {third}"


def _kruskal_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "The Kruskal–Wallis test ranks all values together, with tied values taking their "  # noqa: RUF001
        "average rank, and compares the groups' mean ranks; its statistic has a chi-square "
        "distribution on k - 1 degrees of freedom, with a correction for ties. Epsilon squared "
        "is H divided by N - 1. Dunn's test compares each pair's mean ranks with a normal "
        "approximation, and Holm's method adjusts its p-values. Whether the groups share a "
        "shape is not tested here; the box plot shows it."
    )


def _analyse_kruskal(request: KruskalRequest) -> ToolResult:
    """The Kruskal-Wallis test itself, once the request is known to be this tool's own."""
    check_level(request.level)
    labels = levels(request.groups, minimum=2)
    pairing = aligned(request.values, request.groups)
    groups = _samples(pairing.values.values, pairing.groups[0].labels, labels)
    _require_groups(groups)
    kruskal = _kruskal_h(groups)
    combined = np.concatenate([group.values for group in groups])
    epsilon = Estimate(label="Epsilon squared", value=kruskal.statistic / (combined.size - 1))
    comparisons: tuple[HypothesisTest, ...] = ()
    if len(groups) > 2:
        comparisons = _dunn(groups, request.level)
    test = HypothesisTest(
        name="Kruskal–Wallis test",  # noqa: RUF001
        method="Kruskal–Wallis test, chi-square approximation with ties",  # noqa: RUF001
        null_hypothesis="The groups' mean ranks are equal.",
        statistic=kruskal.statistic,
        df=kruskal.df,
        p_value=kruskal.p_value,
        p_value_text=p_text(kruskal.p_value),
        effect=epsilon,
    )
    ranks = _mean_ranks(groups)
    summaries = [group_summary(group.label, group.values) for group in groups]
    granted = assumption(
        "equal_shapes",
        "Whether the groups share a shape cannot be tested here",
        Verdict.NOT_APPLICABLE,
        "The Kruskal–Wallis test compares mean ranks, and its reading as a shift in location "  # noqa: RUF001
        "holds when the groups have similar shapes; this tool does not test that. Look at the "
        "box plot to compare the groups' spreads and skew.",
        next_step=(
            "Look at the box plot; if the groups' spreads or skew differ, report the test as "
            "a comparison of ranks rather than of medians."
        ),
    )
    listed, unlisted = list_excluded(pairing.excluded)
    used = sum(group.n for group in groups)
    return ToolResult(
        tool="kruskal",
        title=f"Kruskal–Wallis test of {request.values.name} by {request.groups.name}",  # noqa: RUF001
        level=request.level,
        estimates=(epsilon,),
        tests=(test, *comparisons),
        assumptions=(granted,),
        interpretation=_kruskal_words(kruskal, epsilon, groups, comparisons),
        methods_fragment=_kruskal_methods(),
        counts=_row_count(used, len(pairing.excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Ranks by group",
                RANK_COLUMNS,
                (
                    (
                        group.label,
                        group.n,
                        rank,
                        summary.median,
                        summary.q1,
                        summary.q3,
                    )
                    for group, rank, summary in zip(groups, ranks, summaries, strict=True)
                ),
                note=(
                    "Mean rank is the sum of the group's ranks divided by its n, with tied "
                    "values taking their average rank. The median and the quartiles follow "
                    "NumPy's default definition (type 7)."
                ),
            ),
        ),
        plots=(
            box_plot(
                f"Box plot of {request.values.name} by {request.groups.name}",
                request.values.name,
                [(group.label, group.values) for group in groups],
            ),
        ),
    )


TOOLS.register(
    ToolSpec(
        name="anova",
        family="means",
        title="Analysis of variance",
        request=AnovaRequest,
        run=run_anova,
    )
)

TOOLS.register(
    ToolSpec(
        name="kruskal",
        family="ranks",
        title="Kruskal–Wallis test",  # noqa: RUF001
        request=KruskalRequest,
        run=run_kruskal,
    )
)
