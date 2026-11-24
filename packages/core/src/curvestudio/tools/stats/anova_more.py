"""Two-way analysis of variance, repeated measures, and the Friedman test.

The two-way analysis of variance splits the variation of a value into the main effect of
each of two factors and their interaction. An unbalanced design is read by Type II sums of
squares: each main effect is tested after the other main effect, and the interaction after
both, so the order the factors are given in does not change the table. Partial eta squared
divides a term's sum of squares by itself plus the residual sum of squares.

The one-way repeated-measures analysis of variance compares conditions measured on the same
subjects in wide form, one column per condition. Its F test assumes the conditions are
equally correlated, a condition called sphericity; Mauchly's test checks it, and the
Greenhouse-Geisser and Huynh-Feldt corrections scale the degrees of freedom by an epsilon
when it is doubtful. Generalised eta squared says how much of the total variation the
conditions account for.

The Friedman test asks the same question of the conditions' mean ranks, ranking each
subject's conditions together and correcting for ties. Kendall's W is its effect size, and
Durbin-Conover comparisons follow when there are more than two conditions, each p-value
adjusted by Holm's method. Whether the conditions share a shape is not tested here: the box
plot is what a person reads for it.

A p-value is a statement about evidence, never a statement of proof.
"""

import math
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.diagnostics import shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import joined, num, p_threshold
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
    pairs,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    MultiLinePlot,
    XYSeries,
    box_plot,
)
from curvestudio.tools.stats._tables import confidence_interval, group_summary, holm, table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

CAUTION_P = 0.05
"""A p-value below this leaves an assumption in doubt."""

FAIL_P = 0.001
"""A p-value below this contradicts an assumption."""

CONDITION_MIN = 2
"""The fewest conditions a repeated-measures request takes."""

CONDITION_MAX = 20
"""The most conditions a repeated-measures request takes."""

FRIEDMAN_MIN = 3
"""The fewest conditions the Friedman test takes: with two it is a sign test."""

TWO_WAY_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("source", "Source", "text"),
    ("ss", "SS", "number"),
    ("df", "df", "number"),
    ("ms", "MS", "number"),
    ("f", "F", "number"),
    ("p", "p", "p_value"),
    ("partial_eta", "Partial eta squared", "number"),
)
"""The columns of the two-way analysis-of-variance table."""

CELL_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("a", "A", "text"),
    ("b", "B", "text"),
    ("n", "n", "int"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
)
"""The columns of the cell-means table."""

REPEATED_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("condition", "Condition", "text"),
    ("n", "n", "int"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
    ("se", "SE of the mean", "number"),
    ("lower", "Lower", "number"),
    ("upper", "Upper", "number"),
)
"""The columns of the summary by condition."""

MEAN_RANK_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("condition", "Condition", "text"),
    ("n", "n", "int"),
    ("mean_rank", "Mean rank", "number"),
    ("median", "Median", "number"),
    ("q1", "Q1", "number"),
    ("q3", "Q3", "number"),
)
"""The columns of the mean-ranks-by-condition table."""


class TwoWayRequest(BaseModel):
    """One two-way analysis of variance: the values and the two grouping columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column = Field(
        description="The numeric column whose mean is split into two factors and their interaction."
    )
    factor_a: GroupColumn = Field(
        description=(
            "The first factor's group column; each of its levels is a level of the factor, and "
            "at least two levels are needed."
        )
    )
    factor_b: GroupColumn = Field(
        description=(
            "The second factor's group column; each of its levels is a level of the factor, and "
            "at least two levels are needed."
        )
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class RepeatedMeasuresRequest(BaseModel):
    """One one-way repeated-measures analysis of variance, in wide form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    measures: tuple[Column, ...] = Field(
        description=(
            "The columns that hold the measurement under each condition, one column per "
            "condition; between 2 and 20 are needed, and a row missing in any of them is left "
            "out of all of them."
        )
    )
    names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "The names of the conditions, in the same order as the columns; left empty, each "
            "takes its column's name."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class FriedmanRequest(BaseModel):
    """One Friedman test, in wide form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    measures: tuple[Column, ...] = Field(
        description=(
            "The columns that hold the measurement under each condition, one column per "
            "condition; at least 3 and at most 20 are needed, and a row missing in any of them "
            "is left out of all of them."
        )
    )
    names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "The names of the conditions, in the same order as the columns; left empty, each "
            "takes its column's name."
        ),
    )
    level: float = Field(
        default=0.95,
        description="The confidence level of every interval, between 0 and 1.",
    )


class _Term(NamedTuple):
    """One term of the two-way table: its sum of squares, df, mean square, F, p and effect."""

    ss: float
    df: float
    ms: float | None
    statistic: float | None
    p_value: float | None
    partial_eta: float | None


class _TwoWaySums(NamedTuple):
    """The four parts of the two-way table and the residual mean square they are read with."""

    a: _Term
    b: _Term
    ab: _Term
    residual: _Term
    df_residual: float


class _RepeatedSums(NamedTuple):
    """The sums of squares of a one-way repeated-measures design and the F test on them."""

    conditions: float
    subjects: float
    error: float
    total: float
    df_conditions: float
    df_subjects: float
    df_error: float
    df_total: float
    ms_conditions: float
    ms_error: float
    statistic: float
    p_value: float
    generalised_eta: float
    epsilon_gg: float
    epsilon_hf: float
    p_gg: float
    p_hf: float


class _Mauchly(NamedTuple):
    """Mauchly's sphericity test: the statistic W, its chi-square, its df and its p-value."""

    w: float
    chi_square: float
    df: float
    p_value: float


class _FriedmanSums(NamedTuple):
    """The Friedman test: its statistic, df, p-value, Kendall's W and the ranks it read."""

    statistic: float
    df: float
    p_value: float
    kendall_w: float
    ranks: FloatArray
    rank_sums: FloatArray


def _verdict(p_value: float | None) -> Verdict:
    """The verdict a p-value gives an assumption."""
    if p_value is None:
        return Verdict.NOT_APPLICABLE
    if p_value < FAIL_P:
        return Verdict.FAIL
    if p_value < CAUTION_P:
        return Verdict.CAUTION
    return Verdict.PASS


def _evidence(p_value: float | None) -> str:
    """What the p-value says about the null hypothesis, as a sentence of evidence."""
    if p_value is None:
        return "The test could not be run."
    if p_value >= CAUTION_P:
        return "The data are consistent with the null hypothesis."
    if p_value >= FAIL_P:
        return "The data show evidence against the null hypothesis."
    return "The data show strong evidence against the null hypothesis."


def _normality_check(check_id: str, label: str, values: FloatArray) -> AuditCheck:
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


def _levene_check(check_id: str, groups: Sequence[tuple[str, FloatArray]]) -> AuditCheck:
    """Levene's median-centred test of equal variances over ``groups``, as a check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = scipy_stats.levene(*(values for _label, values in groups), center="median")
    statistic_value = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(statistic_value) or not math.isfinite(p_value):
        return assumption(
            check_id,
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
        else "If the spreads differ, read the p-values as an approximation; a rank-based "
        "comparison does not assume equal variances."
    )
    return assumption(
        check_id,
        titles[verdict],
        verdict,
        f"Levene's test, centred on the median, gives {statistic}, {p_text(p_value)}.",
        statistic=statistic,
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _residual(values: FloatArray, blocks: Sequence[FloatArray]) -> tuple[float, int]:
    """The residual sum of squares of a design and its rank, fitted by least squares."""
    n = int(values.size)
    design = np.column_stack([np.ones(n), *blocks])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    return float(residual @ residual), int(np.linalg.matrix_rank(design))


def _term(ss: float, df: float, ms_residual: float, df_residual: float) -> _Term:
    """One term of the two-way table, with F and partial eta squared where they are defined."""
    if df <= 0.0 or ms_residual <= 0.0 or df_residual <= 0.0:
        return _Term(ss=ss, df=df, ms=None, statistic=None, p_value=None, partial_eta=None)
    ms = ss / df
    statistic = ms / ms_residual
    p_value = float(scipy_stats.f.sf(statistic, df, df_residual))
    return _Term(
        ss=ss,
        df=df,
        ms=ms,
        statistic=statistic,
        p_value=p_value,
        partial_eta=ss / (ss + ms_residual * df_residual),
    )


def _twoway_sums(
    values: FloatArray, a_codes: NDArray[np.intp], b_codes: NDArray[np.intp], na: int, nb: int
) -> _TwoWaySums:
    """The Type II sums of squares of a two-way design.

    Each main effect is tested after the other main effect, and the interaction after both:
    the sums of squares are differences of residual sums of squares, so an unbalanced design
    is read in the same way as a balanced one.
    """
    indicators_a = np.eye(na)[a_codes]
    indicators_b = np.eye(nb)[b_codes]
    cells = np.eye(na * nb)[a_codes * nb + b_codes]
    rss_a, rank_a = _residual(values, (indicators_a,))
    rss_b, rank_b = _residual(values, (indicators_b,))
    rss_additive, rank_additive = _residual(values, (indicators_a, indicators_b))
    rss_full, rank_full = _residual(values, (cells,))
    ss_a = max(0.0, rss_b - rss_additive)
    ss_b = max(0.0, rss_a - rss_additive)
    ss_ab = max(0.0, rss_additive - rss_full)
    ss_residual = rss_full
    df_a = float(rank_additive - rank_b)
    df_b = float(rank_additive - rank_a)
    df_ab = float(rank_full - rank_additive)
    df_residual = float(values.size - rank_full)
    ms_residual = ss_residual / df_residual if df_residual > 0.0 else 0.0
    return _TwoWaySums(
        a=_term(ss_a, df_a, ms_residual, df_residual),
        b=_term(ss_b, df_b, ms_residual, df_residual),
        ab=_term(ss_ab, df_ab, ms_residual, df_residual),
        residual=_Term(
            ss=ss_residual,
            df=df_residual,
            ms=ms_residual if df_residual > 0.0 else None,
            statistic=None,
            p_value=None,
            partial_eta=None,
        ),
        df_residual=df_residual,
    )


def _codes(labels: NDArray[np.object_], order: Sequence[str]) -> NDArray[np.intp]:
    """Each label's position in ``order``, as an integer code."""
    position = {label: index for index, label in enumerate(order)}
    return np.array([position[str(label)] for label in labels], dtype=np.intp)


def _cell_values(
    values: FloatArray, a_codes: NDArray[np.intp], b_codes: NDArray[np.intp], na: int, nb: int
) -> list[list[FloatArray]]:
    """The values of each cell, laid out as ``values[i][j]`` for the levels of A and B."""
    cells: list[list[FloatArray]] = []
    for i in range(na):
        row: list[FloatArray] = []
        for j in range(nb):
            row.append(values[(a_codes == i) & (b_codes == j)])
        cells.append(row)
    return cells


def _fitted_cells(
    values: FloatArray, a_codes: NDArray[np.intp], b_codes: NDArray[np.intp], na: int, nb: int
) -> FloatArray:
    """Each value's cell mean, the fitted value of the full two-way model."""
    size = na * nb
    combined = a_codes * nb + b_codes
    totals = np.bincount(combined, weights=values, minlength=size)
    counts = np.bincount(combined, minlength=size)
    means: FloatArray = np.array(
        [totals[at] / counts[at] if counts[at] else 0.0 for at in range(size)], dtype=np.float64
    )
    return np.asarray(means[combined], dtype=np.float64)


def _empty_cells(
    cells: Sequence[Sequence[FloatArray]],
    a_labels: Sequence[str],
    b_labels: Sequence[str],
    factor_a: str,
    factor_b: str,
) -> tuple[str, ...]:
    """A warning for every cell that holds no value."""
    found: list[str] = []
    for i, row in enumerate(cells):
        for j, values in enumerate(row):
            if values.size == 0:
                found.append(
                    f"The cell {a_labels[i]!r} of {factor_a} and {b_labels[j]!r} of {factor_b} "
                    "holds no value, so its mean is undefined."
                )
    return tuple(found)


def _twoway_term_test(
    term: _Term, df_residual: float, name: str, effect_name: str, null_hypothesis: str
) -> HypothesisTest:
    """One hypothesis test of the two-way table, with its partial eta squared as its effect."""
    effect = (
        None
        if term.partial_eta is None
        else Estimate(label=f"Partial eta squared, {effect_name}", value=term.partial_eta)
    )
    reason = None
    if term.p_value is None:
        reason = (
            "the design leaves no degree of freedom for this term, or no residual mean square "
            "to test it against"
        )
    return HypothesisTest(
        name=name,
        method="Two-way analysis of variance, Type II F test",
        null_hypothesis=null_hypothesis,
        statistic=term.statistic,
        df=term.df,
        df_denominator=df_residual,
        p_value=term.p_value,
        p_value_text=p_text(term.p_value),
        effect=effect,
        reason=reason,
    )


def _twoway_words(request: TwoWayRequest, sums: _TwoWaySums) -> str:
    """One to three sentences: the three F tests, the effect sizes and what they say."""
    df_residual = num(sums.df_residual)

    def clause(term: _Term) -> str:
        if term.p_value is None or term.statistic is None:
            return "could not be tested"
        return (
            f"F = {num(term.statistic)} on {num(term.df)} and {df_residual} degrees of "
            f"freedom, {p_text(term.p_value)}"
        )

    first = (
        f"The two-way analysis of variance gives, for {request.factor_a.name}, "
        f"{clause(sums.a)}; for {request.factor_b.name}, {clause(sums.b)}; and for their "
        f"interaction, {clause(sums.ab)}."
    )
    effects = [
        f"{num(term.partial_eta)} for {label}"
        for term, label in (
            (sums.a, request.factor_a.name),
            (sums.b, request.factor_b.name),
            (sums.ab, "the interaction"),
        )
        if term.partial_eta is not None
    ]
    second = "Partial eta squared is " + joined(effects) + "." if effects else ""
    verdict = _evidence(sums.ab.p_value)
    verdict = f"For the interaction, {verdict[:1].lower()}{verdict[1:]}"
    return f"{first} {second} {verdict}".strip()


def _twoway_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "Two-way analysis of variance splits the variation into the main effect of each factor "
        "and their interaction, and the F test of each term divides its mean square by the "
        "residual mean square. Type II sums of squares test each main effect after the other "
        "main effect and the interaction after both, so an unbalanced design is read the same "
        "way as a balanced one. Partial eta squared divides a term's sum of squares by that sum "
        "plus the residual sum of squares. Shapiro-Wilk's W test checks normality of the "
        "residuals, and Levene's test centred on the median checks equal variances across the "
        "cells."
    )


def _twoway_table(sums: _TwoWaySums, a_label: str, b_label: str) -> tuple[tuple[Cell, ...], ...]:
    """The rows of the two-way table, from the sums of squares."""
    rows: list[tuple[Cell, ...]] = []
    for label, term in (
        (a_label, sums.a),
        (b_label, sums.b),
        (f"{a_label} × {b_label}", sums.ab),  # noqa: RUF001
    ):
        rows.append(
            (label, term.ss, term.df, term.ms, term.statistic, term.p_value, term.partial_eta)
        )
    residual = sums.residual
    rows.append(("Residual", residual.ss, residual.df, residual.ms, None, None, None))
    return tuple(rows)


def _twoway_plot(
    request: TwoWayRequest,
    a_labels: Sequence[str],
    b_labels: Sequence[str],
    cells: Sequence[Sequence[FloatArray]],
) -> MultiLinePlot:
    """The interaction plot: one line per level of B, the cell mean at each level of A."""
    series: list[XYSeries] = []
    for j, b_label in enumerate(b_labels):
        xs: list[float] = []
        ys: list[float] = []
        for i, _a_label in enumerate(a_labels):
            values = cells[i][j]
            if values.size:
                xs.append(float(i))
                ys.append(float(np.mean(values)))
        series.append(XYSeries(label=b_label, x=tuple(xs), y=tuple(ys)))
    return MultiLinePlot(
        title=f"Interaction plot of {request.values.name}",
        x_label=request.factor_a.name,
        y_label=request.values.name,
        series=tuple(series),
        x_labels=tuple(a_labels),
    )


def _counts(used: int, excluded: int) -> Counts:
    """The counts a result carries: what was offered, used and left out."""
    return Counts(n_offered=used + excluded, n_used=used, n_excluded=excluded)


def run_twoway(request: BaseModel) -> ToolResult:
    """Run the two-way analysis of variance on its own request."""
    if not isinstance(request, TwoWayRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"twoway takes a TwoWayRequest, not a request of {type(request).__name__}",
        )
    return _analyse_twoway(request)


def _analyse_twoway(request: TwoWayRequest) -> ToolResult:
    """The two-way analysis itself, once the request is known to be this tool's own."""
    check_level(request.level)
    pairing = aligned(request.values, request.factor_a, request.factor_b)
    a_labels = levels(pairing.groups[0], minimum=2)
    b_labels = levels(pairing.groups[1], minimum=2)
    a_codes = _codes(pairing.groups[0].labels, a_labels)
    b_codes = _codes(pairing.groups[1].labels, b_labels)
    values = pairing.values.values
    sums = _twoway_sums(values, a_codes, b_codes, len(a_labels), len(b_labels))
    if sums.df_residual < 1.0:
        raise ToolError(
            "TOO_FEW_ROWS",
            "the design fills every cell once, so no row is left for the residual and no F "
            "test can be made",
        )
    if sums.residual.ms is None or sums.residual.ms <= 0.0:
        raise ToolError(
            "CONSTANT_COLUMN",
            "the values do not vary inside every cell, so the residual mean square is zero and "
            "the F tests are undefined",
        )
    cells = _cell_values(values, a_codes, b_codes, len(a_labels), len(b_labels))
    tests = (
        _twoway_term_test(
            sums.a,
            sums.df_residual,
            f"Main effect of {request.factor_a.name}",
            request.factor_a.name,
            f"The mean of {request.values.name} is equal across the levels of "
            f"{request.factor_a.name}.",
        ),
        _twoway_term_test(
            sums.b,
            sums.df_residual,
            f"Main effect of {request.factor_b.name}",
            request.factor_b.name,
            f"The mean of {request.values.name} is equal across the levels of "
            f"{request.factor_b.name}.",
        ),
        _twoway_term_test(
            sums.ab,
            sums.df_residual,
            f"Interaction of {request.factor_a.name} and {request.factor_b.name}",
            f"the interaction of {request.factor_a.name} and {request.factor_b.name}",
            f"There is no interaction between {request.factor_a.name} and {request.factor_b.name}.",
        ),
    )
    residuals = values - _fitted_cells(values, a_codes, b_codes, len(a_labels), len(b_labels))
    non_empty = [
        (b_labels[j], cells[i][j])
        for i in range(len(a_labels))
        for j in range(len(b_labels))
        if cells[i][j].size
    ]
    effect_estimates = tuple(
        Estimate(label=test.effect.label, value=test.effect.value)
        for test in tests
        if test.effect is not None
    )
    listed, unlisted = list_excluded(pairing.excluded)
    summaries = [
        (
            a_label,
            b_label,
            cells[i][j].size,
            float(np.mean(cells[i][j])) if cells[i][j].size else None,
            float(np.std(cells[i][j], ddof=1)) if cells[i][j].size >= 2 else None,
        )
        for i, a_label in enumerate(a_labels)
        for j, b_label in enumerate(b_labels)
    ]
    return ToolResult(
        tool="twoway",
        title=(
            f"Two-way analysis of variance of {request.values.name} by "
            f"{request.factor_a.name} and {request.factor_b.name}"
        ),
        level=request.level,
        estimates=effect_estimates,
        tests=tests,
        assumptions=(
            _normality_check("normality_of_residuals", "the residuals", residuals),
            _levene_check("equal_variances", non_empty),
        ),
        warnings=_empty_cells(
            cells, a_labels, b_labels, request.factor_a.name, request.factor_b.name
        ),
        interpretation=_twoway_words(request, sums),
        methods_fragment=_twoway_methods(),
        counts=_counts(int(values.size), len(pairing.excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Two-way analysis of variance (Type II)",
                TWO_WAY_COLUMNS,
                _twoway_table(sums, request.factor_a.name, request.factor_b.name),
                note=(
                    "SS is the sum of squares, MS the mean square and df the degrees of "
                    "freedom. Each term is tested by a Type II sum of squares: a main effect "
                    "after the other main effect, and the interaction after both. Partial eta "
                    "squared divides a term's sum of squares by that sum plus the residual sum "
                    "of squares."
                ),
            ),
            table(
                "Cell means",
                CELL_COLUMNS,
                summaries,
                note=(
                    f"One row per cell of {request.factor_a.name} and {request.factor_b.name}; "
                    "SD is the sample standard deviation, with n - 1 in its denominator."
                ),
            ),
        ),
        plots=(_twoway_plot(request, a_labels, b_labels, cells),),
    )


def _contrast_covariance(values: FloatArray) -> FloatArray:
    """The covariance matrix of the orthonormal contrasts of the conditions.

    Sphericity and the epsilons concern the differences between the conditions, so they are
    read from this ``(k - 1) x (k - 1)`` matrix and not from the full ``k x k`` one.
    """
    conditions = int(values.shape[1])
    covariance = np.cov(values, rowvar=False, ddof=1)
    contrasts = np.zeros((conditions, conditions - 1))
    for j in range(conditions - 1):
        contrasts[j, j] = 1.0
        contrasts[conditions - 1, j] = -1.0
    basis = np.linalg.qr(contrasts)[0]
    return basis.T @ covariance @ basis


def _gg_epsilon(contrast: FloatArray) -> float:
    """The Greenhouse-Geisser epsilon of a contrast covariance matrix, at most one."""
    dimension = int(contrast.shape[0])
    if dimension <= 1:
        return 1.0
    spread = float(np.trace(contrast @ contrast))
    if spread <= 0.0:
        return 1.0
    epsilon = float(np.trace(contrast) ** 2 / (dimension * spread))
    return min(epsilon, 1.0)


def _hf_epsilon(epsilon_gg: float, subjects: int, dimension: int) -> float:
    """The Huynh-Feldt epsilon from the Greenhouse-Geisser one, at most one.

    A value that falls below the lower bound ``1 / dimension`` is set to it, so the corrected
    degrees of freedom stay positive on a very small sample.
    """
    if dimension <= 0:
        return 1.0
    denominator = dimension * (subjects - 1 - dimension * epsilon_gg)
    if denominator <= 0.0:
        return 1.0
    epsilon = (subjects * dimension * epsilon_gg - 2.0) / denominator
    return min(max(epsilon, 1.0 / dimension), 1.0)


def _mauchly(contrast: FloatArray, df_residual: int, conditions: int) -> _Mauchly:
    """Mauchly's sphericity test from the contrast covariance matrix.

    The chi-square approximation and its second-order correction follow R's ``mauchly.test``
    (Mauchly 1940; Anderson 2003), as the published implementations compute them.
    """
    dimension = int(contrast.shape[0])
    sign, log_determinant = np.linalg.slogdet(contrast)
    log_w = (
        log_determinant - dimension * math.log(float(np.trace(contrast)) / dimension)
        if sign > 0
        else -math.inf
    )
    w = math.exp(log_w)
    correction = 1.0 - (2.0 * dimension**2 + dimension + 2.0) / (6.0 * dimension * df_residual)
    second_order = (
        (dimension + 2.0)
        * (dimension - 1.0)
        * (dimension - 2.0)
        * (2.0 * dimension**3 + 6.0 * dimension**2 + 3.0 * conditions + 2.0)
        / (288.0 * (df_residual * dimension * correction) ** 2)
    )
    chi_square = -df_residual * correction * log_w
    degrees = dimension * (dimension + 1) / 2.0 - 1.0
    first = float(scipy_stats.chi2.sf(chi_square, degrees))
    second = float(scipy_stats.chi2.sf(chi_square, degrees + 4.0))
    return _Mauchly(
        w=w,
        chi_square=chi_square,
        df=degrees,
        p_value=first + second_order * (second - first),
    )


def _rm_sums(values: FloatArray) -> _RepeatedSums:
    """The sums of squares of a one-way repeated-measures design, its F test and epsilons."""
    subjects, conditions = (int(size) for size in values.shape)
    grand = float(np.mean(values))
    subject_means = np.mean(values, axis=1)
    condition_means = np.mean(values, axis=0)
    ss_total = float(np.sum((values - grand) ** 2))
    ss_conditions = float(subjects * np.sum((condition_means - grand) ** 2))
    ss_subjects = float(conditions * np.sum((subject_means - grand) ** 2))
    residuals = values - condition_means[None, :] - subject_means[:, None] + grand
    ss_error = float(np.sum(residuals**2))
    df_conditions = float(conditions - 1)
    df_subjects = float(subjects - 1)
    df_error = float((subjects - 1) * (conditions - 1))
    df_total = float(subjects * conditions - 1)
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    statistic = ms_conditions / ms_error
    p_value = float(scipy_stats.f.sf(statistic, df_conditions, df_error))
    contrast = _contrast_covariance(values)
    epsilon_gg = _gg_epsilon(contrast)
    epsilon_hf = _hf_epsilon(epsilon_gg, subjects, conditions - 1)
    return _RepeatedSums(
        conditions=ss_conditions,
        subjects=ss_subjects,
        error=ss_error,
        total=ss_total,
        df_conditions=df_conditions,
        df_subjects=df_subjects,
        df_error=df_error,
        df_total=df_total,
        ms_conditions=ms_conditions,
        ms_error=ms_error,
        statistic=statistic,
        p_value=p_value,
        generalised_eta=ss_conditions / ss_total,
        epsilon_gg=epsilon_gg,
        epsilon_hf=epsilon_hf,
        p_gg=float(scipy_stats.f.sf(statistic, epsilon_gg * df_conditions, epsilon_gg * df_error)),
        p_hf=float(scipy_stats.f.sf(statistic, epsilon_hf * df_conditions, epsilon_hf * df_error)),
    )


def _conditions(
    request: RepeatedMeasuresRequest | FriedmanRequest, minimum: int
) -> tuple[tuple[str, ...], tuple[Column, ...]]:
    """The condition names and columns of a wide-form request, checked for the tool.

    Raises:
        ToolError: ``INVALID_OPTION`` for a count of conditions outside the tool's range, or
            names that do not match the columns.
    """
    measures = request.measures
    if not minimum <= len(measures) <= CONDITION_MAX:
        raise ToolError(
            "INVALID_OPTION",
            f"the request holds {len(measures)} condition columns, and this tool takes between "
            f"{minimum} and {CONDITION_MAX}",
        )
    names = request.names
    if names is not None and len(names) != len(measures):
        raise ToolError(
            "INVALID_OPTION",
            f"names holds {len(names)} names for {len(measures)} condition columns",
        )
    labels = tuple(names) if names is not None else tuple(column.name for column in measures)
    return labels, measures


def _sphericity_check(values: FloatArray) -> AuditCheck:
    """Mauchly's test of sphericity, or why it does not apply."""
    subjects, conditions = (int(size) for size in values.shape)
    dimension = conditions - 1
    if conditions < 3:
        return assumption(
            "sphericity",
            "Sphericity does not apply to two conditions",
            Verdict.NOT_APPLICABLE,
            "With two conditions the sphericity assumption always holds, and no test is needed.",
        )
    if subjects - 1 < dimension:
        return assumption(
            "sphericity",
            "Mauchly's test could not be read from these data",
            Verdict.NOT_APPLICABLE,
            "The covariance matrix of the contrasts needs more subjects than conditions, and "
            "these data have too few.",
        )
    contrast = _contrast_covariance(values)
    found = _mauchly(contrast, subjects - 1, conditions)
    if not math.isfinite(found.chi_square) or not math.isfinite(found.p_value):
        return assumption(
            "sphericity",
            "Mauchly's test could not be read from these data",
            Verdict.NOT_APPLICABLE,
            "The covariance matrix of the contrasts is singular, so the test has no statistic.",
        )
    verdict = _verdict(found.p_value)
    titles = {
        Verdict.PASS: "The conditions are consistent with sphericity",
        Verdict.CAUTION: "Sphericity is in doubt",
        Verdict.FAIL: "The conditions show evidence against sphericity",
    }
    statistic = f"W = {num(found.w)}"
    next_step = (
        None
        if verdict is Verdict.PASS
        else "Read the Greenhouse-Geisser or Huynh-Feldt corrected test, which does not assume "
        "sphericity."
    )
    return assumption(
        "sphericity",
        titles[verdict],
        verdict,
        (
            f"Mauchly's test gives {statistic}, chi-square = {num(found.chi_square)} on "
            f"{num(found.df)} degrees of freedom, {p_text(found.p_value)}."
        ),
        statistic=statistic,
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _repeated_words(sums: _RepeatedSums) -> str:
    """One to three sentences: the F test, the corrections and the effect size."""
    first = (
        f"The one-way repeated-measures analysis of variance gives F = {num(sums.statistic)} on "
        f"{num(sums.df_conditions)} and {num(sums.df_error)} degrees of freedom, "
        f"{p_text(sums.p_value)}; generalised eta squared is {num(sums.generalised_eta)}."
    )
    second = (
        f"With the Greenhouse-Geisser correction (epsilon = {num(sums.epsilon_gg)}) the p-value "
        f"is {p_text(sums.p_gg)}, and with the Huynh-Feldt correction (epsilon = "
        f"{num(sums.epsilon_hf)}) it is {p_text(sums.p_hf)}."
    )
    return f"{first} {second} {_evidence(sums.p_value)}"


def _repeated_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "One-way repeated-measures analysis of variance splits the variation into the part "
        "between the conditions, the part between the subjects and the residual, and the F test "
        "divides the conditions' mean square by the residual mean square on k - 1 and "
        "(n - 1)(k - 1) degrees of freedom. Generalised eta squared divides the conditions' sum "
        "of squares by the total sum of squares. Mauchly's test checks sphericity, and the "
        "Greenhouse-Geisser and Huynh-Feldt epsilons scale both degrees of freedom when it "
        "fails. Shapiro-Wilk's W test checks normality of the residuals."
    )


def _condition_summary(
    labels: Sequence[str], values: FloatArray, level: float
) -> list[tuple[Cell, ...]]:
    """One summary row per condition: its n, mean, SD, SE of the mean and interval."""
    rows: list[tuple[Cell, ...]] = []
    for j, label in enumerate(labels):
        column = values[:, j]
        summary = group_summary(label, column)
        lower = upper = None
        if summary.se is not None and summary.mean is not None:
            lower, upper = confidence_interval(
                summary.mean, summary.se, float(summary.count - 1), level
            )
        rows.append(
            (
                label,
                summary.count,
                summary.mean,
                summary.sd,
                summary.se,
                lower,
                upper,
            )
        )
    return rows


def run_repeated(request: BaseModel) -> ToolResult:
    """Run the repeated-measures analysis of variance on its own request."""
    if not isinstance(request, RepeatedMeasuresRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"repeated takes a RepeatedMeasuresRequest, not a request of {type(request).__name__}",
        )
    return _analyse_repeated(request)


def _analyse_repeated(request: RepeatedMeasuresRequest) -> ToolResult:
    """The repeated-measures analysis itself, once the request is this tool's own."""
    check_level(request.level)
    labels, measures = _conditions(request, CONDITION_MIN)
    pairing = pairs(*measures)
    values = np.column_stack([column.values for column in pairing.columns])
    subjects = int(values.shape[0])
    if subjects < 2:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the repeated-measures analysis needs at least 2 complete subjects, and it has "
            f"{subjects}",
        )
    sums = _rm_sums(values)
    if not math.isfinite(sums.statistic) or sums.ms_error <= 0.0:
        raise ToolError(
            "CONSTANT_COLUMN",
            "the values do not vary within the subjects, so the residual mean square is zero "
            "and the F test is undefined",
        )
    effect = Estimate(label="Generalised eta squared", value=sums.generalised_eta)
    tests = (
        HypothesisTest(
            name="One-way repeated-measures analysis of variance, F test",
            method="One-way repeated-measures analysis of variance, F test",
            null_hypothesis="The conditions have equal means.",
            statistic=sums.statistic,
            df=sums.df_conditions,
            df_denominator=sums.df_error,
            p_value=sums.p_value,
            p_value_text=p_text(sums.p_value),
            effect=effect,
        ),
        HypothesisTest(
            name=(
                "One-way repeated-measures analysis of variance with the "
                "Greenhouse–Geisser correction"  # noqa: RUF001
            ),
            method=(
                "One-way repeated-measures analysis of variance, F test with the "
                f"Greenhouse–Geisser correction, epsilon = {num(sums.epsilon_gg)}"  # noqa: RUF001
            ),
            null_hypothesis="The conditions have equal means.",
            statistic=sums.statistic,
            df=sums.epsilon_gg * sums.df_conditions,
            df_denominator=sums.epsilon_gg * sums.df_error,
            p_value=sums.p_gg,
            p_value_text=p_text(sums.p_gg),
            effect=effect,
        ),
        HypothesisTest(
            name=(
                "One-way repeated-measures analysis of variance with the Huynh–Feldt correction"  # noqa: RUF001
            ),
            method=(
                "One-way repeated-measures analysis of variance, F test with the "
                f"Huynh–Feldt correction, epsilon = {num(sums.epsilon_hf)}"  # noqa: RUF001
            ),
            null_hypothesis="The conditions have equal means.",
            statistic=sums.statistic,
            df=sums.epsilon_hf * sums.df_conditions,
            df_denominator=sums.epsilon_hf * sums.df_error,
            p_value=sums.p_hf,
            p_value_text=p_text(sums.p_hf),
            effect=effect,
        ),
    )
    residuals = (
        values
        - np.mean(values, axis=0)[None, :]
        - np.mean(values, axis=1)[:, None]
        + float(np.mean(values))
    )
    listed, unlisted = list_excluded(pairing.excluded)
    return ToolResult(
        tool="repeated",
        title=f"One-way repeated-measures analysis of variance of {joined(labels)}",
        level=request.level,
        estimates=(effect,),
        tests=tests,
        assumptions=(
            _sphericity_check(values),
            _normality_check("normality_of_residuals", "the residuals", residuals.reshape(-1)),
        ),
        interpretation=_repeated_words(sums),
        methods_fragment=_repeated_methods(),
        counts=_counts(subjects, len(pairing.excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Repeated-measures analysis of variance",
                TWO_WAY_COLUMNS[:6],
                (
                    (
                        "Conditions",
                        sums.conditions,
                        sums.df_conditions,
                        sums.ms_conditions,
                        sums.statistic,
                        sums.p_value,
                    ),
                    ("Subjects", sums.subjects, sums.df_subjects, None, None, None),
                    ("Error", sums.error, sums.df_error, sums.ms_error, None, None),
                    ("Total", sums.total, sums.df_total, None, None, None),
                ),
                note=(
                    "SS is the sum of squares, MS the mean square and df the degrees of "
                    "freedom. The conditions' mean square is divided by the residual mean "
                    "square for the F test. The total is the corrected total."
                ),
            ),
            table(
                "Summary by condition",
                REPEATED_COLUMNS,
                _condition_summary(labels, values, request.level),
                note=(
                    "SD is the sample standard deviation, with n - 1 in its denominator; the "
                    "SE of the mean is SD divided by the square root of n. The interval is the "
                    "mean plus or minus t times the SE at the requested confidence level, with "
                    "t on n - 1 degrees of freedom."
                ),
            ),
        ),
        plots=(
            box_plot(
                "Box plot of the measurements by condition",
                "Measurement",
                [(label, values[:, j]) for j, label in enumerate(labels)],
            ),
        ),
    )


def _friedman_sums(values: FloatArray) -> _FriedmanSums:
    """The Friedman statistic with its tie correction, Kendall's W and the ranks."""
    subjects, conditions = (int(size) for size in values.shape)
    ranks = scipy_stats.rankdata(values, axis=1, method="average")
    rank_sums = np.sum(ranks, axis=0)
    ties = 0.0
    for row in values:
        counts = np.unique(row, return_counts=True)[1]
        ties += float(np.sum(counts.astype(np.float64) ** 3 - counts))
    correction = 1.0 - ties / (subjects * (conditions**3 - conditions))
    statistic = 12.0 / (subjects * conditions * (conditions + 1)) * float(np.sum(rank_sums**2))
    statistic -= 3.0 * subjects * (conditions + 1)
    if correction <= 0.0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = statistic / correction
        p_value = float(scipy_stats.chi2.sf(statistic, conditions - 1))
    return _FriedmanSums(
        statistic=statistic,
        df=float(conditions - 1),
        p_value=p_value,
        kendall_w=statistic / (subjects * (conditions - 1)),
        ranks=ranks,
        rank_sums=rank_sums,
    )


def _durbin_conover(
    sums: _FriedmanSums, labels: Sequence[str], subjects: int, level: float
) -> tuple[HypothesisTest, ...]:
    """One Durbin-Conover comparison per pair of conditions, adjusted by Holm's method."""
    conditions = len(labels)
    ranks = sums.ranks
    rank_sums = sums.rank_sums
    total_squares = float(np.sum(ranks**2))
    centre = conditions * subjects * (conditions + 1) ** 2 / 4.0
    variance = (total_squares - centre) / (conditions - 1)
    if variance <= 0.0:
        return ()
    t2 = float(np.sum((rank_sums - subjects * (conditions + 1) / 2.0) ** 2)) / variance
    df = float((subjects - 1) * (conditions - 1))
    scale_a = variance * 2.0 * subjects * (conditions - 1) / df
    scale_b = 1.0 - t2 / (subjects * (conditions - 1))
    if scale_a <= 0.0 or scale_b <= 0.0:
        return ()
    denominator = math.sqrt(scale_a) * math.sqrt(scale_b)
    critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    tests: list[HypothesisTest] = []
    for first in range(conditions):
        for second in range(first + 1, conditions):
            difference = (float(rank_sums[first]) - float(rank_sums[second])) / subjects
            statistic = abs(difference) * subjects / denominator
            p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), df))
            se = denominator / subjects
            tests.append(
                HypothesisTest(
                    name=(
                        f"Durbin–Conover test, {labels[first]} less {labels[second]}"  # noqa: RUF001
                    ),
                    method="Durbin–Conover test, two-sided, with Holm's adjustment",  # noqa: RUF001
                    null_hypothesis=(
                        f"The mean ranks of {labels[first]} and {labels[second]} are equal."
                    ),
                    statistic=statistic,
                    df=df,
                    p_value=p_value,
                    p_value_text=p_text(p_value),
                    effect=Estimate(
                        label=(f"Difference of mean ranks, {labels[first]} less {labels[second]}"),
                        value=difference,
                        se=se,
                        lower=difference - critical * se,
                        upper=difference + critical * se,
                    ),
                )
            )
    return holm(tests)


def _friedman_words(sums: _FriedmanSums, comparisons: Sequence[HypothesisTest]) -> str:
    """One to three sentences for the Friedman test."""
    first = (
        f"The Friedman test gives chi-square = {num(sums.statistic)} on {num(sums.df)} degrees "
        f"of freedom, {p_text(sums.p_value)}; Kendall's W is {num(sums.kendall_w)}."
    )
    if comparisons:
        rejected = sum(
            1 for test in comparisons if test.adjusted_p is not None and test.adjusted_p < CAUTION_P
        )
        smallest = min(test.adjusted_p for test in comparisons if test.adjusted_p is not None)
        verb = "has" if rejected == 1 else "have"
        second = (
            f"Durbin–Conover tests adjust the {len(comparisons)} comparisons by Holm's "  # noqa: RUF001
            f"method: {rejected} of them {verb} an adjusted p below {num(CAUTION_P)}, the "
            f"smallest {p_text(smallest)}."
        )
        return f"{first} {second}"
    return f"{first} {_evidence(sums.p_value)}"


def _friedman_methods() -> str:
    """One fragment a methods paragraph can carry as it stands."""
    return (
        "The Friedman test ranks each subject's conditions together, with tied values taking "
        "their average rank, and compares the conditions' rank sums with a chi-square "
        "approximation on k - 1 degrees of freedom, corrected for ties. Kendall's W divides the "
        "statistic by n(k - 1). Durbin–Conover tests compare each pair's rank sums with a t "  # noqa: RUF001
        "approximation, and Holm's method adjusts their p-values. Whether the conditions share "
        "a shape is not tested here; the box plot shows it."
    )


def _mean_rank_rows(
    labels: Sequence[str], values: FloatArray, sums: _FriedmanSums
) -> list[tuple[Cell, ...]]:
    """One row per condition: its n, mean rank, median and quartiles."""
    rows: list[tuple[Cell, ...]] = []
    for j, label in enumerate(labels):
        column = values[:, j]
        summary = group_summary(label, column)
        rows.append(
            (
                label,
                summary.count,
                float(sums.rank_sums[j]) / int(values.shape[0]),
                summary.median,
                summary.q1,
                summary.q3,
            )
        )
    return rows


def run_friedman(request: BaseModel) -> ToolResult:
    """Run the Friedman test on its own request."""
    if not isinstance(request, FriedmanRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"friedman takes a FriedmanRequest, not a request of {type(request).__name__}",
        )
    return _analyse_friedman(request)


def _analyse_friedman(request: FriedmanRequest) -> ToolResult:
    """The Friedman test itself, once the request is known to be this tool's own."""
    check_level(request.level)
    labels, measures = _conditions(request, FRIEDMAN_MIN)
    pairing = pairs(*measures)
    values = np.column_stack([column.values for column in pairing.columns])
    subjects = int(values.shape[0])
    if subjects < 2:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the Friedman test needs at least 2 complete subjects, and it has {subjects}",
        )
    sums = _friedman_sums(values)
    comparisons = _durbin_conover(sums, labels, subjects, request.level)
    effect = Estimate(label="Kendall's W", value=sums.kendall_w)
    test = HypothesisTest(
        name="Friedman test",
        method="Friedman test, chi-square approximation with ties",
        null_hypothesis="The conditions have equal mean ranks.",
        statistic=sums.statistic,
        df=sums.df,
        p_value=sums.p_value,
        p_value_text=p_text(sums.p_value),
        effect=effect,
    )
    listed, unlisted = list_excluded(pairing.excluded)
    return ToolResult(
        tool="friedman",
        title=f"Friedman test of {joined(labels)}",
        level=request.level,
        estimates=(effect,),
        tests=(test, *comparisons),
        interpretation=_friedman_words(sums, comparisons),
        methods_fragment=_friedman_methods(),
        counts=_counts(subjects, len(pairing.excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Mean ranks by condition",
                MEAN_RANK_COLUMNS,
                _mean_rank_rows(labels, values, sums),
                note=(
                    "Mean rank is the sum of the condition's within-subject ranks divided by n, "
                    "with tied values taking their average rank. The median and the quartiles "
                    "follow NumPy's default definition (type 7)."
                ),
            ),
        ),
        plots=(
            box_plot(
                "Box plot of the measurements by condition",
                "Measurement",
                [(label, values[:, j]) for j, label in enumerate(labels)],
            ),
        ),
    )


TOOLS.register(
    ToolSpec(
        name="twoway",
        family="means",
        title="Two-way analysis of variance",
        request=TwoWayRequest,
        run=run_twoway,
    )
)

TOOLS.register(
    ToolSpec(
        name="repeated",
        family="means",
        title="Repeated-measures analysis of variance",
        request=RepeatedMeasuresRequest,
        run=run_repeated,
    )
)

TOOLS.register(
    ToolSpec(
        name="friedman",
        family="ranks",
        title="Friedman test",
        request=FriedmanRequest,
        run=run_friedman,
    )
)
