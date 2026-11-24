"""Rank-based comparisons: the Mann-Whitney, the Wilcoxon signed-rank and the sign test.

The three tests read the order of the values, never their size, so an outlier moves a rank no
further than the next value would. The Mann-Whitney compares two independent samples and
reports the Hodges-Lehmann shift with its distribution-free interval and the rank-biserial
correlation. The Wilcoxon signed-rank compares paired differences (or one column of
differences) against a null value, leaves zero differences out as Wilcoxon's original rule
does, and reports the pseudo-median (the median of the Walsh averages) with its interval and
the matched-pairs rank-biserial correlation. The sign test counts the signs of the differences
about the null value and reports the median with the interval the sign test itself gives.

Every p-value travels with an effect size or an interval, and every interval carries the level
it was built at. A p-value is a statement about evidence, never a statement of proof: a small
one is evidence against the null, and a large one is not evidence for it. Rows a tool leaves
out are listed, never dropped without a word.
"""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.data.dataset import IntArray
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import num
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
from curvestudio.tools.stats._display import Cell, ColumnKind, FloatArray, Table, box_plot
from curvestudio.tools.stats._tables import quantiles, table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MAX_ROWS = 2_000
"""The most rows a rank tool takes: its pairwise differences or Walsh averages are quadratic."""

MANN_WHITNEY_EXACT_MAX = 8
"""Each sample may hold at most this many values for the exact Mann-Whitney distribution."""

WILCOXON_EXACT_MAX = 20
"""The most nonzero differences the exact signed-rank distribution is used for."""

SUMMARY_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("group", "Group", "text"),
    ("n", "n", "int"),
    ("median", "Median", "number"),
    ("q1", "Q1", "number"),
    ("q3", "Q3", "number"),
)
"""The columns of the summary by group, as :func:`table` takes them."""

SIGN_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("sign", "Sign", "text"),
    ("count", "Count", "int"),
)
"""The columns of the table of the signs of the differences."""

_ZERO_REASON = "no difference from the null value (left out of the signed-rank test)"


class MannWhitneyRequest(BaseModel):
    """Two independent samples, given as two columns or as one column and a group column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    a: Column = Field(description="The first sample's column, or the values a group column splits.")
    b: Column | None = Field(
        default=None,
        description="The second sample's column; leave it empty when a group column is given.",
    )
    groups: GroupColumn | None = Field(
        default=None,
        description="A group column with exactly two levels, compared as the two samples.",
    )
    level: float = Field(
        default=0.95,
        description="The confidence level for the intervals, between 0 and 1.",
    )


class WilcoxonRequest(BaseModel):
    """Paired differences in one column, or between two columns, tested against a null value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    a: Column = Field(
        description="The differences, or the first column of a pair when b is given.",
    )
    b: Column | None = Field(
        default=None,
        description="The second column of a pair; when empty, the values of a are the differences.",
    )
    null: float = Field(
        default=0.0,
        description="The null value the differences are tested against.",
        json_schema_extra={"label": "Null value"},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level for the intervals, between 0 and 1.",
    )


class SignTestRequest(BaseModel):
    """Paired differences in one column, or between two columns, tested against a null value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    a: Column = Field(
        description="The differences, or the first column of a pair when b is given.",
    )
    b: Column | None = Field(
        default=None,
        description="The second column of a pair; when empty, the values of a are the differences.",
    )
    null: float = Field(
        default=0.0,
        description="The null value the differences are tested against.",
        json_schema_extra={"label": "Null value"},
    )
    level: float = Field(
        default=0.95,
        description="The confidence level for the intervals, between 0 and 1.",
    )


class _Differences(NamedTuple):
    """The differences a paired tool reads, the positions they cover and the rows it left out."""

    values: FloatArray
    rows: IntArray
    offered: int
    excluded: tuple[ExcludedRow, ...]
    label: str


# --------------------------------------------------------------------------- shared helpers


def _percent(level: float) -> str:
    return f"{num(level * 100)} %"


def _check_rows(used: int) -> None:
    """Refuse more usable rows than a quadratic rank statistic is built for."""
    if used > MAX_ROWS:
        raise ToolError(
            "TOO_MANY_ROWS",
            f"this rank tool takes at most {MAX_ROWS:,} rows, and the data hold {used:,}",
        )


def _offered(*columns: Column) -> int:
    """The distinct positions the given columns cover."""
    return int(np.unique(np.concatenate([column.rows for column in columns])).size)


def _independent_values(column: Column) -> tuple[FloatArray, tuple[ExcludedRow, ...]]:
    """One independent column's finite values, and the positions it leaves out and why.

    Two independent samples are not a pairing: each column keeps its own rows, so neither
    sample is cut down to the rows the other one happens to cover.
    """
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


def _differences(request: WilcoxonRequest | SignTestRequest) -> _Differences:
    """The paired differences of a request, aligned and with the rows it left out.

    With two columns a position must hold a finite value in both, and the difference is the
    first column's value minus the second's. With one column its values are the differences.
    """
    if request.b is None:
        pairing = pairs(request.a)
        column = pairing.columns[0]
        values = column.values
        rows = column.rows
        label = request.a.name
        excluded = pairing.excluded
        offered = _offered(request.a)
    else:
        pairing = pairs(request.a, request.b)
        first, second = pairing.columns
        values = first.values - second.values
        rows = first.rows
        label = f"{request.a.name} - {request.b.name}"
        excluded = pairing.excluded
        offered = _offered(request.a, request.b)
    if values.size == 0:
        raise ToolError(
            "NO_DATA", f"{label} holds no pair to compare: every row is missing a value"
        )
    return _Differences(values, rows, offered, tuple(excluded), label)


def _subset_sum_counts(total: int, size: int | None) -> NDArray[np.int64]:
    """How many subsets of ``{1, ..., total}`` of ``size`` elements sum to each value.

    ``size=None`` counts the subsets of every size. Each subset of a fixed size is one rank
    assignment, so the counts are those of the exact null distribution the interval reads.
    """
    top = sum(range(total - size + 1, total + 1)) if size is not None else total * (total + 1) // 2
    if size is None:
        flat: NDArray[np.int64] = np.zeros(top + 1, dtype=np.int64)
        flat[0] = 1
        for value in range(1, total + 1):
            for running in range(top, value - 1, -1):
                flat[running] += flat[running - value]
        return flat
    grid: NDArray[np.int64] = np.zeros((size + 1, top + 1), dtype=np.int64)
    grid[0, 0] = 1
    for value in range(1, total + 1):
        for chosen in range(min(size, value), 0, -1):
            grid[chosen, value:] += grid[chosen - 1, : top + 1 - value]
    counts: NDArray[np.int64] = grid[size]
    return counts


def _mann_whitney_counts(m: int, n: int) -> NDArray[np.int64]:
    """The exact null distribution of the Mann-Whitney U for samples of ``m`` and ``n``."""
    counts = _subset_sum_counts(m + n, m)
    base = m * (m + 1) // 2
    return counts[base : base + m * n + 1]


def _signed_rank_counts(n: int) -> NDArray[np.int64]:
    """The exact null distribution of the signed-rank statistic for ``n`` differences."""
    return _subset_sum_counts(n, None)


def _critical_index(counts: NDArray[np.int64], alpha_half: float) -> int:
    """The largest index in the left tail that holds at most ``alpha_half`` of the null mass."""
    total = int(counts.sum())
    cumulative = np.cumsum(counts)
    allowed = np.flatnonzero(cumulative <= alpha_half * total)
    return int(allowed[-1]) if allowed.size else -1


def _normal_index(mean: float, sigma: float, alpha_half: float) -> int:
    """The left-tail cut of a normal approximation at ``alpha_half``, floored at 0."""
    z = float(scipy_stats.norm.ppf(1.0 - alpha_half))
    return max(0, math.floor(mean - z * sigma))


def _u_sigma(m: int, n: int, combined: FloatArray) -> float:
    """The standard deviation of the Mann-Whitney U under the null, corrected for ties."""
    total = m + n
    variance = m * n * (total + 1) / 12.0
    _unique, counts = np.unique(combined, return_counts=True)
    tied = counts[counts > 1].astype(np.float64)
    if tied.size:
        variance -= m * n * float(np.sum(tied**3 - tied)) / (12.0 * total * (total - 1))
    return math.sqrt(max(variance, 0.0))


def _signed_rank_sigma(nonzero: FloatArray) -> float:
    """The standard deviation of the signed-rank statistic under the null, corrected for ties."""
    ranks = scipy_stats.rankdata(np.abs(nonzero), method="average")
    _unique, first, counts = np.unique(np.abs(nonzero), return_index=True, return_counts=True)
    group_ranks = ranks[first]
    return math.sqrt(0.25 * float(np.sum(counts * group_ranks**2)))


def _walsh_averages(values: FloatArray) -> FloatArray:
    """Every ``(x_i + x_j) / 2`` for ``i <= j``, the values the pseudo-median is read from."""
    first, second = np.triu_indices(values.size)
    return (values[first] + values[second]) / 2.0


def _relative_effect(first: FloatArray, second: FloatArray) -> tuple[float, float | None]:
    """The relative effect ``P(first > second) + P(tie) / 2`` and its DeLong standard error."""
    m, n = first.size, second.size
    second_sorted = np.sort(second)
    left = np.searchsorted(second_sorted, first, side="left")
    right = np.searchsorted(second_sorted, first, side="right")
    below = (left + 0.5 * (right - left)) / n
    first_sorted = np.sort(first)
    left_of = np.searchsorted(first_sorted, second, side="left")
    right_of = np.searchsorted(first_sorted, second, side="right")
    above = (m - right_of + 0.5 * (right_of - left_of)) / m
    effect = float(np.mean(below))
    if m < 2 or n < 2:
        return effect, None
    spread = float(np.sum((below - effect) ** 2)) / (m * (m - 1))
    spread += float(np.sum((above - effect) ** 2)) / (n * (n - 1))
    return effect, math.sqrt(max(spread, 0.0))


def _jackknife_se(values: FloatArray, statistic: Callable[[FloatArray], float]) -> float | None:
    """The delete-one jackknife standard error of ``statistic`` over ``values``."""
    if values.size < 2:
        return None
    leave_one_out = np.array(
        [statistic(np.delete(values, index)) for index in range(values.size)], dtype=np.float64
    )
    centre = float(np.mean(leave_one_out))
    spread = float(np.sum((leave_one_out - centre) ** 2))
    return math.sqrt((values.size - 1) / values.size * spread)


def _signed_rank_correlation(shifted: FloatArray) -> float:
    """The matched-pairs rank-biserial correlation ``(T+ - T-) / (n(n + 1) / 2)``."""
    size = shifted.size
    ranks = scipy_stats.rankdata(np.abs(shifted), method="average")
    positive = float(np.sum(ranks[shifted > 0]))
    total = size * (size + 1) / 2.0
    return 2.0 * positive / total - 1.0


def _interval(value: float, se: float | None, level: float) -> tuple[float | None, float | None]:
    """The normal interval ``value +/- z * se``, clamped to ``[-1, 1]`` for a correlation."""
    if se is None:
        return None, None
    z = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    return max(-1.0, value - z * se), min(1.0, value + z * se)


def _summary_row(label: str, values: FloatArray) -> tuple[Cell, ...]:
    lower, upper = quantiles(values)
    return (label, int(values.size), float(np.median(values)), lower, upper)


def _summary_table(rows: Sequence[tuple[Cell, ...]]) -> Table:
    return table(
        "Summary by group",
        SUMMARY_COLUMNS,
        rows,
        note=(
            "The median and the quartiles follow NumPy's default definition (linear "
            "interpolation, type 7)."
        ),
    )


def _independence(pairs_or_samples: str) -> AuditCheck:
    return assumption(
        "independence",
        "Independence cannot be checked from the columns",
        Verdict.NOT_APPLICABLE,
        f"Whether the {pairs_or_samples} are independent is a fact about how the data were "
        "collected, which a column cannot record.",
        next_step=("Check how the data were collected; dependence changes what the p-value means."),
    )


# --------------------------------------------------------------------------- Mann-Whitney


def run_mannwhitney(request: BaseModel) -> ToolResult:
    """Run the Mann-Whitney comparison on its own request."""
    if not isinstance(request, MannWhitneyRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"mannwhitney takes a MannWhitneyRequest, not a request of {type(request).__name__}",
        )
    return _mann_whitney(request)


def _read_mann_whitney(
    request: MannWhitneyRequest,
) -> tuple[FloatArray, FloatArray, tuple[str, str], int, int, tuple[ExcludedRow, ...]]:
    """The two samples, their labels, the rows offered and used, and the rows left out."""
    if (request.b is None) == (request.groups is None):
        raise ToolError(
            "INVALID_OPTION",
            "give the second sample as b, or a group column as groups, but not both",
        )
    if request.groups is None:
        assert request.b is not None
        first_values, first_excluded = _independent_values(request.a)
        second_values, second_excluded = _independent_values(request.b)
        excluded_rows = tuple(
            sorted((*first_excluded, *second_excluded), key=lambda row: row.index)
        )
        return (
            first_values,
            second_values,
            (request.a.name, request.b.name),
            int(first_values.size) + int(second_values.size) + len(excluded_rows),
            int(first_values.size) + int(second_values.size),
            excluded_rows,
        )
    paired = aligned(request.a, request.groups)
    found = levels(request.groups, minimum=2)
    if len(found) != 2:
        raise ToolError(
            "LEVELS_REQUIRED",
            f"the group column {request.groups.name!r} holds {len(found)} labels, and the "
            "comparison needs exactly two",
        )
    labels_in_rows = paired.groups[0].labels
    values = paired.values.values
    first = values[labels_in_rows == found[0]]
    second = values[labels_in_rows == found[1]]
    used = int(first.size) + int(second.size)
    return first, second, (found[0], found[1]), _offered(request.a), used, paired.excluded


def _mann_whitney(request: MannWhitneyRequest) -> ToolResult:
    check_level(request.level)
    first, second, labels, offered, used, excluded_rows = _read_mann_whitney(request)
    m, n = int(first.size), int(second.size)
    if m == 0 or n == 0:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the Mann-Whitney test needs at least one value in each sample, and {labels[0]!r} "
            f"holds {m} and {labels[1]!r} holds {n}",
        )
    _check_rows(m + n)

    combined = np.concatenate([first, second])
    no_ties = np.unique(combined).size == combined.size
    exact = no_ties and max(m, n) <= MANN_WHITNEY_EXACT_MAX
    tested = scipy_stats.mannwhitneyu(
        first,
        second,
        alternative="two-sided",
        method="exact" if exact else "asymptotic",
        use_continuity=not exact,
    )
    statistic = float(tested.statistic)
    p_value = float(tested.pvalue)

    differences = np.sort((first[:, np.newaxis] - second[np.newaxis, :]).ravel())
    total = differences.size
    alpha_half = (1.0 - request.level) / 2.0
    if exact:
        cut = _critical_index(_mann_whitney_counts(m, n), alpha_half)
    else:
        cut = _normal_index(m * n / 2.0, _u_sigma(m, n, combined), alpha_half)
    cut = min(max(cut, 0), (total - 1) // 2)
    shift = Estimate(
        label="Hodges-Lehmann shift",
        value=float(np.median(differences)),
        lower=float(differences[cut]),
        upper=float(differences[total - 1 - cut]),
    )

    effect, effect_se = _relative_effect(first, second)
    correlation = 2.0 * effect - 1.0
    doubled = None if effect_se is None else 2.0 * effect_se
    lower, upper = _interval(correlation, doubled, request.level)
    rank_biserial = Estimate(
        label="Rank-biserial correlation",
        value=correlation,
        se=None if effect_se is None else 2.0 * effect_se,
        lower=lower,
        upper=upper,
    )

    method = (
        "Mann-Whitney U test, two-sided, exact"
        if exact
        else "Mann-Whitney U test, two-sided, normal approximation with ties and continuity "
        "correction"
    )
    null = "the two samples are drawn from the same distribution"
    test = HypothesisTest(
        name="Mann-Whitney U",
        method=method,
        null_hypothesis=null,
        statistic=statistic,
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=rank_biserial,
    )

    rows = [_summary_row(labels[0], first), _summary_row(labels[1], second)]
    if request.b is not None:
        plot_title = f"Box plot of {request.a.name} and {request.b.name}"
        title = f"Mann-Whitney comparison of {request.a.name} and {request.b.name}"
    else:
        assert request.groups is not None
        plot_title = f"Box plot of {request.a.name} by {request.groups.name}"
        title = f"Mann-Whitney comparison of {request.a.name} by {request.groups.name}"
    plot = box_plot(plot_title, "Value", [(labels[0], first), (labels[1], second)])

    excluded, unlisted = list_excluded(excluded_rows)
    return ToolResult(
        tool="mannwhitney",
        title=title,
        level=request.level,
        estimates=(shift, rank_biserial),
        tests=(test,),
        assumptions=(_independence("two samples"),),
        interpretation=_mann_whitney_words(labels, p_value, shift, rank_biserial, request.level),
        methods_fragment=_mann_whitney_methods(
            labels, m, n, exact, request.level, differences.size
        ),
        counts=Counts(n_offered=offered, n_used=used, n_excluded=offered - used),
        excluded=excluded,
        excluded_unlisted=unlisted,
        tables=(_summary_table(rows),),
        plots=(plot,),
    )


def _mann_whitney_words(
    labels: tuple[str, str],
    p_value: float,
    shift: Estimate,
    rank_biserial: Estimate,
    level: float,
) -> str:
    if p_value >= 0.05:
        clause = "the data are consistent with no shift between the samples"
    elif p_value >= 0.001:
        clause = "the data show evidence of a shift between the samples"
    else:
        clause = "the data show strong evidence of a shift between the samples"
    interval = (
        f"{num(shift.lower)} to {num(shift.upper)}"
        if shift.lower is not None and shift.upper is not None
        else "not able to be built"
    )
    if rank_biserial.lower is not None and rank_biserial.upper is not None:
        correlation = (
            f"{num(rank_biserial.value)} ({_percent(level)} interval "
            f"{num(rank_biserial.lower)} to {num(rank_biserial.upper)})"
        )
    else:
        correlation = f"{num(rank_biserial.value)}, with no interval from a sample of one"
    return (
        f"The Mann-Whitney test gives {p_text(p_value)}; {clause}. "
        f"The Hodges-Lehmann estimate of the shift from {labels[1]} to {labels[0]} is "
        f"{num(shift.value)}, with a {_percent(level)} interval from {interval}, and the "
        f"rank-biserial correlation is {correlation}."
    )


def _mann_whitney_methods(
    labels: tuple[str, str],
    m: int,
    n: int,
    exact: bool,
    level: float,
    differences: int,
) -> str:
    tested = (
        "the exact null distribution"
        if exact
        else "the normal approximation with ties and continuity correction"
    )
    interval_source = (
        "the exact Mann-Whitney distribution"
        if exact
        else "the normal approximation with a tie correction"
    )
    return (
        f"Two samples of {m} and {n} values were compared with the Mann-Whitney U test, "
        f"two-sided, using {tested}. The Hodges-Lehmann shift is the median of the "
        f"{differences} pairwise differences {labels[0]} - {labels[1]}, and its "
        f"{_percent(level)} interval is the distribution-free interval read from "
        f"{interval_source}. The rank-biserial correlation is 2U/(n1 n2) - 1, positive when "
        f"{labels[0]} tends to be larger, and its interval is the normal interval from the "
        "DeLong variance of the relative effect."
    )


# --------------------------------------------------------------------------- Wilcoxon


def run_wilcoxon(request: BaseModel) -> ToolResult:
    """Run the Wilcoxon signed-rank test on its own request."""
    if not isinstance(request, WilcoxonRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"wilcoxon takes a WilcoxonRequest, not a request of {type(request).__name__}",
        )
    return _wilcoxon(request)


def _wilcoxon(request: WilcoxonRequest) -> ToolResult:
    check_level(request.level)
    read = _differences(request)
    _check_rows(read.values.size)
    shifted = read.values - request.null
    zero = shifted == 0
    nonzero = shifted[~zero]
    if nonzero.size == 0:
        raise ToolError(
            "NO_DATA",
            "every difference equals the null value, and the signed-rank test needs at least "
            "one nonzero difference",
        )

    size = int(nonzero.size)
    ranks = scipy_stats.rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    statistic = min(positive, negative)
    no_ties = np.unique(np.abs(nonzero)).size == size
    exact = no_ties and size <= WILCOXON_EXACT_MAX
    tested = scipy_stats.wilcoxon(
        nonzero,
        alternative="two-sided",
        method="exact" if exact else "approx",
        correction=not exact,
        zero_method="wilcox",
    )
    p_value = float(tested.pvalue)

    walsh = np.sort(_walsh_averages(nonzero))
    total = walsh.size
    alpha_half = (1.0 - request.level) / 2.0
    if exact:
        cut = _critical_index(_signed_rank_counts(size), alpha_half)
    else:
        cut = _normal_index(size * (size + 1) / 4.0, _signed_rank_sigma(nonzero), alpha_half)
    cut = min(max(cut, 0), (total - 1) // 2)
    pseudo = Estimate(
        label="Pseudo-median of the differences",
        value=float(np.median(walsh)),
        lower=float(walsh[cut]),
        upper=float(walsh[total - 1 - cut]),
    )

    correlation = _signed_rank_correlation(nonzero)
    jackknife = _jackknife_se(nonzero, _signed_rank_correlation)
    lower, upper = _interval(correlation, jackknife, request.level)
    rank_biserial = Estimate(
        label="Matched-pairs rank-biserial correlation",
        value=correlation,
        se=jackknife,
        lower=lower,
        upper=upper,
    )

    zeros = read.values[zero]
    zero_rows = tuple(
        ExcludedRow(index=int(row), reason=_ZERO_REASON)
        for row, dropped in zip(read.rows, zero, strict=True)
        if dropped
    )
    left_out = tuple(sorted(read.excluded + zero_rows, key=lambda row: row.index))
    excluded, unlisted = list_excluded(left_out)

    method = (
        "Wilcoxon signed-rank test, two-sided, exact distribution"
        if exact
        else "Wilcoxon signed-rank test, two-sided, normal approximation with ties and "
        "continuity correction"
    )
    method += ", zero differences left out following Wilcoxon's original rule"
    test = HypothesisTest(
        name="Wilcoxon signed-rank",
        method=method,
        null_hypothesis=(f"the pseudo-median of the differences is {num(request.null)}"),
        statistic=statistic,
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=rank_biserial,
    )

    return ToolResult(
        tool="wilcoxon",
        title=f"Wilcoxon signed-rank test of {read.label}",
        level=request.level,
        estimates=(pseudo, rank_biserial),
        tests=(test,),
        assumptions=(_independence("pairs"),),
        interpretation=_wilcoxon_words(
            request.null, p_value, pseudo, rank_biserial, request.level, int(zeros.size)
        ),
        methods_fragment=_wilcoxon_methods(size, total, exact, request.level, int(zeros.size)),
        counts=Counts(n_offered=read.offered, n_used=size, n_excluded=read.offered - size),
        excluded=excluded,
        excluded_unlisted=unlisted,
    )


def _wilcoxon_words(
    null: float,
    p_value: float,
    pseudo: Estimate,
    rank_biserial: Estimate,
    level: float,
    zeros: int,
) -> str:
    if p_value >= 0.05:
        clause = f"the data are consistent with a pseudo-median of {num(null)}"
    elif p_value >= 0.001:
        clause = f"the data show evidence that the pseudo-median differs from {num(null)}"
    else:
        clause = f"the data show strong evidence that the pseudo-median differs from {num(null)}"
    interval = (
        f"{num(pseudo.lower)} to {num(pseudo.upper)}"
        if pseudo.lower is not None and pseudo.upper is not None
        else "not able to be built"
    )
    if rank_biserial.lower is not None and rank_biserial.upper is not None:
        correlation = (
            f"{num(rank_biserial.value)} ({_percent(level)} interval "
            f"{num(rank_biserial.lower)} to {num(rank_biserial.upper)})"
        )
    else:
        correlation = f"{num(rank_biserial.value)}, with no interval from a single difference"
    noun = "difference was" if zeros == 1 else "differences were"
    zero_clause = "" if zeros == 0 else f" {zeros} zero {noun} left out."
    return (
        f"The Wilcoxon signed-rank test gives {p_text(p_value)}; {clause}. "
        f"The pseudo-median of the differences is {num(pseudo.value)}, with a {_percent(level)} "
        f"interval from {interval}, and the matched-pairs rank-biserial correlation is "
        f"{correlation}.{zero_clause}"
    )


def _wilcoxon_methods(size: int, total: int, exact: bool, level: float, zeros: int) -> str:
    tested = (
        "the exact null distribution"
        if exact
        else "the normal approximation with tie and continuity corrections"
    )
    interval_source = (
        "the exact signed-rank distribution"
        if exact
        else "the normal approximation with a tie correction"
    )
    noun = "difference was" if zeros == 1 else "differences were"
    zero_sentence = "" if zeros == 0 else f" {zeros} zero {noun} left out."
    return (
        f"The differences were tested with the Wilcoxon signed-rank test, two-sided, using "
        f"{tested}, with zero differences left out following Wilcoxon's original rule. The "
        f"pseudo-median is the median of the {total} Walsh averages of the {size} nonzero "
        f"differences, and its {_percent(level)} interval is the distribution-free interval "
        f"read from {interval_source}. The matched-pairs rank-biserial correlation is "
        "(T+ - T-)/(n(n + 1)/2), and its interval is the normal interval from a delete-one "
        f"jackknife standard error.{zero_sentence}"
    )


# --------------------------------------------------------------------------- the sign test


def run_signtest(request: BaseModel) -> ToolResult:
    """Run the sign test on its own request."""
    if not isinstance(request, SignTestRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"signtest takes a SignTestRequest, not a request of {type(request).__name__}",
        )
    return _signtest(request)


def _signtest(request: SignTestRequest) -> ToolResult:
    check_level(request.level)
    read = _differences(request)
    _check_rows(read.values.size)
    shifted = read.values - request.null
    positive = int(np.count_nonzero(shifted > 0))
    negative = int(np.count_nonzero(shifted < 0))
    zeros = int(np.count_nonzero(shifted == 0))
    nonzero = positive + negative

    if nonzero == 0:
        p_value = None
        reason = "every difference equals the null value, so there is no sign to count"
    else:
        p_value = float(
            scipy_stats.binomtest(positive, nonzero, 0.5, alternative="two-sided").pvalue
        )
        reason = None

    ordered = np.sort(read.values)
    size = int(ordered.size)
    alpha_half = (1.0 - request.level) / 2.0
    admissible = [
        index
        for index in range(size)
        if float(scipy_stats.binom.cdf(index, size, 0.5)) <= alpha_half
    ]
    cut = min(admissible[-1] if admissible else 0, (size - 1) // 2)
    median = Estimate(
        label="Median of the differences",
        value=float(np.median(read.values)),
        lower=float(ordered[cut]),
        upper=float(ordered[size - 1 - cut]),
    )

    test = HypothesisTest(
        name="Sign test",
        method="Sign test, two-sided, exact binomial",
        null_hypothesis=f"the median of the differences is {num(request.null)}",
        statistic=None if nonzero == 0 else float(min(positive, negative)),
        p_value=p_value,
        p_value_text=p_text(p_value),
        effect=median,
        reason=reason,
    )

    sign_rows = (("positive", positive), ("negative", negative), ("zero", zeros))
    excluded, unlisted = list_excluded(read.excluded)
    return ToolResult(
        tool="signtest",
        title=f"Sign test of {read.label}",
        level=request.level,
        estimates=(median,),
        tests=(test,),
        assumptions=(_independence("differences"),),
        interpretation=_signtest_words(request.null, p_value, median, request.level),
        methods_fragment=_signtest_methods(request.null, request.level, nonzero),
        counts=Counts(n_offered=read.offered, n_used=size, n_excluded=read.offered - size),
        excluded=excluded,
        excluded_unlisted=unlisted,
        tables=(
            table(
                "Signs of the differences",
                SIGN_COLUMNS,
                sign_rows,
                note=(
                    "A difference is positive when it lies above the null value "
                    f"{num(request.null)}, negative when it lies below it, and zero at it."
                ),
            ),
        ),
    )


def _signtest_words(null: float, p_value: float | None, median: Estimate, level: float) -> str:
    interval = (
        f"{num(median.lower)} to {num(median.upper)}"
        if median.lower is not None and median.upper is not None
        else "not able to be built"
    )
    words = (
        f"The median of the differences is {num(median.value)}, with a {_percent(level)} "
        f"interval from {interval}."
    )
    if p_value is None:
        return (
            "The sign test could not be run: every difference equals the null value "
            f"{num(null)}. {words}"
        )
    if p_value >= 0.05:
        clause = f"the data are consistent with a median of {num(null)}"
    elif p_value >= 0.001:
        clause = f"the data show evidence that the median differs from {num(null)}"
    else:
        clause = f"the data show strong evidence that the median differs from {num(null)}"
    return f"The sign test gives {p_text(p_value)}; {clause}. {words}"


def _signtest_methods(null: float, level: float, nonzero: int) -> str:
    return (
        "The differences were tested with the sign test, two-sided, using the exact binomial "
        f"distribution of the {nonzero} nonzero signs about the null value {num(null)}. The "
        "median of the differences is reported with the distribution-free interval from the "
        f"binomial order statistics at the {_percent(level)} level. Zero differences are "
        "counted in the table and left out of the binomial test."
    )


# --------------------------------------------------------------------------- registration


TOOLS.register(
    ToolSpec(
        name="mannwhitney",
        family="ranks",
        title="Mann-Whitney U test",
        request=MannWhitneyRequest,
        run=run_mannwhitney,
        max_rows=MAX_ROWS,
    )
)
TOOLS.register(
    ToolSpec(
        name="wilcoxon",
        family="ranks",
        title="Wilcoxon signed-rank test",
        request=WilcoxonRequest,
        run=run_wilcoxon,
        max_rows=MAX_ROWS,
    )
)
TOOLS.register(
    ToolSpec(
        name="signtest",
        family="ranks",
        title="Sign test",
        request=SignTestRequest,
        run=run_signtest,
        max_rows=MAX_ROWS,
    )
)
